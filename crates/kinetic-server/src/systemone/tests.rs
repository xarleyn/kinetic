use super::*;
use std::io::BufReader;
use std::net::TcpListener;
use std::thread;

const MIXED: &str = r#"{
    "model":"jev-latest",
    "state":{"z":"fixed","a":"duplicate charge"},
    "questions":{
        "team":{"type":"choice","criteria":{"tech":"technical","billing":null}},
        "urgent":{"type":"score","criteria":["low",{"priority":"high"}]},
        "duplicate":{"type":"noul"}
    }
}"#;
const MIXED_ANSWER: &str = r#"{
    "model":"actual",
    "usage":{"input_tokens":93,"output_tokens":0},
    "answers":{
        "team":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"tech":0.1,"billing":0.9}},
        "urgent":{"type":"score","score":0.8,"confidence":0.6,"legend":{"0":"low","1":{"priority":"high"}},"probabilities":{"0":0.2,"1":0.8}},
        "duplicate":{"type":"noul","noul":0.85}
    }
}"#;

fn server(base_url: &str) -> Server {
    let config = json!({
        "models": [{
            "name": "jev-latest",
            "base_url": base_url,
            "upstream_model": "actual",
            "description": "decision model fixture",
            "release_date": "2026-09-15",
            "supports_images": true
        }]
    });
    Server::from_json(config.to_string().as_bytes(), Path::new(".")).unwrap()
}

fn call(server: &Server, headers: &str, body: &[u8]) -> (String, Value) {
    let request = format!("{headers}\r\n\r\n");
    let head = http::read_head(&mut request.as_bytes()).unwrap();
    let mut response = Vec::new();
    server.route(&head, body, &mut response);
    let response = String::from_utf8(response).unwrap();
    let (headers, body) = response.split_once("\r\n\r\n").unwrap();
    (headers.to_owned(), serde_json::from_str(body).unwrap())
}

fn provider(status: u16, answer: &'static str) -> (Server, thread::JoinHandle<(Head, Vec<u8>)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let server = server(&format!(
        "http://{}/prefix/v1",
        listener.local_addr().unwrap()
    ));
    let upstream = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut reader = BufReader::new(&stream);
        let head = http::read_head(&mut reader).unwrap();
        let body = http::read_body(&mut reader, head.content_length(MAX_BODY).unwrap()).unwrap();
        http::write_response_with_headers(
            &mut stream,
            status,
            "application/json",
            &[("Retry-After", "1")],
            answer.as_bytes(),
        )
        .unwrap();
        (head, body)
    });
    (server, upstream)
}

#[test]
fn catalogs_models_and_routes_a_compressed_mixed_request() {
    let (server, upstream) = provider(200, MIXED_ANSWER);
    let (headers, list) = call(&server, "GET /v1/models HTTP/1.1", b"");
    assert!(headers.starts_with("HTTP/1.1 200"));
    assert_eq!(list["models"][0]["name"], "jev-latest");

    let compressed = zstd::stream::encode_all(MIXED.as_bytes(), 1).unwrap();
    let (headers, response) = call(&server,
        "POST /v1/systemone HTTP/1.1\r\nContent-Encoding: zstd\r\nAuthorization: Bearer client-only\r\nTypeSafe-Danger-Zone-Long-Context: true",
        &compressed,
    );
    assert!(headers.starts_with("HTTP/1.1 200"), "{response}");
    assert_eq!(
        response,
        serde_json::from_str::<Value>(MIXED_ANSWER).unwrap()
    );
    let (head, bytes) = upstream.join().unwrap();
    assert_eq!(head.path, "/prefix/v1/systemone");
    assert_eq!(head.header("authorization"), None);
    assert_eq!(
        head.header("typesafe-danger-zone-long-context"),
        Some("true")
    );
    assert_eq!(head.header("content-encoding"), None);
    let body = String::from_utf8(bytes).unwrap();
    assert!(body.contains(r#""model":"actual""#));
    assert!(body.contains(r#"{"z":"fixed","a":"duplicate charge"}"#));
    assert!(body.contains(r#"{"tech":"technical","billing":null}"#));
}

#[test]
fn forwards_box_requests_and_does_not_reencode_image_bytes() {
    let (server, upstream) = provider(
        200,
        r#"{"model":"actual","usage":{"input_tokens":5,"output_tokens":0},
        "answers":{"item":{"type":"box","box":{"top":0,"bottom":1,"left":0,"right":1},"confidence":null}}}"#,
    );
    let prefix = b"--sample\r\nContent-Disposition: form-data; name=\"request\"\r\nContent-Type: application/json\r\n\r\n";
    let json = br#"{"model":"jev-latest","state":"<p>","questions":{"item":{"type":"box"}},"assets":{"<p>":{"type":"image","part":"picture"}}}"#;
    let image = b"\r\n--sample\r\nContent-Disposition: form-data; name=\"picture\"\r\n\r\n\x00\x01\xff\r\n--sample--\r\n";
    let body = [prefix.as_slice(), json, image].concat();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
    encoder.write_all(&body).unwrap();
    let (headers, response) = call(&server,
        "POST /v1/systemone HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=\"sample\"\r\nContent-Encoding: gzip\r\nTypeSafe-Danger-Zone-Images: true",
        &encoder.finish().unwrap(),
    );
    assert!(headers.starts_with("HTTP/1.1 200"), "{response}");
    assert_eq!(response["answers"]["item"]["type"], "box");
    let (head, forwarded) = upstream.join().unwrap();
    assert_eq!(head.header("typesafe-danger-zone-images"), Some("true"));
    assert!(forwarded.starts_with(prefix));
    assert!(forwarded.ends_with(image));
}

#[test]
fn keeps_validation_locations_and_overload_retry_headers() {
    let (server, upstream) = provider(529, r#"{"detail":"model busy"}"#);
    let post = "POST /v1/systemone HTTP/1.1";
    let (headers, response) = call(&server, post, br#"{"model":"unknown"}"#);
    assert!(headers.starts_with("HTTP/1.1 422"));
    assert_eq!(response["detail"][0]["loc"], json!(["body", "model"]));
    let (headers, response) = call(&server, post, MIXED.as_bytes());
    assert!(headers.starts_with("HTTP/1.1 529"));
    assert!(headers.contains("Retry-After: 1"));
    assert_eq!(response["detail"], "model busy");
    upstream.join().unwrap();
}

#[test]
fn refuses_malformed_compression_and_overdeep_state() {
    let server = server("http://127.0.0.1:9");
    let (headers, _) = call(
        &server,
        "POST /v1/systemone HTTP/1.1\r\nContent-Encoding: gzip",
        b"broken gzip",
    );
    assert!(headers.starts_with("HTTP/1.1 400"));
    let (headers, _) = call(
        &server,
        "POST /v1/systemone HTTP/1.1\r\nContent-Encoding: gzip\r\nContent-Encoding: zstd",
        b"",
    );
    assert!(headers.starts_with("HTTP/1.1 415"));
    let state = format!("{}0{}", "[".repeat(65), "]".repeat(65));
    let request = format!(
        r#"{{"model":"jev-latest","state":{state},"questions":{{"q":{{"type":"noul"}}}}}}"#
    );
    let (headers, _) = call(&server, "POST /v1/systemone HTTP/1.1", request.as_bytes());
    assert!(headers.starts_with("HTTP/1.1 400"));
}

#[test]
fn rejects_incomplete_success_responses_and_cross_origin_form_posts() {
    let (server, upstream) = provider(
        200,
        r#"{"model":"actual","usage":{"input_tokens":0,"output_tokens":0},"answers":{}}"#,
    );
    let (headers, _) = call(&server, "POST /v1/systemone HTTP/1.1", MIXED.as_bytes());
    assert!(headers.starts_with("HTTP/1.1 502"));
    upstream.join().unwrap();
    let (headers, _) = call(&server,
        "POST /v1/systemone HTTP/1.1\r\nHost: localhost:8080\r\nOrigin: https://other.example\r\nContent-Type: multipart/form-data",
        b"",
    );
    assert!(headers.starts_with("HTTP/1.1 403"));
}
