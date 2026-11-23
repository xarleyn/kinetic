//! The small part of HTTP/1.1 the API needs: one request per connection,
//! bodies sized by `Content-Length`, and responses that are either sized or
//! a server-sent event stream ended by closing the connection.

use std::io::{self, BufRead, Read, Write};

/// Longest request line plus headers.
pub const MAX_HEAD: usize = 16 * 1024;
/// Largest request body.
pub const MAX_BODY: usize = 16 * 1024 * 1024;
const MAX_HEADERS: usize = 100;

/// A request line and headers.
#[derive(Debug, PartialEq, Eq)]
pub struct Head {
    pub method: String,
    /// The target without its query string.
    pub path: String,
    headers: Vec<(String, String)>,
}

impl Head {
    /// The first header called `name`, ignoring case.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn values<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a str> {
        self.headers
            .iter()
            .filter_map(move |(n, v)| n.eq_ignore_ascii_case(name).then_some(v.as_str()))
    }

    /// The body length, checked against [`MAX_BODY`].
    pub fn content_length(&self, max_body: usize) -> Result<usize, Status> {
        if self.header("transfer-encoding").is_some() {
            return Err(Status(411, "send the body with a Content-Length"));
        }
        match self.header("content-length") {
            None => Ok(0),
            Some(v) => match v.trim().parse::<usize>() {
                Ok(n) if n > max_body => Err(Status(413, "the request body is too large")),
                Ok(n) => Ok(n),
                Err(_) => Err(Status(400, "Content-Length is not a number")),
            },
        }
    }

    pub fn expects_continue(&self) -> bool {
        self.header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    }
}

/// A status code and the message to answer with.
#[derive(Debug, PartialEq, Eq)]
pub struct Status(pub u16, pub &'static str);

/// Why no request could be read.
#[derive(Debug)]
pub enum ReadError {
    /// The connection failed or timed out; nothing can be answered.
    Io,
    /// The client sent something to be refused with this status.
    Bad(Status),
}

impl From<io::Error> for ReadError {
    fn from(_: io::Error) -> Self {
        ReadError::Io
    }
}

impl From<Status> for ReadError {
    fn from(s: Status) -> Self {
        ReadError::Bad(s)
    }
}

/// Reads a request line and headers.
pub fn read_head(r: &mut impl BufRead) -> Result<Head, ReadError> {
    let mut used = 0;
    let mut line = Vec::new();
    let mut next_line = |line: &mut Vec<u8>| -> Result<(), ReadError> {
        line.clear();
        let n = r
            .by_ref()
            .take((MAX_HEAD - used) as u64 + 1)
            .read_until(b'\n', line)?;
        used += n;
        if used > MAX_HEAD {
            return Err(Status(431, "the request headers are too large").into());
        }
        if !line.ends_with(b"\n") {
            return Err(io::Error::from(io::ErrorKind::UnexpectedEof).into());
        }
        line.pop();
        if line.ends_with(b"\r") {
            line.pop();
        }
        Ok(())
    };

    next_line(&mut line)?;
    let bad = || ReadError::Bad(Status(400, "malformed request"));
    let request_line = std::str::from_utf8(&line).map_err(|_| bad())?;
    let mut parts = request_line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(bad());
    };
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(Status(505, "only HTTP/1.1 is supported").into());
    }
    let path = target.split('?').next().unwrap_or(target).to_owned();
    let method = method.to_owned();

    let mut headers = Vec::new();
    loop {
        next_line(&mut line)?;
        if line.is_empty() {
            break;
        }
        if headers.len() == MAX_HEADERS {
            return Err(Status(431, "too many request headers").into());
        }
        let text = std::str::from_utf8(&line).map_err(|_| bad())?;
        let (name, value) = text.split_once(':').ok_or_else(bad)?;
        headers.push((name.trim().to_owned(), value.trim().to_owned()));
    }
    Ok(Head {
        method,
        path,
        headers,
    })
}

/// Reads a body of exactly `len` bytes.
pub fn read_body(r: &mut impl BufRead, len: usize) -> io::Result<Vec<u8>> {
    let mut body = vec![0; len];
    r.read_exact(&mut body)?;
    Ok(body)
}

/// Reads a chunked body with the same aggregate bound as a sized body.
pub fn read_chunked(r: &mut impl BufRead, max_body: usize) -> Result<Vec<u8>, ReadError> {
    let malformed = || Status(400, "malformed chunked body");
    let mut body = Vec::new();
    let mut line = Vec::new();
    loop {
        line.clear();
        r.by_ref().take(1024).read_until(b'\n', &mut line)?;
        let line = line.strip_suffix(b"\r\n").ok_or_else(malformed)?;
        let size = line.split(|&b| b == b';').next().unwrap_or_default();
        let size = std::str::from_utf8(size).map_err(|_| malformed())?;
        if size.is_empty() || !size.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(malformed().into());
        }
        let size = usize::from_str_radix(size, 16)
            .map_err(|_| Status(413, "the request body is too large"))?;
        if size == 0 {
            // There is no next request on this connection. Consume the optional
            // trailers within the same header allowance as the request head.
            let mut trailers = r.take(MAX_HEAD as u64);
            let mut line = Vec::new();
            loop {
                line.clear();
                trailers.read_until(b'\n', &mut line)?;
                if line == b"\r\n" {
                    return Ok(body);
                }
                if !line.ends_with(b"\r\n") || !line.contains(&b':') {
                    return Err(malformed().into());
                }
            }
        }
        let start = body.len();
        if size > max_body - start {
            return Err(Status(413, "the request body is too large").into());
        }
        body.resize(start + size, 0);
        r.read_exact(&mut body[start..])?;
        let mut end = [0; 2];
        r.read_exact(&mut end)?;
        if end != *b"\r\n" {
            return Err(malformed().into());
        }
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        422 => "Unprocessable Content",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        529 => "Overloaded",
        _ => "",
    }
}

/// Tells a client waiting on `Expect: 100-continue` to send the body.
pub fn write_continue(w: &mut impl Write) -> io::Result<()> {
    w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    w.flush()
}

/// Writes a complete response and asks the client to close.
pub fn write_response(
    w: &mut impl Write,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    write_response_with_headers(w, status, content_type, &[], body)
}

pub fn write_response_with_headers(
    w: &mut impl Write,
    status: u16,
    content_type: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> io::Result<()> {
    write!(
        w,
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n",
        reason(status),
        body.len()
    )?;
    for (name, value) in headers {
        write!(w, "{name}: {value}\r\n")?;
    }
    w.write_all(b"\r\n")?;
    w.write_all(body)?;
    w.flush()
}

/// Starts a server-sent event stream, which ends when the connection does.
pub fn write_event_stream_head(w: &mut impl Write) -> io::Result<()> {
    write_stream_head(w, "text/event-stream")
}

/// Starts a response of `content_type` that ends when the connection does.
pub fn write_stream_head(w: &mut impl Write, content_type: &str) -> io::Result<()> {
    write!(
        w,
        "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    w.flush()
}

/// Sends one server-sent event carrying `data`, line by line.
pub fn write_event(w: &mut impl Write, data: &str) -> io::Result<()> {
    let mut event = String::with_capacity(data.len() + 8);
    for line in data.split('\n') {
        event.push_str("data: ");
        event.push_str(line.strip_suffix('\r').unwrap_or(line));
        event.push('\n');
    }
    event.push('\n');
    w.write_all(event.as_bytes())?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_bodies_are_bounded_and_consume_trailers() {
        let mut body = &b"3;name=value\r\nabc\r\n2\r\nde\r\n0\r\nTrailer: yes\r\n\r\n"[..];
        assert_eq!(read_chunked(&mut body, 5).unwrap(), b"abcde");
        assert!(body.is_empty());
        let mut too_large = &b"3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n"[..];
        assert!(matches!(
            read_chunked(&mut too_large, 5),
            Err(ReadError::Bad(Status(413, _)))
        ));
        let mut malformed = &b"4\r\nabcd  \r\n0\r\n\r\n"[..];
        assert!(matches!(
            read_chunked(&mut malformed, 8),
            Err(ReadError::Bad(Status(400, _)))
        ));
    }

    fn head(raw: &str) -> Result<Head, ReadError> {
        read_head(&mut raw.as_bytes())
    }

    #[test]
    fn reads_a_request() {
        let raw = "POST /v1/chat/completions?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\ncontent-length: 2\r\n\r\n{}";
        let mut r = raw.as_bytes();
        let h = read_head(&mut r).unwrap();
        assert_eq!(
            (h.method.as_str(), h.path.as_str()),
            ("POST", "/v1/chat/completions")
        );
        assert_eq!(h.header("Content-Type"), Some("application/json"));
        assert_eq!(h.content_length(MAX_BODY), Ok(2));
        assert_eq!(read_body(&mut r, 2).unwrap(), b"{}");
    }

    #[test]
    fn accepts_bare_newlines() {
        let h = head("GET /health HTTP/1.1\nHost: x\n\n").unwrap();
        assert_eq!(h.path, "/health");
        assert_eq!(h.content_length(MAX_BODY), Ok(0));
    }

    #[test]
    fn refuses_bad_requests() {
        let status = |raw: &str| match head(raw) {
            Err(ReadError::Bad(Status(code, _))) => code,
            other => panic!("{other:?}"),
        };
        assert_eq!(status("GARBAGE\r\n\r\n"), 400);
        assert_eq!(status("GET / HTTP/2\r\n\r\n"), 505);
        assert_eq!(status("GET / HTTP/1.1\r\nno colon\r\n\r\n"), 400);
        let huge = format!("GET / HTTP/1.1\r\nX: {}\r\n\r\n", "a".repeat(MAX_HEAD));
        assert_eq!(status(&huge), 431);
    }

    #[test]
    fn a_cut_off_head_is_an_io_error() {
        assert!(matches!(head("GET / HTTP/1.1\r\nHost"), Err(ReadError::Io)));
    }

    #[test]
    fn checks_the_body_length() {
        let h = head(&format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        ))
        .unwrap();
        assert_eq!(h.content_length(MAX_BODY).unwrap_err().0, 413);
        let h = head("POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n").unwrap();
        assert_eq!(h.content_length(MAX_BODY).unwrap_err().0, 411);
        let h = head("POST / HTTP/1.1\r\nExpect: 100-continue\r\n\r\n").unwrap();
        assert!(h.expects_continue());
    }

    #[test]
    fn frames_events() {
        let mut out = Vec::new();
        write_event(&mut out, r#"{"a":1}"#).unwrap();
        write_event(&mut out, "[DONE]").unwrap();
        assert_eq!(out, b"data: {\"a\":1}\n\ndata: [DONE]\n\n");
    }

    #[test]
    fn multi_line_data_becomes_several_data_lines() {
        let mut out = Vec::new();
        write_event(&mut out, "one\r\ntwo").unwrap();
        assert_eq!(out, b"data: one\ndata: two\n\n");
    }

    #[test]
    fn sized_responses_close_the_connection() {
        let mut out = Vec::new();
        write_response(&mut out, 404, "application/json", b"{}").unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.starts_with("HTTP/1.1 404 Not Found\r\n"));
        assert!(text.contains("Content-Length: 2\r\n"));
        assert!(text.ends_with("Connection: close\r\n\r\n{}"));
    }
}
