#:property PublishAot=false
// Real-model acceptance probe. Uses an isolated loopback server and stops only
// the process it starts. Model files and existing servers are never modified.
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

var options = new Dictionary<string, string>();
for (var i = 0; i < args.Length; i += 2)
{
    if (i + 1 == args.Length || !args[i].StartsWith("--"))
        throw new ArgumentException("Expected --name value pairs");
    options.Add(args[i][2..], args[i + 1]);
}
if (!options.ContainsKey("binary") || !options.ContainsKey("model") || !options.ContainsKey("output"))
{
    Console.Error.WriteLine("Usage: dotnet run --file scripts/smoke-model.cs -- --binary FILE --model GGUF --output NEW_DIRECTORY [--mmproj GGUF] [--template JINJA] [--device cuda|cpu] [--off rad1] [--expect 4] [--gpu-layers N]");
    return 2;
}
var output = Path.GetFullPath(options["output"]);
if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
    throw new IOException("Output directory must be new or empty");
Directory.CreateDirectory(output);
foreach (var key in new[] { "binary", "model", "mmproj", "template" })
    if (options.TryGetValue(key, out var path) && !File.Exists(path))
        throw new FileNotFoundException(key, path);
var listener = new TcpListener(IPAddress.Loopback, 0);
listener.Start();
var port = ((IPEndPoint)listener.LocalEndpoint).Port;
listener.Stop();
var command = new List<string> { "serve", Path.GetFullPath(options["model"]), "--host", "127.0.0.1", "--port", port.ToString(), "--device", options.GetValueOrDefault("device", "cuda"), "--parallel", "1", "--no-prompt-cache", "-c", options.GetValueOrDefault("context", "4096"), "-t", "6", "--alias", "model-under-test" };
foreach (var (key, flag) in new[] { ("mmproj", "--mmproj"), ("template", "--chat-template-file"), ("off", "--off"), ("gpu-layers", "--gpu-layers") })
    if (options.TryGetValue(key, out var value)) command.AddRange(new[] { flag, value });
var info = new ProcessStartInfo(Path.GetFullPath(options["binary"])) { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true, RedirectStandardError = true };
foreach (var arg in command) info.ArgumentList.Add(arg);
var binaryHash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(await File.ReadAllBytesAsync(info.FileName))).ToLowerInvariant();
await File.WriteAllTextAsync(Path.Combine(output, "run.json"), JsonSerializer.Serialize(new { binary = info.FileName, binary_sha256 = binaryHash, model_bytes = new FileInfo(options["model"]).Length, arguments = command, startedUtc = DateTimeOffset.UtcNow }, new JsonSerializerOptions { WriteIndented = true }));
using var process = Process.Start(info) ?? throw new IOException("Cannot start server");
var stdout = process.StandardOutput.ReadToEndAsync();
var stderr = process.StandardError.ReadToEndAsync();
var results = new List<object>();
var errors = new List<string>();
try
{
    using var client = new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}"), Timeout = TimeSpan.FromMinutes(15) };
    var startup = Stopwatch.StartNew();
    var ready = false;
    while (startup.Elapsed < TimeSpan.FromMinutes(10) && !process.HasExited)
    {
        try
        {
            using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(2));
            using var health = await client.GetAsync("/health", timeout.Token);
            ready = health.IsSuccessStatusCode;
        }
        catch (Exception e) when (e is HttpRequestException or TaskCanceledException) { }
        if (ready) break;
        await Task.Delay(500);
    }
    if (!ready) throw new IOException("Server failed to become healthy");
    await File.WriteAllTextAsync(Path.Combine(output, "models.json"), await client.GetStringAsync("/v1/models"));
    await Probe("text", "/v1/chat/completions", new {
        model = "model-under-test", messages = new[] { new { role = "user", content = "Сколько будет 2 + 2? Ответь только числом." } },
        temperature = 0, max_tokens = 64, chat_template_kwargs = new { enable_thinking = false, preserve_thinking = true }
    }, response => {
        var content = response["choices"]?[0]?["message"]?["content"]?.GetValue<string>();
        if (string.IsNullOrWhiteSpace(content)) throw new IOException("Empty text answer");
        if (options.TryGetValue("expect", out var expected) && content.Trim() != expected)
            throw new IOException($"Expected {expected}, got {content}");
    });
    await Probe("raw", "/v1/completions", new {
        model = "model-under-test", prompt = "The capital of France is", temperature = 0, top_k = 1, repeat_penalty = 1.0, max_tokens = 24, logprobs = 2
    }, response => {
        if (string.IsNullOrWhiteSpace(response["choices"]?[0]?["text"]?.GetValue<string>())) throw new IOException("Empty raw completion");
    });
    if (options.GetValueOrDefault("history") == "true")
    {
        var promptLengths = new Dictionary<bool, int>();
        foreach (var preserve in new[] { false, true })
        {
            await Probe("history-" + preserve, "/v1/chat/completions", new {
                model = "model-under-test", messages = new object[] {
                    new { role = "user", content = "What is 2 + 2?" },
                    new { role = "assistant", content = "4", reasoning_content = "Adding two units and two more units gives four units." },
                    new { role = "user", content = "What is 3 + 3? Answer only the number." }
                }, temperature = 0, max_tokens = 32,
                chat_template_kwargs = new { enable_thinking = false, preserve_thinking = preserve }
            }, response => {
                promptLengths[preserve] = response["usage"]!["prompt_tokens"]!.GetValue<int>();
                if (response["choices"]?[0]?["message"]?["content"]?.GetValue<string>()?.Trim() != "6")
                    throw new IOException("History answer was not 6");
            });
        }
        if (promptLengths.Count != 2 || promptLengths[true] <= promptLengths[false])
            errors.Add("preserve_thinking did not retain additional reasoning tokens");
    }
    var tools = new[] { new { type = "function", function = new { name = "click", description = "Click a position in the image. Coordinates are normalized to integers from 0 to 1000, measured from the top left.", parameters = new { type = "object", properties = new { x = new { type = "integer", minimum = 0, maximum = 1000 }, y = new { type = "integer", minimum = 0, maximum = 1000 } }, required = new[] { "x", "y" }, additionalProperties = false } } } };
    foreach (var mirror in options.ContainsKey("mmproj") ? new[] { false, true } : new[] { false })
    {
    object content = "Call click with x=750 and y=500.";
    if (options.ContainsKey("mmproj"))
    {
        // 256x256 BMP: left half red, right half blue. No image libraries needed.
        var bmp = new byte[54 + 256 * 256 * 3];
        using (var writer = new BinaryWriter(new MemoryStream(bmp)))
        {
            writer.Write((ushort)0x4d42); writer.Write(bmp.Length); writer.Write(0); writer.Write(54);
            writer.Write(40); writer.Write(256); writer.Write(256); writer.Write((ushort)1); writer.Write((ushort)24);
            writer.Write(new byte[24]);
        }
        for (var y = 0; y < 256; y++) for (var x = 0; x < 256; x++) bmp[54 + (y * 256 + x) * 3 + ((x < 128) != mirror ? 2 : 0)] = 255;
        await File.WriteAllBytesAsync(Path.Combine(output, mirror ? "blue-red.bmp" : "red-blue.bmp"), bmp);
        content = new object[] {
            new { type = "text", text = "Click the center of the BLUE half of this image. Use the click tool with coordinates normalized to 0..1000." },
            new { type = "image_url", image_url = new { url = "data:image/bmp;base64," + Convert.ToBase64String(bmp) } }
        };
    }
    JsonNode? completedCall = null;
    await Probe(options.ContainsKey("mmproj") ? (mirror ? "vision-tool-mirror" : "vision-tool") : "tool", "/v1/chat/completions", new {
        model = "model-under-test", messages = new[] { new { role = "user", content } }, tools,
        tool_choice = new { type = "function", function = new { name = "click" } }, temperature = 0, max_tokens = 160,
        chat_template_kwargs = new { enable_thinking = false }
    }, response => {
        var call = response["choices"]?[0]?["message"]?["tool_calls"]?[0]?["function"] ?? throw new IOException("No parsed tool call");
        if (call["name"]?.GetValue<string>() != "click") throw new IOException("Wrong tool");
        var arguments = JsonNode.Parse(call["arguments"]!.GetValue<string>())!;
        var x = arguments["x"]!.GetValue<int>(); var y = arguments["y"]!.GetValue<int>();
        if (x < 0 || x > 1000 || y < 0 || y > 1000 || (mirror ? x >= 500 : x <= 500)) throw new IOException($"Click is outside the blue half: {x}, {y}");
        completedCall = response["choices"]![0]!["message"]!["tool_calls"]![0]!.DeepClone();
    });
    if (!mirror && completedCall is not null)
    {
        await Probe("tool-history", "/v1/chat/completions", new {
            model = "model-under-test", messages = new object[] {
                new { role = "user", content },
                new { role = "assistant", content = (string?)null, tool_calls = new[] { completedCall } },
                new { role = "tool", tool_call_id = completedCall["id"]!.GetValue<string>(), content = "{\"ok\":true}" },
                new { role = "user", content = "What is 2 + 2? Answer only the number." }
            }, tools, tool_choice = "none", temperature = 0, max_tokens = 32,
            chat_template_kwargs = new { enable_thinking = false }
        }, response => {
            if (response["choices"]?[0]?["message"]?["content"]?.GetValue<string>()?.Trim() != "4")
                throw new IOException("Tool history did not produce the expected follow-up answer");
        });
    }
    }

    async Task Probe(string name, string route, object body, Action<JsonNode> validate)
    {
        try
        {
            var json = JsonSerializer.Serialize(body);
            await File.WriteAllTextAsync(Path.Combine(output, name + ".request.json"), json);
            var watch = Stopwatch.StartNew();
            using var payload = new StringContent(json, Encoding.UTF8, "application/json");
            using var response = await client.PostAsync(route, payload);
            var text = await response.Content.ReadAsStringAsync();
            await File.WriteAllTextAsync(Path.Combine(output, name + ".response.json"), text);
            response.EnsureSuccessStatusCode();
            validate(JsonNode.Parse(text) ?? throw new IOException("No JSON response"));
            results.Add(new { name, status = "passed", seconds = watch.Elapsed.TotalSeconds });
            Console.WriteLine($"PASS {name}: {watch.Elapsed.TotalSeconds:F2}s");
        }
        catch (Exception e) { errors.Add(name + ": " + e.Message); Console.Error.WriteLine($"FAIL {name}: {e.Message}"); }
    }
}
catch (Exception e) { errors.Add(e.ToString()); Console.Error.WriteLine(e.Message); }
finally
{
    if (!process.HasExited) process.Kill(entireProcessTree: true);
    await process.WaitForExitAsync();
    await File.WriteAllTextAsync(Path.Combine(output, "server.stdout.log"), await stdout);
    await File.WriteAllTextAsync(Path.Combine(output, "server.stderr.log"), await stderr);
    await File.WriteAllTextAsync(Path.Combine(output, "results.json"), JsonSerializer.Serialize(new { results, errors }, new JsonSerializerOptions { WriteIndented = true }));
}
return errors.Count == 0 ? 0 : 1;
