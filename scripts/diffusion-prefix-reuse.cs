#:property PublishAot=false
// Exact CPU diffusion identity and in-process generation timings on an existing
// FP32 fixture. Writes only a new/empty explicitly named output directory.
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;

var options = new Dictionary<string, string>();
var dryRun = false;
var verifyOnly = false;
for (var i = 0; i < args.Length; i++)
{
    if (args[i] == "--dry-run") { dryRun = true; continue; }
    if (args[i] == "--verify-only") { verifyOnly = true; continue; }
    if (args[i] == "--help")
    {
        Console.WriteLine("dotnet run --file scripts/diffusion-prefix-reuse.cs -- --kinetic FILE --fixture DIRECTORY --output NEW_DIRECTORY [--threads 6] [--runs 9] [--verify-only] [--dry-run]");
        return 0;
    }
    if (!args[i].StartsWith("--") || i + 1 == args.Length)
        throw new ArgumentException("Expected --name value pairs");
    var key = args[i][2..];
    if (key is not ("kinetic" or "fixture" or "output" or "threads" or "runs"))
        throw new ArgumentException($"Unknown option {args[i]}");
    options.Add(key, args[++i]);
}
string Required(string key) => options.TryGetValue(key, out var value) ? Path.GetFullPath(value)
    : throw new ArgumentException($"Missing --{key}; use --help");
var binary = Required("kinetic");
var fixture = Required("fixture");
var output = Required("output");
var model = Path.Combine(fixture, "lfm2moe-diffusion-fixture-f32.gguf");
var mask = Path.Combine(fixture, "mask.safetensors");
var configPath = Path.Combine(fixture, "hf", "config.json");
foreach (var path in new[] { binary, model, mask, configPath })
    if (!File.Exists(path)) throw new FileNotFoundException("Required input", path);
var threads = int.Parse(options.GetValueOrDefault("threads", "6"));
var runs = int.Parse(options.GetValueOrDefault("runs", "9"));
if (threads < 1 || threads > 256 || runs < 3 || runs > 100)
    throw new ArgumentException("threads must be 1..256 and runs 3..100");
if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
    throw new IOException("Output directory must be new or empty; previous results are never overwritten");
var config = JsonNode.Parse(File.ReadAllText(configPath))!;
if (config["vocab_size"]!.GetValue<int>() != 384 || config["hidden_size"]!.GetValue<int>() != 32
    || config["num_hidden_layers"]!.GetValue<int>() != 40 || config["num_experts"]!.GetValue<int>() != 64)
    throw new ArgumentException("This probe expects the existing 40-layer width-32 FP32 fixture");
var jsonOptions = new JsonSerializerOptions { WriteIndented = true };
void Save(string path, object? value) => File.WriteAllText(path, JsonSerializer.Serialize(value, jsonOptions));
static string SourcePath([CallerFilePath] string path = "") => path;
static string Hash(string path)
{
    using var file = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(file));
}
if (dryRun)
{
    Console.WriteLine(JsonSerializer.Serialize(new { binary, model, mask, output, threads, runs, verifyOnly, dryRun }));
    return 0;
}
Directory.CreateDirectory(output);
Save(Path.Combine(output, "run.json"), new {
    utc = DateTimeOffset.UtcNow, binary, binary_sha256 = Hash(binary), model, model_sha256 = Hash(model),
    mask, mask_sha256 = Hash(mask), harness = SourcePath(), harness_sha256 = Hash(SourcePath()),
    config, threads, runs, warmups = 2, platform = RuntimeInformation.OSDescription,
    architecture = RuntimeInformation.ProcessArchitecture.ToString(), logical_processors = Environment.ProcessorCount,
    comparison = "bitwise F32 logits and exact confidence, candidates, commits, canvas, tokens",
    timer = "Kinetic Instant; complete generation including per-block prefix preparation; excludes load/tokenization/trace IO",
    full_24b_weights = false
});

async Task<JsonNode> Run(string directory, string prompt, int canvas, int steps, int maximum, int window,
    bool reuse, bool trace, bool benchmark)
{
    Directory.CreateDirectory(directory);
    var command = new List<string> { "diffuse", model, "--mask-embedding", mask, "--prompt-tokens", prompt,
        "--canvas", canvas.ToString(), "--steps", steps.ToString(), "--max-new-tokens", maximum.ToString(),
        "--prefix-window", window.ToString(), "-t", threads.ToString() };
    if (reuse) command.Add("--prefix-reuse");
    if (trace) command.AddRange(["--trace-dir", Path.Combine(directory, "trace")]);
    if (benchmark) command.AddRange(["--benchmark-runs", runs.ToString()]);
    Save(Path.Combine(directory, "command.json"), new[] { binary }.Concat(command).ToArray());
    var start = new ProcessStartInfo(binary) { UseShellExecute = false, CreateNoWindow = true,
        RedirectStandardOutput = true, RedirectStandardError = true };
    foreach (var arg in command) start.ArgumentList.Add(arg);
    using var process = Process.Start(start) ?? throw new IOException("Could not start Kinetic");
    var stdoutTask = process.StandardOutput.ReadToEndAsync();
    var stderrTask = process.StandardError.ReadToEndAsync();
    using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(10));
    try { await process.WaitForExitAsync(deadline.Token); }
    catch (OperationCanceledException) { process.Kill(entireProcessTree: true); throw; }
    var stdout = await stdoutTask; var stderr = await stderrTask;
    File.WriteAllText(Path.Combine(directory, "stdout.json"), stdout);
    File.WriteAllText(Path.Combine(directory, "stderr.log"), stderr);
    if (process.ExitCode != 0) throw new InvalidOperationException($"Kinetic exit {process.ExitCode}: {stderr}");
    return JsonNode.Parse(stdout) ?? throw new IOException("Missing JSON output");
}

void Equal(JsonNode? a, JsonNode? b, string message)
{
    if (!JsonNode.DeepEquals(a, b)) throw new InvalidOperationException(message);
}
long CompareTraces(string directory)
{
    var fullDir = Path.Combine(directory, "full", "trace");
    var reuseDir = Path.Combine(directory, "reuse", "trace");
    var a = JsonNode.Parse(File.ReadAllText(Path.Combine(fullDir, "trace.json")))!;
    var b = JsonNode.Parse(File.ReadAllText(Path.Combine(reuseDir, "trace.json")))!;
    var fullSteps = a["steps"]!.AsArray(); var reuseSteps = b["steps"]!.AsArray();
    if (fullSteps.Count != reuseSteps.Count) throw new InvalidOperationException("Pass counts differ");
    long count = 0;
    for (var step = 0; step < fullSteps.Count; step++)
    {
        var f = fullSteps[step]!; var r = reuseSteps[step]!;
        foreach (var key in new[] { "block", "step", "prefix", "canvas_before", "masked_before", "logits_shape",
            "candidates", "log_confidence", "committed", "canvas_after" })
            Equal(f[key], r[key], $"{directory}, pass {step}: {key} differs");
        var x = File.ReadAllBytes(Path.Combine(fullDir, f["logits_file"]!.GetValue<string>()));
        var y = File.ReadAllBytes(Path.Combine(reuseDir, r["logits_file"]!.GetValue<string>()));
        if (!x.AsSpan().SequenceEqual(y))
            throw new InvalidOperationException($"{directory}, pass {step}: F32 logits are not bit-identical");
        count += x.Length / 4;
    }
    Equal(a["output"]!["tokens"], b["output"]!["tokens"], "Final token IDs differ");
    Equal(a["output"]!["forward_calls"], b["output"]!["forward_calls"], "Forward calls differ");
    var blocks = fullSteps.Last()!["block"]!.GetValue<int>() + 1;
    if (b["output"]!["prefix_builds"]!.GetValue<int>() != blocks)
        throw new InvalidOperationException("Reuse did not prepare exactly one prefix per block");
    return count;
}

string Prompt(string directory, int length)
{
    var ids = new[] { 256 }.Concat(Enumerable.Range(0, length - 1).Select(i => 10 + i * 37 % 240)).ToArray();
    Directory.CreateDirectory(directory);
    var path = Path.Combine(directory, "prompt.json"); Save(path, ids); return path;
}
static double Quantile(double[] values, double fraction)
{
    var sorted = values.Order().ToArray();
    var position = (sorted.Length - 1) * fraction;
    var low = (int)position; var high = (int)Math.Ceiling(position);
    return sorted[low] + (sorted[high] - sorted[low]) * (position - low);
}
var conformance = new List<object>(); var benchmarks = new List<object>();
void Results() => Save(Path.Combine(output, "results.json"), new { conformance, benchmarks });
var cases = new[] { (1, 4, 3, 2, 9), (16, 1, 16, 8, 9), (16, 4, 33, 8, 37),
    (32, 3, 39, 8, 9), (64, 4, 71, 8, 37), (16, 8, 39, 8, 257),
    (32, 8, 71, 8, 257), (64, 8, 135, 8, 257) };
foreach (var (canvas, steps, maximum, window, length) in cases)
{
    var name = $"verify-c{canvas}-s{steps}-p{length}";
    var folder = Path.Combine(output, name); var prompt = Prompt(folder, length);
    var a = await Run(Path.Combine(folder, "full"), prompt, canvas, steps, maximum, window, false, true, false);
    var b = await Run(Path.Combine(folder, "reuse"), prompt, canvas, steps, maximum, window, true, true, false);
    var logits = CompareTraces(folder);
    conformance.Add(new { name, status = "passed", bit_identical = true, logits_compared = logits,
        passes = a["forward_calls"]!.GetValue<int>(), tokens = a["tokens"]!.AsArray().Count,
        prefix_builds = b["prefix_builds"]!.GetValue<int>() });
    Results(); Console.WriteLine($"PASS {name}: {logits} bit-identical logits, exact commits and tokens");
}
if (!verifyOnly)
{
    var caseIndex = 0;
    foreach (var length in new[] { 37, 257 })
    foreach (var steps in new[] { 4, 8 })
    foreach (var canvas in new[] { 16, 32, 64 })
    {
        var name = $"bench-c{canvas}-s{steps}-p{length}";
        var folder = Path.Combine(output, name); var prompt = Prompt(folder, length);
        var outputs = new JsonNode[2];
        // Alternate which process runs first, so ordering does not always favor reuse.
        foreach (var mode in caseIndex++ % 2 == 0 ? new[] { 0, 1 } : new[] { 1, 0 })
            outputs[mode] = await Run(Path.Combine(folder, mode == 0 ? "full" : "reuse"), prompt,
                canvas, steps, canvas * 2 + 7, 64, mode == 1, true, true);
        // Trace only the initial control generation; timed repeats never write
        // trace files. Verify the exact history-window workload being timed.
        var comparedLogits = CompareTraces(folder);
        Equal(outputs[0]["tokens"], outputs[1]["tokens"], "Benchmark tokens differ");
        Equal(outputs[0]["forward_calls"], outputs[1]["forward_calls"], "Benchmark pass counts differ");
        var full = outputs[0]["benchmark"]!["samples_ms"]!.AsArray().Select(v => v!.GetValue<double>()).ToArray();
        var reuse = outputs[1]["benchmark"]!["samples_ms"]!.AsArray().Select(v => v!.GetValue<double>()).ToArray();
        var fullMedian = Quantile(full, 0.5); var reuseMedian = Quantile(reuse, 0.5);
        var speedup = fullMedian / reuseMedian;
        benchmarks.Add(new { name, canvas, steps, prompt_length = length,
            bit_identical = true, logits_compared = comparedLogits,
            output_tokens = outputs[0]["tokens"]!.AsArray().Count,
            passes = outputs[0]["forward_calls"]!.GetValue<int>(), full_samples_ms = full, reuse_samples_ms = reuse,
            full_median_ms = fullMedian, reuse_median_ms = reuseMedian, speedup,
            full_p10_ms = Quantile(full, 0.1), full_p90_ms = Quantile(full, 0.9),
            reuse_p10_ms = Quantile(reuse, 0.1), reuse_p90_ms = Quantile(reuse, 0.9) });
        Results(); Console.WriteLine($"BENCH {name}: full {fullMedian:F3} ms, reuse {reuseMedian:F3} ms, {speedup:F3}x");
    }
}
Console.WriteLine($"PASS: {conformance.Count} conformance cases, {benchmarks.Count} timing cases. Results: {output}");
return 0;
