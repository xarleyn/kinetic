#:property PublishAot=false
using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json.Nodes;

var dryRun = args.Length == 6 && args[5] == "--dry-run";
if (args.Length != 5 && !dryRun)
{
    Console.Error.WriteLine("Usage: dotnet run --file scripts/download-hf-file.cs -- REPO COMMIT FILENAME OUTPUT WORKERS [--dry-run]");
    return 2;
}
var repo = args[0]; var revision = args[1]; var name = args[2];
var output = Path.GetFullPath(args[3]); var workers = int.Parse(args[4]);
if (revision.Length != 40 || !revision.All(Uri.IsHexDigit) || workers is < 1 or > 64)
    throw new ArgumentException("Pin a full commit SHA and use 1..64 workers");
using var client = new HttpClient(new SocketsHttpHandler { MaxConnectionsPerServer = workers, AutomaticDecompression = DecompressionMethods.None }) { Timeout = TimeSpan.FromMinutes(3) };
var metadata = JsonNode.Parse(await client.GetStringAsync($"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))!;
var entry = metadata["siblings"]!.AsArray().Single(n => n!["rfilename"]!.GetValue<string>() == name)!;
var length = entry["size"]!.GetValue<long>();
var expected = entry["lfs"]?["sha256"]?.GetValue<string>() ?? throw new IOException("File has no LFS SHA-256");
if (dryRun)
{
    Console.WriteLine($"Would download {repo}@{revision}/{name}: {length} bytes to {output}, {workers} workers; verify SHA-256 {expected}");
    return 0;
}
if (File.Exists(output))
{
    await Verify();
    Console.WriteLine("Existing file verified: " + output);
    return 0;
}
Directory.CreateDirectory(Path.GetDirectoryName(output)!);
var parts = output + ".ranges";
Directory.CreateDirectory(parts);
const int chunk = 4 * 1024 * 1024;
var count = (int)((length + chunk - 1) / chunk);
var origin = new Uri($"https://huggingface.co/{repo}/resolve/{revision}/{name}");
var staging = output + ".assembling";
var identity = $"{repo}\n{revision}\n{name}\n{expected}\n{length}\n";
var manifest = Path.Combine(parts, "source.txt");
if (File.Exists(manifest) && await File.ReadAllTextAsync(manifest) != identity)
    throw new IOException("Download cache belongs to another source");
if (File.Exists(staging) && !File.Exists(manifest))
    throw new IOException("Unrecognized assembly file already exists");
await File.WriteAllTextAsync(manifest, identity);
long assembled = File.Exists(staging) ? new FileInfo(staging).Length : 0;
if (assembled > length) throw new IOException("Assembly file exceeds expected size");
if (assembled != length && assembled % chunk != 0)
{
    // An interrupted copy may leave a partial final chunk; it is downloaded
    // again. The source manifest above establishes ownership of this file.
    assembled -= assembled % chunk;
    using var recovery = File.OpenWrite(staging);
    recovery.SetLength(assembled);
}
// Resolve the CDN once, retaining signed URLs only in memory.
async Task<Uri> Resolve()
{
    using var request = new HttpRequestMessage(HttpMethod.Get, origin);
    request.Headers.Range = new RangeHeaderValue(0, 0);
    using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
    response.EnsureSuccessStatusCode();
    return response.RequestMessage!.RequestUri!;
}
var url = await Resolve();
using var refresh = new SemaphoreSlim(1);
var watch = Stopwatch.StartNew();
long done = 0; long transferred = 0; long lastReported = 0;
await Parallel.ForEachAsync(Enumerable.Range(0, count), new ParallelOptions { MaxDegreeOfParallelism = workers }, async (index, cancellation) => {
    var start = (long)index * chunk; var size = Math.Min(chunk, length - start);
    var path = Path.Combine(parts, index.ToString("D6"));
    if (start >= assembled && (!File.Exists(path) || new FileInfo(path).Length != size))
    {
        for (var attempt = 1; ; attempt++)
        {
            try
            {
                using var request = new HttpRequestMessage(HttpMethod.Get, url);
                request.Headers.Range = new RangeHeaderValue(start, start + size - 1);
                using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellation);
                if (response.StatusCode is HttpStatusCode.Forbidden or HttpStatusCode.Unauthorized)
                {
                    await refresh.WaitAsync(cancellation);
                    try
                    {
                        if (url == request.RequestUri) url = await Resolve();
                    }
                    finally { refresh.Release(); }
                    throw new IOException("CDN authorization expired; refreshed download URL");
                }
                var range = response.Content.Headers.ContentRange;
                if (response.StatusCode != HttpStatusCode.PartialContent || range?.From != start || range.To != start + size - 1 || range.Length != length)
                    throw new IOException("Server returned an incorrect byte range");
                using var file = new FileStream(path + ".tmp", FileMode.Create, FileAccess.Write, FileShare.None, 128 * 1024, true);
                await response.Content.CopyToAsync(file, cancellation);
                if (file.Length != size) throw new IOException("Truncated range");
                file.Close();
                File.Move(path + ".tmp", path, overwrite: true);
                Interlocked.Add(ref transferred, size);
                break;
            }
            catch (Exception e) when (attempt < 6 && e is HttpRequestException or IOException or TaskCanceledException)
            {
                await Task.Delay(TimeSpan.FromSeconds(attempt * 2), cancellation);
            }
        }
    }
    var completed = Interlocked.Add(ref done, size);
    var elapsed = watch.ElapsedMilliseconds;
    var previous = Interlocked.Read(ref lastReported);
    if (elapsed - previous > 15000 && Interlocked.CompareExchange(ref lastReported, elapsed, previous) == previous)
        Console.WriteLine($"{completed / 1e9:F2}/{length / 1e9:F2} GB; {Interlocked.Read(ref transferred) / 1e6 / watch.Elapsed.TotalSeconds:F1} MB/s received this run");
});
using (var destination = new FileStream(staging, FileMode.Append, FileAccess.Write, FileShare.None, 1024 * 1024, true))
{
    for (var index = (int)((assembled + chunk - 1) / chunk); index < count; index++)
    {
        var part = Path.Combine(parts, index.ToString("D6"));
        using (var source = File.OpenRead(part)) await source.CopyToAsync(destination);
        await destination.FlushAsync();
        File.Delete(part);
    }
}
using (var file = File.OpenRead(staging))
    if (file.Length != length || Convert.ToHexString(await SHA256.HashDataAsync(file)).ToLowerInvariant() != expected)
        throw new IOException("Assembled file failed SHA-256 verification");
File.Move(staging, output);
// Remove only the verified chunks created by this downloader.
for (var index = 0; index < count; index++) File.Delete(Path.Combine(parts, index.ToString("D6")));
File.Delete(manifest);
if (!Directory.EnumerateFileSystemEntries(parts).Any()) Directory.Delete(parts);
Console.WriteLine($"Verified {output}: SHA-256 {expected}; {watch.Elapsed}");
return 0;

async Task Verify()
{
    using var file = File.OpenRead(output);
    if (file.Length != length || Convert.ToHexString(await SHA256.HashDataAsync(file)).ToLowerInvariant() != expected)
        throw new IOException("Existing file does not match pinned metadata");
}
