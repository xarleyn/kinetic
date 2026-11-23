#:property PublishAot=false
#:property TargetFramework=net10.0
using System.Globalization;
using System.Text.Json;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
if (args.Length is < 3 or > 4 || (args.Length == 4 && args[3] != "--dry-run"))
{
    Console.Error.WriteLine("Usage: summarize-activation-clustering EXPERIMENT_DIR QUALITY_DIR NEW_OUTPUT.json [--dry-run]");
    return 2;
}
try
{
    var experiment = Path.GetFullPath(args[0]); var qualityDir = Path.GetFullPath(args[1]); var output = Path.GetFullPath(args[2]);
    if (File.Exists(output) || Directory.Exists(output)) throw new IOException("Use a new output path.");
    var rows = JsonSerializer.Deserialize<Row[]>(File.ReadAllText(Path.Combine(experiment, "activation-results.json")))!;
    using var quality = JsonDocument.Parse(File.ReadAllText(Path.Combine(qualityDir, "quality.json")));
    var comparisons = quality.RootElement.GetProperty("comparisons").Deserialize<Comparison[]>()!;
    using var verification = JsonDocument.Parse(File.ReadAllText(Path.Combine(experiment, "verification.json")));
    using var profiles = JsonDocument.Parse(File.ReadAllText(Path.Combine(experiment, "validation-profiles.json")));
    if (rows.Length != 324 || comparisons.Length != 10 || verification.RootElement.GetArrayLength() != 9) throw new InvalidDataException("Expected complete 324-candidate / 9-model experiment.");
    var baseline = comparisons.Single(r => r.Model == "baseline-control").Ppl;
    var byPayload = rows.ToDictionary(r => r.Payload);
    var lloyd = rows.Where(r => r.Algorithm == "lloyd").ToDictionary(r => (r.Tensor, r.Seed, r.Recipe));
    var training = rows.GroupBy(r => r.Algorithm + r.Epochs).Select(group => new {
        method = group.Key, cases = group.Count(), median_train_ms = Median(group.Select(r => r.TrainMs)),
        median_paired_speedup_vs_lloyd25 = Median(group.Select(r => lloyd[(r.Tensor, r.Seed, r.Recipe)].TrainMs / r.TrainMs)),
        median_paired_real_output_nmse_ratio = Median(group.Select(r => r.OutputNmse / lloyd[(r.Tensor, r.Seed, r.Recipe)].OutputNmse))
    }).ToArray();
    var models = verification.RootElement.EnumerateArray().Select(item => {
        var filename = item.GetProperty("file").GetString()!; var name = Path.GetFileNameWithoutExtension(filename);
        var parts = name.Split('.'); var profile = parts[1]; var method = parts[2];
        var comparison = comparisons.Single(r => r.Model == name);
        var raw = comparisons.Single(r => r.Model == "reconstructed.raw." + method);
        var selected = item.GetProperty("payloads").EnumerateArray().Select(p => byPayload[p.GetString()!]).ToArray();
        if (selected.Length != 6 || selected.Select(r => r.Tensor).Distinct().Count() != 6) throw new InvalidDataException("A profile must quantize all six distinct matrices.");
        var val = profiles.RootElement.EnumerateArray().Single(p => p.GetProperty("algorithm").GetString() + p.GetProperty("epochs").GetInt32() == method);
        return new {
            model = name, profile, method, comparison.Ppl, ppl_change_percent = (comparison.Ppl / baseline - 1) * 100,
            comparison.Kld, comparison.SameTopPercent,
            selected_by_validation = val.GetProperty("selected_by_validation").GetString() == profile,
            fraction_excess_nll_removed_vs_raw = 1 - Math.Log(comparison.Ppl / baseline) / Math.Log(raw.Ppl / baseline),
            fraction_excess_ppl_removed_vs_raw = 1 - (comparison.Ppl - baseline) / (raw.Ppl - baseline),
            payload_bytes = selected.Sum(r => r.PayloadBytes), bits_per_weight = selected.Sum(r => r.PayloadBytes) * 8.0 / 3649536,
            total_selected_train_ms = selected.Sum(r => r.TrainMs), median_real_output_nmse = Median(selected.Select(r => r.OutputNmse)),
            median_weight_nmse = Median(selected.Select(r => r.WeightNmse)),
            recipes = selected.Select(r => new { r.Tensor, r.Recipe, r.OutputNmse }).ToArray()
        };
    }).OrderBy(r => r.method).ThenBy(r => r.profile).ToArray();
    var result = new { baseline_ppl = baseline, training, models };
    if (args.Length == 4) { Console.WriteLine("DRY RUN: complete inputs validated; no files written."); return 0; }
    using (var stream = File.Open(output, FileMode.CreateNew)) JsonSerializer.Serialize(stream, result, new JsonSerializerOptions { WriteIndented = true });
    foreach (var row in models) Console.WriteLine($"{row.method,-8} {row.profile,-12} PPL={row.Ppl:F6} change={row.ppl_change_percent:+0.000;-0.000;0}% excess-NLL-removed={row.fraction_excess_nll_removed_vs_raw:P2} selected={row.selected_by_validation}");
    return 0;
}
catch (Exception ex) { Console.Error.WriteLine(ex.Message); return 1; }

static double Median(IEnumerable<double> values)
{
    var a = values.Order().ToArray(); var i = a.Length / 2;
    return a.Length % 2 == 0 ? (a[i - 1] + a[i]) / 2 : a[i];
}
record Row(string Tensor, string Algorithm, int Epochs, int Seed, string Recipe, double TrainMs, double WeightNmse, double OutputNmse, string Payload, long PayloadBytes);
record Comparison(string Model, double Ppl, double Kld, double SameTopPercent);
