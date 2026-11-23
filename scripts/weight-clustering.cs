#:property PublishAot=false
#:property TargetFramework=net10.0
// CPU experiment; no inference-engine changes or third-party dependencies.
// E-means recipe: faiss 2b648f0605f2ef3d14c4705350f8a0b214463878,
// contrib/e_means/{base,e_means}.py (MIT). Independent C# implementation;
// same update/schedule/revival/finalization, different RNG and distance kernel.
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
try { return MainRun(args); }
catch (Exception ex) { Console.Error.WriteLine(ex.Message); return 1; }

static int MainRun(string[] args)
{
    if (args.Length == 0 || args.Contains("--help"))
    {
        Console.WriteLine("weight-clustering --model FILE --out NEW_DIR [--tensors comma,separated] [--dims 2,4,8] [--seeds 42,43,44] [--train-vectors 65536] [--e-epochs 2,3,5,10] [--threads 6] [--patch-seed 42] [--timing-repeats 5] [--dry-run true]\nQuality: --evaluate EXPERIMENT_DIR --quality-out NEW_DIR --kinetic EXE --corpus FILE [--ctx 512] [--chunks 64] [--threads 6] [--dry-run true]\nOther modes: --inspect FILE; --self-test. Outputs exact VQ payloads, JSON/CSV, and reconstructed FP16 GGUF copies for the patch seed. Output directory must not exist. No original file is modified.");
        Console.WriteLine("Activation-aware mode: --model FILE --out NEW_DIR --activation-dir TRAIN_CAPTURE --validation-activations VALID_CAPTURE --calibration-corpus TRAIN_TEXT --validation-corpus VALID_TEXT --kinetic EXE [--validation-chunks 8] [--seeds 42,43,44] [--e-epochs 3,5] [--timing-repeats 5] [--dry-run true]. Always d=2, k=256; fixed six-recipe grid; no test-set selection.");
        return 0;
    }
    if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
    if (args.Length == 2 && args[0] == "--inspect")
    {
        var g = Gguf.Open(args[1]);
        foreach (var t in g.Tensors.Where(t => t.Dims.Length == 2))
            Console.WriteLine($"{t.Name} [{string.Join(',', t.Dims)}] type={t.Type}");
        return 0;
    }
    var opts = new Dictionary<string, string>();
    for (var i = 0; i < args.Length; i += 2)
    {
        if (i + 1 == args.Length || !args[i].StartsWith("--") || !opts.TryAdd(args[i], args[i + 1]))
            throw new ArgumentException("Expected unique --option value pairs.");
    }
    string Get(string key, string? fallback = null) => opts.GetValueOrDefault(key) ?? fallback ?? throw new ArgumentException($"Missing {key}");
    var allowed = new[] { "--model", "--out", "--tensors", "--dims", "--seeds", "--train-vectors", "--e-epochs", "--threads", "--patch-seed", "--timing-repeats", "--evaluate", "--quality-out", "--kinetic", "--corpus", "--ctx", "--chunks", "--dry-run", "--activation-dir", "--validation-activations", "--validation-corpus", "--validation-chunks", "--calibration-corpus" };
    if (opts.Keys.Any(k => !allowed.Contains(k))) throw new ArgumentException("Unknown option.");
    if (opts.ContainsKey("--evaluate")) return Evaluate(opts);
    if (opts.ContainsKey("--activation-dir")) return ActivationExperiment(opts);
    var model = Path.GetFullPath(Get("--model"));
    var output = Path.GetFullPath(Get("--out"));
    var dims = Get("--dims", "2,4,8").Split(',').Select(int.Parse).ToArray();
    var seeds = Get("--seeds", "42,43,44").Split(',').Select(int.Parse).ToArray();
    var budgets = Get("--e-epochs", "2,3,5,10").Split(',').Select(int.Parse).ToArray();
    var trainLimit = int.Parse(Get("--train-vectors", "65536"));
    var threads = int.Parse(Get("--threads", "6"));
    var patchSeed = int.Parse(Get("--patch-seed", "42"));
    var repeats = int.Parse(Get("--timing-repeats", "5"));
    if (dims.Any(d => d is not (2 or 4 or 8)) || budgets.Any(t => t < 1 || t > 100) ||
        seeds.Length == 0 || trainLimit < 1024 || threads < 1 || threads > Environment.ProcessorCount || repeats < 1 || repeats > 31 ||
        dims.Distinct().Count() != dims.Length || seeds.Distinct().Count() != seeds.Length || budgets.Distinct().Count() != budgets.Length)
        throw new ArgumentException("Invalid/duplicate dimensions, seeds, budgets, training size or thread count.");
    if (Directory.Exists(output) || File.Exists(output)) throw new IOException($"Output already exists: {output}. Use a fresh directory.");
    var gguf = Gguf.Open(model);
    var names = Get("--tensors", "blk.0.attn_q.weight,blk.0.ffn_down.weight,blk.15.attn_q.weight,blk.15.ffn_down.weight,blk.29.attn_q.weight,blk.29.ffn_down.weight").Split(',');
    if (names.Distinct().Count() != names.Length) throw new ArgumentException("Duplicate tensor names.");
    var tensors = names.Select(n => gguf.Tensors.SingleOrDefault(t => t.Name == n) ?? throw new ArgumentException($"No tensor: {n}")).ToArray();
    foreach (var t in tensors)
        if (t.Type != 1 || t.Dims.Length != 2 || dims.Any(d => t.Dims[0] % (ulong)d != 0))
            throw new ArgumentException($"{t.Name}: requires FP16 two-dimensional weights with row length divisible by all vector dimensions.");
    if (bool.Parse(Get("--dry-run", "false")))
    {
        Console.WriteLine($"DRY RUN: read {model}; create {output}; {tensors.Length * dims.Length * seeds.Length * (budgets.Length + 1)} VQ payloads; {(seeds.Contains(patchSeed) ? dims.Length * (budgets.Length + 1) : 0)} reconstructed GGUF copies. No files written.");
        return 0;
    }
    Directory.CreateDirectory(output);
    var json = new JsonSerializerOptions { WriteIndented = true };
    var sourceHash = Hash(model);
    File.WriteAllText(Path.Combine(output, "manifest.json"), JsonSerializer.Serialize(new {
        schema = 1, created_utc = DateTime.UtcNow, model, source_sha256 = sourceHash,
        source_bytes = new FileInfo(model).Length, kinetic_commit = GitHead(),
        tool_sha256 = Hash(SourcePath()),
        upstream_commit = "2b648f0605f2ef3d14c4705350f8a0b214463878", backend = "C# CPU, shared exact squared-L2 kernel, FP32 statistics",
        runtime = RuntimeInformation.FrameworkDescription, os = RuntimeInformation.OSDescription,
        cpu = Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER"), logical_processors = Environment.ProcessorCount,
        vector256_accelerated = Vector256.IsHardwareAccelerated, threads, dims, seeds, budgets, timing_repeats = repeats,
        tiered_compilation_environment = Environment.GetEnvironmentVariable("DOTNET_TieredCompilation"),
        train_vector_limit = trainLimit, k = 256, codebook_storage = "FP16", patch_seed = patchSeed,
        quality_scope = "selected matrices only; gaussian linear probes are synthetic, not real hidden activations; evaluate generated GGUF copies separately for perplexity",
        tensors = tensors.Select(t => new { t.Name, t.Dims, t.Type, t.Offset })
    }, json));
    // Warm both algorithms and the distance/codec code paths before any timings.
    Warmup(threads);
    var results = new List<ResultRow>();
    foreach (var tensor in tensors)
    {
        var weights = gguf.ReadF16(tensor);
        foreach (var d in dims)
        {
            var n = weights.Length / d;
            var split = Permutation(n, new Random(20261112));
            var nt = Math.Min(trainLimit, n * 4 / 5);
            var nh = Math.Min(16384, n - nt);
            var train = new float[nt * d];
            var hold = new float[nh * d];
            for (var i = 0; i < nt; i++) Array.Copy(weights, split[i] * d, train, i * d, d);
            for (var i = 0; i < nh; i++) Array.Copy(weights, split[nt + i] * d, hold, i * d, d);
            foreach (var seed in seeds)
            {
                var specs = new List<(string Algorithm, int Epochs)> { ("lloyd", 25) };
                specs.AddRange(budgets.Select(b => ("emeans", b)));
                // Alternate the position of Lloyd to reduce systematic thermal/order bias.
                if ((seed & 1) != 0) specs.Reverse();
                foreach (var (algorithm, epochs) in specs)
                {
                    var timings = new double[repeats];
                    Training? trained = null;
                    for (var trial = 0; trial < repeats; trial++)
                    {
                        var next = Train(train, d, seed, algorithm, epochs, threads);
                        if (trained is not null && !trained.Centers.AsSpan().SequenceEqual(next.Centers)) throw new InvalidDataException("Timing repetitions changed the result.");
                        trained = next; timings[trial] = next.Milliseconds;
                    }
                    var trainedResult = trained!;
                    var codebook = trainedResult.Centers.Select(x => (float)(Half)x).ToArray();
                    var encTimer = Stopwatch.StartNew();
                    var labels = Assign(weights, d, codebook, threads);
                    encTimer.Stop();
                    var restored = Decode(labels, codebook, d);
                    var full = Error(weights, restored);
                    var holdLabels = Assign(hold, d, codebook, threads);
                    var held = Error(hold, Decode(holdLabels, codebook, d));
                    var floatHeld = Error(hold, Decode(Assign(hold, d, trainedResult.Centers, threads), trainedResult.Centers, d));
                    var probes = Probe(weights, restored, checked((int)tensor.Dims[0]), checked((int)tensor.Dims[1]));
                    var stem = $"{tensor.Name}.d{d}.seed{seed}.{algorithm}{epochs}";
                    var payload = Path.Combine(output, stem + ".vq");
                    var packTimer = Stopwatch.StartNew();
                    WritePayload(payload, tensor, d, codebook, labels);
                    packTimer.Stop();
                    var roundtrip = ReadPayload(payload);
                    if (!restored.AsSpan().SequenceEqual(roundtrip)) throw new InvalidDataException("VQ serialization roundtrip failed.");
                    if (seed == patchSeed)
                    {
                        var patch = Path.Combine(output, $"reconstructed.d{d}.{algorithm}{epochs}.gguf");
                        if (!File.Exists(patch)) File.Copy(model, patch);
                        gguf.PatchF16(patch, tensor, roundtrip);
                    }
                    var bytes = new FileInfo(payload).Length;
                    var row = new ResultRow(tensor.Name, checked((int)tensor.Dims[1]), checked((int)tensor.Dims[0]), d, seed,
                        algorithm, epochs, nt, nh, trainedResult.Batch, trainedResult.Alpha0, trainedResult.Splits,
                        timings.Order().ElementAt(repeats / 2), string.Join(';', timings.Select(t => t.ToString("F4"))), encTimer.Elapsed.TotalMilliseconds, packTimer.Elapsed.TotalMilliseconds,
                        bytes, weights.LongLength * 2, bytes * 8.0 / weights.Length, weights.LongLength * 2.0 / bytes,
                        full.Nmse, Math.Sqrt(full.Nmse), held.Nmse, floatHeld.Nmse,
                        probes.Nrmse, probes.Cosine, labels.Distinct().Count(), Path.GetFileName(payload));
                    results.Add(row);
                    File.WriteAllText(Path.Combine(output, "results.json"), JsonSerializer.Serialize(results, json));
                    WriteCsv(Path.Combine(output, "results.csv"), results);
                    Console.WriteLine($"{tensor.Name,-26} d={d} seed={seed} {algorithm}{epochs,-2} train={row.TrainMs / 1000:F3}s encode={row.EncodeMs / 1000:F3}s bpw={row.BitsPerWeight:F4} hold-NMSE={row.HoldoutNmse:P3} full-NRMSE={row.WeightNrmse:P2}");
                }
            }
        }
    }
    if (Hash(model) != sourceHash) throw new InvalidDataException("Source model changed during benchmark.");
    VerifyPatches(gguf, model, output, results, patchSeed);
    Summarize(output, results);
    Console.WriteLine($"Completed {results.Count} runs; source hash unchanged; every VQ payload decoded and checked.");
    return 0;
}

static int ActivationExperiment(Dictionary<string, string> opts)
{
    string Get(string k, string? value = null) => opts.GetValueOrDefault(k) ?? value ?? throw new ArgumentException($"Missing {k}");
    var model = Path.GetFullPath(Get("--model")); var output = Path.GetFullPath(Get("--out"));
    var activationDir = Path.GetFullPath(Get("--activation-dir"));
    var validationActs = Path.GetFullPath(Get("--validation-activations"));
    var calibrationText = Path.GetFullPath(Get("--calibration-corpus"));
    var validationText = Path.GetFullPath(Get("--validation-corpus"));
    var kinetic = Path.GetFullPath(Get("--kinetic"));
    var threads = int.Parse(Get("--threads", "6")); var repeats = int.Parse(Get("--timing-repeats", "5"));
    var seeds = Get("--seeds", "42,43,44").Split(',').Select(int.Parse).ToArray();
    var patchSeed = int.Parse(Get("--patch-seed", "42"));
    var validChunks = int.Parse(Get("--validation-chunks", "8"));
    var trainLimit = int.Parse(Get("--train-vectors", "65536"));
    var budgets = Get("--e-epochs", "3,5").Split(',').Select(int.Parse).ToArray();
    if (threads < 1 || threads > Environment.ProcessorCount || repeats < 1 || repeats > 31 || trainLimit < 1024 || validChunks < 1 ||
        !seeds.Contains(patchSeed) || seeds.Distinct().Count() != seeds.Length || budgets.Any(b => b < 1 || b > 100) || budgets.Distinct().Count() != budgets.Length)
        throw new ArgumentException("Invalid activation experiment configuration.");
    if (Directory.Exists(output) || File.Exists(output)) throw new IOException("Use a fresh output directory.");
    if (Hash(calibrationText) == Hash(validationText)) throw new ArgumentException("Calibration and validation corpora must differ.");
    if (!File.Exists(kinetic)) throw new FileNotFoundException("Kinetic executable missing.");
    var gguf = Gguf.Open(model);
    var names = Get("--tensors", "blk.0.attn_q.weight,blk.0.ffn_down.weight,blk.15.attn_q.weight,blk.15.ffn_down.weight,blk.29.attn_q.weight,blk.29.ffn_down.weight").Split(',');
    var tensors = names.Select(name => gguf.Tensors.Single(t => t.Name == name)).ToArray();
    foreach (var t in tensors)
    {
        if (t.Type != 1 || t.Dims.Length != 2 || t.Dims[0] % 2 != 0) throw new ArgumentException("Activation experiment requires even-width FP16 matrices.");
        foreach (var directory in new[] { activationDir, validationActs })
            if (!File.Exists(Path.Combine(directory, t.Name + ".act"))) throw new FileNotFoundException($"Missing real activations: {t.Name}");
    }
    if (bool.Parse(Get("--dry-run", "false")))
    {
        Console.WriteLine($"DRY RUN: {tensors.Length} matrices, {seeds.Length} seeds, {budgets.Length + 1} algorithms/budgets, 6 scaling recipes; output {output}; validation-only sensitivity/selection; source unchanged. No files written or inference processes started.");
        return 0;
    }
    var totalTimer = Stopwatch.StartNew();
    Directory.CreateDirectory(output);
    Directory.CreateDirectory(Path.Combine(output, "validation-logs"));
    var sourceHash = Hash(model); var json = new JsonSerializerOptions { WriteIndented = true };
    var specs = new List<(string Algorithm, int Epochs)> { ("lloyd", 25) };
    specs.AddRange(budgets.Select(b => ("emeans", b)));
    File.WriteAllText(Path.Combine(output, "manifest.json"), JsonSerializer.Serialize(new {
        schema = 2, model, source_sha256 = sourceHash, source_bytes = new FileInfo(model).Length,
        tool_sha256 = Hash(SourcePath()), kinetic_sha256 = Hash(kinetic), kinetic_commit = GitHead(),
        runtime = RuntimeInformation.FrameworkDescription, threads, seeds, patch_seed = patchSeed, timing_repeats = repeats,
        train_vector_limit = trainLimit, dimension = 2, clusters = 256, nominal_index_bits_per_weight = 4,
        channel_exponents = new[] { 0.0, 0.5, 1.0 }, row_normalization = new[] { false, true },
        calibration_corpus = calibrationText, calibration_sha256 = Hash(calibrationText),
        validation_corpus = validationText, validation_sha256 = Hash(validationText), validation_context = 512, validation_chunks = validChunks,
        calibration_capture = File.ReadAllText(Path.Combine(activationDir, "capture.txt")),
        validation_capture = File.ReadAllText(Path.Combine(validationActs, "capture.txt")),
        activation_files = tensors.Select(t => new { t.Name, train_sha256 = Hash(Path.Combine(activationDir, t.Name + ".act")), validation_sha256 = Hash(Path.Combine(validationActs, t.Name + ".act")) }),
        codec = "KVQACT01: 24-byte header, 256x2 FP16 codebook, FP16 per-input-channel and per-output-row scales, byte indices; all recipes including raw store both scale tables",
        selection = "No test corpus access. Local: minimum real-output NMSE on 128 evenly spaced validation activations. Sensitivity: single-matrix validation NLL ranking, then greedy validation NLL selection from raw plus best two nonraw local recipes; all six matrices always quantized."
    }, json));
    Warmup(threads);
    var results = new List<ActivationRow>();
    foreach (var tensor in tensors)
    {
        var weights = gguf.ReadF16(tensor); var cols = (int)tensor.Dims[0]; var rows = (int)tensor.Dims[1];
        var calib = ReadActivations(Path.Combine(activationDir, tensor.Name + ".act"), cols);
        var validation = ReadActivations(Path.Combine(validationActs, tensor.Name + ".act"), cols);
        var rms = new float[cols];
        for (var c = 0; c < cols; c++)
        {
            double sum = 0; for (var i = c; i < calib.Length; i += cols) sum += (double)calib[i] * calib[i];
            rms[c] = (float)Math.Sqrt(sum / (calib.Length / cols));
        }
        var typical = Math.Exp(rms.Select(v => Math.Log(Math.Max(v, 1e-8))).Average());
        foreach (var exponent in new[] { 0.0, 0.5, 1.0 }) foreach (var normalizeRows in new[] { false, true })
        {
            var channel = rms.Select(v => (float)(Half)Math.Clamp(Math.Pow(Math.Max(v, 1e-8) / typical, exponent), 0.125, 8)).ToArray();
            var rowScales = Enumerable.Repeat(1f, rows).ToArray();
            var scaled = new float[weights.Length];
            double totalEnergy = 0;
            for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++)
            {
                var i = r * cols + c; scaled[i] = weights[i] * channel[c]; totalEnergy += (double)scaled[i] * scaled[i];
            }
            if (normalizeRows)
            {
                var typicalRowRms = Math.Sqrt(totalEnergy / weights.Length);
                for (var r = 0; r < rows; r++)
                {
                    double energy = 0; for (var c = 0; c < cols; c++) energy += (double)scaled[r * cols + c] * scaled[r * cols + c];
                    rowScales[r] = (float)(Half)Math.Clamp(Math.Sqrt(energy / cols) / Math.Max(typicalRowRms, 1e-12), 1.0 / 64, 64);
                    for (var c = 0; c < cols; c++) scaled[r * cols + c] /= rowScales[r];
                }
            }
            var split = Permutation(weights.Length / 2, new Random(20261112));
            var nt = Math.Min(trainLimit, split.Length * 4 / 5); var train = new float[nt * 2];
            for (var i = 0; i < nt; i++) Array.Copy(scaled, split[i] * 2, train, i * 2, 2);
            foreach (var seed in seeds)
            {
                var order = (seed & 1) == 0 ? specs : specs.AsEnumerable().Reverse();
                foreach (var (algorithm, epochs) in order)
                {
                    var samples = new double[repeats]; Training? fitted = null;
                    for (var trial = 0; trial < repeats; trial++)
                    {
                        var next = Train(train, 2, seed, algorithm, epochs, threads);
                        if (fitted is not null && !fitted.Centers.AsSpan().SequenceEqual(next.Centers)) throw new InvalidDataException("Nondeterministic activation training.");
                        fitted = next; samples[trial] = next.Milliseconds;
                    }
                    var centers = fitted!.Centers.Select(v => (float)(Half)v).ToArray();
                    var timer = Stopwatch.StartNew(); var labels = Assign(scaled, 2, centers, threads); timer.Stop();
                    var recipe = $"a{exponent.ToString("0.0", CultureInfo.InvariantCulture)}r{(normalizeRows ? 1 : 0)}";
                    var payload = $"{tensor.Name}.seed{seed}.{algorithm}{epochs}.{recipe}.avq";
                    WriteActivationPayload(Path.Combine(output, payload), rows, cols, centers, channel, rowScales, labels);
                    var restored = ReadActivationPayload(Path.Combine(output, payload));
                    var exact = RestoreActivation(labels, centers, channel, rowScales, rows, cols);
                    if (!restored.AsSpan().SequenceEqual(exact)) throw new InvalidDataException("Activation codec roundtrip failed.");
                    var outputNmse = RealOutputError(weights, restored, validation, cols, rows, threads);
                    var row = new ActivationRow(tensor.Name, algorithm, epochs, seed, recipe, exponent, normalizeRows,
                        nt, calib.Length / cols, validation.Length / cols, Median(samples), string.Join(';', samples.Select(v => v.ToString("F4"))),
                        timer.Elapsed.TotalMilliseconds, new FileInfo(Path.Combine(output, payload)).Length,
                        Error(weights, restored).Nmse, outputNmse, payload);
                    results.Add(row);
                    File.WriteAllText(Path.Combine(output, "activation-results.json"), JsonSerializer.Serialize(results, json));
                    Console.WriteLine($"{tensor.Name,-26} {algorithm}{epochs} seed={seed} {recipe} train={row.TrainMs:F2}ms real-output-NMSE={outputNmse:P3}");
                }
            }
        }
    }
    var candidateWallMs = totalTimer.Elapsed.TotalMilliseconds;
    var selectionTimer = Stopwatch.StartNew();
    // Full-model selection uses validation only, and only the predeclared patch seed.
    var work = Path.Combine(output, "validation-work.gguf");
    var trials = new List<object>(); var verification = new List<object>();
    double Score(string stem, IEnumerable<ActivationRow> selection)
    {
        File.Copy(model, work, true); // owned file in this newly created experiment directory
        foreach (var row in selection) gguf.PatchF16(work, tensors.Single(t => t.Name == row.Tensor), ReadActivationPayload(Path.Combine(output, row.Payload)));
        var run = RunProcess(kinetic, new[] { "perplexity", work, validationText, "-c", "512", "--chunks", validChunks.ToString(), "-t", threads.ToString() }, Path.Combine(output, "validation-logs", stem + ".log"));
        var ppl = Number(run.Stdout, @"Final estimate: PPL = ([0-9.]+)");
        trials.Add(new { name = stem, ppl, payloads = selection.Select(r => r.Payload).ToArray(), milliseconds = run.Milliseconds });
        File.WriteAllText(Path.Combine(output, "validation-trials.json"), JsonSerializer.Serialize(trials, json));
        Console.WriteLine($"validation {stem}: PPL={ppl:F4}");
        return ppl;
    }
    void Materialize(string name, List<ActivationRow> selection)
    {
        var path = Path.Combine(output, $"reconstructed.{name}.gguf"); File.Copy(model, path);
        foreach (var row in selection) gguf.PatchF16(path, tensors.Single(t => t.Name == row.Tensor), ReadActivationPayload(Path.Combine(output, row.Payload)));
        var patched = Gguf.Open(path);
        using var a = File.OpenRead(model); using var b = File.OpenRead(path); long position = 0;
        foreach (var row in selection.OrderBy(r => tensors.Single(t => t.Name == r.Tensor).Offset))
        {
            var t = tensors.Single(t => t.Name == row.Tensor); var restored = ReadActivationPayload(Path.Combine(output, row.Payload));
            if (!restored.AsSpan().SequenceEqual(patched.ReadF16(t))) throw new InvalidDataException("Patched activation weights differ from payload.");
            CompareRange(a, b, position, t.Offset); position = t.Offset + restored.LongLength * 2;
        }
        CompareRange(a, b, position, a.Length);
        verification.Add(new { file = Path.GetFileName(path), sha256 = Hash(path), unchanged_outside_selected_tensors = true, matches_payloads = true,
            payload_bytes = selection.Sum(r => r.PayloadBytes), selected_original_bytes = selection.Sum(r => checked((long)tensors.Single(t => t.Name == r.Tensor).Dims.Aggregate(2UL, (x, y) => x * y))), payloads = selection.Select(r => r.Payload).ToArray() });
        File.WriteAllText(Path.Combine(output, "verification.json"), JsonSerializer.Serialize(verification, json));
    }
    var baselinePpl = Score("baseline", Array.Empty<ActivationRow>());
    var sensitivity = new List<object>();
    var profiles = new List<object>();
    foreach (var (algorithm, epochs) in specs)
    {
        var available = results.Where(r => r.Algorithm == algorithm && r.Epochs == epochs && r.Seed == patchSeed).ToArray();
        var raw = names.Select(n => available.Single(r => r.Tensor == n && r.Recipe == "a0.0r0")).ToList();
        Materialize($"raw.{algorithm}{epochs}", raw);
        var rawPpl = Score($"raw-combined.{algorithm}{epochs}", raw);
        var local = names.Select(n => available.Where(r => r.Tensor == n).OrderBy(r => r.OutputNmse).ThenBy(r => r.Recipe).First()).ToList();
        Materialize($"local.{algorithm}{epochs}", local);
        var localPpl = Score($"local-combined.{algorithm}{epochs}", local);
        var ranked = raw.Select(r => {
            var ppl = Score($"sensitivity.{algorithm}{epochs}.{r.Tensor}", new[] { r });
            return (r.Tensor, DeltaNll: Math.Log(ppl / baselinePpl));
        }).OrderByDescending(r => r.DeltaNll).ToArray();
        sensitivity.Add(new { algorithm, epochs, layers = ranked.Select(r => new { tensor = r.Tensor, delta_nll = r.DeltaNll }).ToArray() });
        File.WriteAllText(Path.Combine(output, "sensitivity.json"), JsonSerializer.Serialize(sensitivity, json));
        var selected = new List<ActivationRow>();
        var sensitivityPpl = double.NaN;
        foreach (var (name, _) in ranked)
        {
            var shortlist = available.Where(r => r.Tensor == name && r.Recipe != "a0.0r0").OrderBy(r => r.OutputNmse).ThenBy(r => r.Recipe).Take(2)
                .Prepend(raw.Single(r => r.Tensor == name)).ToArray();
            var candidates = shortlist.Select(r => (Row: r, Ppl: Score($"greedy.{algorithm}{epochs}.{name}.{r.Recipe}", selected.Concat(new[] { r })))).OrderBy(r => r.Ppl).ThenBy(r => r.Row.Recipe).ToArray();
            selected.Add(candidates[0].Row);
            sensitivityPpl = candidates[0].Ppl;
        }
        Materialize($"sensitivity.{algorithm}{epochs}", selected);
        var best = new[] { (Profile: "raw", Ppl: rawPpl), (Profile: "local", Ppl: localPpl), (Profile: "sensitivity", Ppl: sensitivityPpl) }.OrderBy(p => p.Ppl).First();
        profiles.Add(new { algorithm, epochs, raw_ppl = rawPpl, local_ppl = localPpl, sensitivity_ppl = sensitivityPpl,
            selected_by_validation = best.Profile, selected_recipes = selected.Select(r => new { r.Tensor, r.Recipe }).ToArray() });
        File.WriteAllText(Path.Combine(output, "validation-profiles.json"), JsonSerializer.Serialize(profiles, json));
    }
    if (Hash(model) != sourceHash) throw new InvalidDataException("Source model changed.");
    File.WriteAllText(Path.Combine(output, "timings.json"), JsonSerializer.Serialize(new { candidate_wall_ms = candidateWallMs,
        selection_wall_ms = selectionTimer.Elapsed.TotalMilliseconds, total_wall_ms = totalTimer.Elapsed.TotalMilliseconds,
        candidates = results.Count, validation_inference_runs = trials.Count, reconstructed_models = verification.Count }, json));
    Console.WriteLine($"Completed {results.Count} activation-aware candidates and {verification.Count} verified full-model copies. No test data accessed.");
    return 0;
}

static float[] ReadActivations(string path, int expectedCols)
{
    using var r = new BinaryReader(File.OpenRead(path));
    if (Encoding.ASCII.GetString(r.ReadBytes(8)) != "KACT0001") throw new InvalidDataException("Invalid activation magic.");
    var rows = r.ReadInt32(); var cols = r.ReadInt32();
    if (rows < 128 || rows > 65536 || cols != expectedCols || r.BaseStream.Length != 16L + 4L * rows * cols) throw new InvalidDataException("Invalid activation shape/size.");
    var x = new float[checked(rows * cols)];
    for (var i = 0; i < x.Length; i++) x[i] = r.ReadSingle();
    if (x.Any(v => !float.IsFinite(v))) throw new InvalidDataException("Nonfinite captured activation.");
    return x;
}

static float[] RestoreActivation(byte[] labels, float[] centers, float[] channel, float[] rowScales, int rows, int cols)
{
    var output = new float[checked(rows * cols)];
    for (var r = 0; r < rows; r++) for (var c = 0; c < cols; c++)
        output[r * cols + c] = (float)(Half)(centers[labels[(r * cols + c) / 2] * 2 + c % 2] * rowScales[r] / channel[c]);
    return output;
}

static void WriteActivationPayload(string path, int rows, int cols, float[] centers, float[] channel, float[] rowScales, byte[] labels)
{
    using var w = new BinaryWriter(File.Open(path, FileMode.CreateNew));
    w.Write(Encoding.ASCII.GetBytes("KVQACT01")); w.Write(rows); w.Write(cols); w.Write(2); w.Write(256);
    foreach (var data in new[] { centers, channel, rowScales }) foreach (var v in data) w.Write(BitConverter.HalfToUInt16Bits((Half)v));
    w.Write(labels);
}

static float[] ReadActivationPayload(string path)
{
    using var r = new BinaryReader(File.OpenRead(path));
    if (Encoding.ASCII.GetString(r.ReadBytes(8)) != "KVQACT01") throw new InvalidDataException("Invalid activation VQ magic.");
    var rows = r.ReadInt32(); var cols = r.ReadInt32(); var d = r.ReadInt32(); var k = r.ReadInt32();
    if (rows < 1 || cols < 2 || cols % 2 != 0 || d != 2 || k != 256 || r.BaseStream.Length != 24L + 1024 + 2L * (rows + cols) + (long)rows * cols / 2)
        throw new InvalidDataException("Invalid activation VQ shape/size.");
    var centers = new float[512]; var channel = new float[cols]; var rowScales = new float[rows];
    foreach (var data in new[] { centers, channel, rowScales }) for (var i = 0; i < data.Length; i++) data[i] = (float)BitConverter.UInt16BitsToHalf(r.ReadUInt16());
    if (centers.Any(v => !float.IsFinite(v)) || channel.Concat(rowScales).Any(v => !float.IsFinite(v) || v <= 0)) throw new InvalidDataException("Invalid activation VQ scales.");
    return RestoreActivation(r.ReadBytes(checked(rows * cols / 2)), centers, channel, rowScales, rows, cols);
}

static double RealOutputError(float[] weights, float[] restored, float[] activations, int cols, int rows, int threads)
{
    const int probes = 128;
    var difference = weights.Zip(restored, (a, b) => b - a).ToArray();
    var errors = new double[rows]; var energies = new double[rows]; var count = activations.Length / cols;
    Parallel.For(0, rows, new ParallelOptions { MaxDegreeOfParallelism = threads }, row => {
        var at = row * cols;
        for (var p = 0; p < probes; p++)
        {
            var input = p * (count - 1) / (probes - 1) * cols;
            var error = DotVectors(difference, at, activations, input, cols);
            var y = DotVectors(weights, at, activations, input, cols);
            errors[row] += (double)error * error; energies[row] += (double)y * y;
        }
    });
    return errors.Sum() / energies.Sum();
}

static float DotVectors(float[] a, int ai, float[] b, int bi, int n)
{
    var sum = Vector256<float>.Zero; var c = 0;
    ref var ar = ref MemoryMarshal.GetArrayDataReference(a); ref var br = ref MemoryMarshal.GetArrayDataReference(b);
    if (Vector256.IsHardwareAccelerated)
        for (; c <= n - 8; c += 8) sum += Vector256.LoadUnsafe(ref ar, (nuint)(ai + c)) * Vector256.LoadUnsafe(ref br, (nuint)(bi + c));
    var value = Vector256.Sum(sum);
    for (; c < n; c++) value += a[ai + c] * b[bi + c];
    return value;
}

static Training Train(float[] x, int d, int seed, string algorithm, int epochs, int threads)
{
    const int k = 256;
    var sw = Stopwatch.StartNew();
    var n = x.Length / d;
    var init = Permutation(n, new Random(seed));
    var centers = new float[k * d];
    for (var j = 0; j < k; j++) Array.Copy(x, init[j] * d, centers, j * d, d);
    var batch = n;
    var alpha0 = 1f;
    if (algorithm == "emeans")
    {
        var raw = 0.00439175 * Math.Pow(k, 0.134749) * Math.Pow(n, 0.876473) * Math.Pow(epochs, 0.813344);
        var low = Math.Pow(2, Math.Floor(Math.Log2(Math.Max(raw, 1))));
        batch = Math.Clamp((int)(raw / low < low * 2 / raw ? low : low * 2), 1, n / 2);
        alpha0 = (float)Math.Clamp(0.8244 * Math.Pow(k, -0.4617) * Math.Pow(batch, 0.9003) * Math.Pow(n, -0.4498) * Math.Pow(epochs, -0.1057), 1e-4, 1);
    }
    var usage = Enumerable.Repeat(1f / 65536, k).ToArray();
    var rng = new Random(seed);
    var totalSteps = epochs * ((n + batch - 1) / batch);
    var step = 0;
    var splits = 0;
    var sums = new float[k * d];
    var counts = new int[k];
    var closingSums = new float[k * d];
    var closingCounts = new int[k];
    for (var epoch = 0; epoch < epochs; epoch++)
    {
        var order = algorithm == "emeans" ? Permutation(n, new Random(unchecked(seed * 1000003 + epoch))) : null;
        for (var begin = 0; begin < n; begin += batch)
        {
            var length = Math.Min(batch, n - begin);
            var labels = Assign(x, d, centers, threads, order, begin, length);
            Array.Clear(sums); Array.Clear(counts);
            for (var i = 0; i < length; i++)
            {
                var row = (order is null ? begin + i : order[begin + i]) * d;
                var j = labels[i];
                counts[j]++;
                for (var c = 0; c < d; c++) sums[j * d + c] += x[row + c];
            }
            var alpha = algorithm == "lloyd" ? 1f : alpha0 * (float)(0.1 + 0.9 * 0.5 * (1 + Math.Cos(Math.PI * step / Math.Max(1, totalSteps))));
            for (var j = 0; j < k; j++)
            {
                if (algorithm == "emeans") usage[j] *= 1 - alpha;
                if (counts[j] > 0)
                {
                    var mass = usage[j] + alpha * counts[j];
                    var weight = algorithm == "lloyd" ? 1f : alpha * counts[j] / mass;
                    for (var c = 0; c < d; c++)
                    {
                        var mean = sums[j * d + c] / counts[j];
                        centers[j * d + c] = weight == 1 ? mean : centers[j * d + c] + weight * (mean - centers[j * d + c]);
                    }
                    usage[j] = mass;
                }
                if (epoch == epochs - 1)
                {
                    closingCounts[j] += counts[j];
                    for (var c = 0; c < d; c++) closingSums[j * d + c] += sums[j * d + c];
                }
            }
            step++;
            if (algorithm == "emeans" && d > 2 && step % 5 == 0) splits += Revive(centers, usage, d, rng);
        }
    }
    if (algorithm == "emeans")
        for (var j = 0; j < k; j++) if (closingCounts[j] > 0)
            for (var c = 0; c < d; c++) centers[j * d + c] = closingSums[j * d + c] / closingCounts[j];
    sw.Stop();
    return new(centers, batch, alpha0, splits, sw.Elapsed.TotalMilliseconds);
}

static int Revive(float[] centers, float[] usage, int d, Random rng)
{
    var threshold = usage.Sum() / 256 * 0.1f;
    var dead = Enumerable.Range(0, 256).Where(j => usage[j] < threshold).ToArray();
    if (dead.Length == 0) return 0;
    // Weighted sampling without replacement: Gumbel top-k, as upstream.
    var donors = Enumerable.Range(0, 256).Where(j => usage[j] >= threshold && usage[j] > 0)
        .Select(j => (Index: j, Key: Math.Log(usage[j]) - Math.Log(-Math.Log(Math.Max(rng.NextDouble(), double.Epsilon)))))
        .OrderByDescending(x => x.Key).Take(dead.Length).ToArray();
    for (var s = 0; s < donors.Length; s++)
    {
        var donor = donors[s].Index; var child = dead[s];
        var identical = true;
        for (var c = 0; c < d; c++)
        {
            var v = centers[donor * d + c];
            var offset = (c % 2 == 0 ? 1 : -1) * 1e-4f;
            centers[donor * d + c] = v * (1 - offset);
            centers[child * d + c] = v * (1 + offset);
            identical &= centers[donor * d + c] == centers[child * d + c];
        }
        if (identical) centers[child * d] = float.BitIncrement(centers[child * d]);
        usage[donor] /= 2; usage[child] = usage[donor];
    }
    return donors.Length;
}

static byte[] Assign(float[] x, int d, float[] centers, int threads, int[]? order = null, int begin = 0, int length = -1)
{
    if (length < 0) length = x.Length / d;
    var labels = new byte[length];
    var soa = new float[256 * d];
    for (var j = 0; j < 256; j++) for (var c = 0; c < d; c++) soa[c * 256 + j] = centers[j * d + c];
    var blocks = (length + 1023) / 1024;
    Parallel.For(0, blocks, new ParallelOptions { MaxDegreeOfParallelism = threads }, block => {
        var end = Math.Min(length, (block + 1) * 1024);
        for (var i = block * 1024; i < end; i++)
        {
            var row = (order is null ? begin + i : order[begin + i]) * d;
            labels[i] = (byte)Nearest(x, row, soa, d);
        }
    });
    return labels;
}

static int Nearest(float[] x, int row, float[] soa, int d)
{
    if (!Vector256.IsHardwareAccelerated) return NearestScalar(x, row, soa, d);
    var best = Vector256.Create(float.PositiveInfinity);
    var ids = Vector256<int>.Zero;
    var lanes = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
    ref var origin = ref MemoryMarshal.GetArrayDataReference(soa);
    for (var j = 0; j < 256; j += 8)
    {
        var dist = Vector256<float>.Zero;
        for (var c = 0; c < d; c++)
        {
            var delta = Vector256.Create(x[row + c]) - Vector256.LoadUnsafe(ref origin, (nuint)(c * 256 + j));
            dist += delta * delta;
        }
        var mask = Vector256.LessThan(dist, best);
        ids = Vector256.ConditionalSelect(mask.AsInt32(), lanes + Vector256.Create(j), ids);
        best = Vector256.Min(dist, best);
    }
    var min = float.PositiveInfinity; var index = 0;
    for (var c = 0; c < 8; c++)
        if (best[c] < min || (best[c] == min && ids[c] < index)) { min = best[c]; index = ids[c]; }
    return index;
}

static int NearestScalar(float[] x, int row, float[] soa, int d)
{
    var min = float.PositiveInfinity; var index = 0;
    for (var j = 0; j < 256; j++)
    {
        var distance = 0f;
        for (var c = 0; c < d; c++) { var delta = x[row + c] - soa[c * 256 + j]; distance += delta * delta; }
        if (distance < min) { min = distance; index = j; }
    }
    return index;
}

static float[] Decode(byte[] labels, float[] centers, int d)
{
    var restored = new float[labels.Length * d];
    for (var i = 0; i < labels.Length; i++) Array.Copy(centers, labels[i] * d, restored, i * d, d);
    return restored;
}

static (double Nmse, double Mse) Error(float[] original, float[] restored)
{
    double error = 0, energy = 0;
    for (var i = 0; i < original.Length; i++) { var delta = (double)original[i] - restored[i]; error += delta * delta; energy += (double)original[i] * original[i]; }
    return (energy > 0 ? error / energy : error == 0 ? 0 : double.PositiveInfinity, error / original.Length);
}

static (double Nrmse, double Cosine) Probe(float[] weights, float[] restored, int cols, int rows)
{
    // Fixed 32 N(0,I) inputs. This is a linear distortion check, not language quality.
    var rng = new Random(12026);
    double error = 0, energy = 0, reconEnergy = 0, product = 0;
    var x = new float[cols];
    for (var p = 0; p < 32; p++)
    {
        for (var c = 0; c < cols; c++) x[c] = (float)(Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
        for (var r = 0; r < rows; r++)
        {
            double y = 0, z = 0;
            for (var c = 0; c < cols; c++) { y += (double)weights[r * cols + c] * x[c]; z += (double)restored[r * cols + c] * x[c]; }
            error += (y - z) * (y - z); energy += y * y; reconEnergy += z * z; product += y * z;
        }
    }
    return (Math.Sqrt(error / energy), product / Math.Sqrt(energy * reconEnergy));
}

static void WritePayload(string path, Tensor tensor, int d, float[] centers, byte[] labels)
{
    using var writer = new BinaryWriter(File.Open(path, FileMode.CreateNew, FileAccess.Write));
    writer.Write(Encoding.ASCII.GetBytes("KVQEXP01"));
    writer.Write(checked((int)tensor.Dims[1])); writer.Write(checked((int)tensor.Dims[0]));
    writer.Write(d); writer.Write(256); // 24-byte complete header; fixed FP16 codebook, byte indices.
    foreach (var value in centers) writer.Write(BitConverter.HalfToUInt16Bits((Half)value));
    writer.Write(labels);
}

static float[] ReadPayload(string path)
{
    using var reader = new BinaryReader(File.OpenRead(path));
    if (Encoding.ASCII.GetString(reader.ReadBytes(8)) != "KVQEXP01") throw new InvalidDataException("Invalid VQ magic.");
    var rows = reader.ReadInt32(); var cols = reader.ReadInt32(); var d = reader.ReadInt32(); var k = reader.ReadInt32();
    if (rows <= 0 || cols <= 0 || d is not (2 or 4 or 8) || k != 256 || cols % d != 0) throw new InvalidDataException("Invalid VQ dimensions.");
    var n = checked(rows * cols / d);
    if (reader.BaseStream.Length != 24L + k * d * 2 + n) throw new InvalidDataException("Invalid VQ payload length.");
    var centers = new float[k * d];
    for (var i = 0; i < centers.Length; i++) centers[i] = (float)BitConverter.UInt16BitsToHalf(reader.ReadUInt16());
    return Decode(reader.ReadBytes(n), centers, d);
}

static int[] Permutation(int n, Random random)
{
    var order = Enumerable.Range(0, n).ToArray();
    random.Shuffle(order);
    return order;
}

static string Hash(string path, bool required = true)
{
    if (!required && !File.Exists(path)) return "unavailable; hash source script separately";
    using var stream = File.OpenRead(path);
    return Convert.ToHexStringLower(SHA256.HashData(stream));
}

static string SourcePath([CallerFilePath] string source = "") => source;

static string GitHead()
{
    using var p = Process.Start(new ProcessStartInfo("git", "rev-parse HEAD") { RedirectStandardOutput = true, UseShellExecute = false });
    var value = p!.StandardOutput.ReadToEnd().Trim(); p.WaitForExit(); return value;
}

static void WriteCsv(string path, List<ResultRow> rows)
{
    var properties = typeof(ResultRow).GetProperties();
    using var writer = new StreamWriter(path);
    writer.WriteLine(string.Join(',', properties.Select(p => p.Name)));
    foreach (var row in rows) writer.WriteLine(string.Join(',', properties.Select(p => Convert.ToString(p.GetValue(row), CultureInfo.InvariantCulture))));
}

static void Warmup(int threads)
{
    var rng = new Random(7);
    foreach (var d in new[] { 2, 4, 8 })
    {
        var x = Enumerable.Range(0, 2048 * d).Select(_ => (float)rng.NextDouble()).ToArray();
        Train(x, d, 7, "lloyd", 2, threads); Train(x, d, 7, "emeans", 2, threads);
    }
}

static void SelfTest()
{
    var rng = new Random(18);
    foreach (var d in new[] { 2, 4, 8 })
    {
        var soa = Enumerable.Range(0, 256 * d).Select(_ => (float)rng.NextDouble()).ToArray();
        var x = Enumerable.Range(0, 512 * d).Select(_ => (float)rng.NextDouble()).ToArray();
        for (var i = 0; i < 512; i++)
            if (Nearest(x, i * d, soa, d) != NearestScalar(x, i * d, soa, d)) throw new Exception("SIMD/scalar nearest mismatch.");
        Array.Clear(soa);
        if (Nearest(x, 0, soa, d) != 0) throw new Exception("Tie must select lowest centroid index.");
        var centers = new float[256 * d]; var usage = Enumerable.Repeat(1f, 256).ToArray(); usage[4] = 0;
        if (d > 2 && Revive(centers, usage, d, rng) != 1) throw new Exception("Revival failed.");
        var massBefore = usage.Sum();
        if (massBefore != 255) throw new Exception("Revival must conserve mass.");
        // Repeated vectors exercise deterministic training and empty clusters.
        var exact = new float[1024 * d];
        for (var i = 0; i < 1024; i++) for (var c = 0; c < d; c++) exact[i * d + c] = i % 256;
        foreach (var algorithm in new[] { "lloyd", "emeans" })
        {
            var a = Train(exact, d, 42, algorithm, 3, 2);
            var b = Train(exact, d, 42, algorithm, 3, 2);
            if (!a.Centers.AsSpan().SequenceEqual(b.Centers) || a.Centers.Any(v => !float.IsFinite(v))) throw new Exception("Training determinism/finiteness failed.");
        }
        var distinct = new float[256 * d];
        for (var i = 0; i < 256; i++) for (var c = 0; c < d; c++) distinct[i * d + c] = i;
        foreach (var algorithm in new[] { "lloyd", "emeans" })
        {
            // Lloyd must preserve a perfect k=n codebook. At n/k=1,
            // e-means revival can intentionally perturb even that codebook;
            // use identical zero inputs to test its exact empty-cluster case.
            var input = algorithm == "lloyd" || d == 2 ? distinct : new float[256 * d];
            var fitted = Train(input, d, 42, algorithm, 3, 2);
            if (Error(input, Decode(Assign(input, d, fitted.Centers, 2), fitted.Centers, d)).Nmse != 0)
                throw new Exception($"Exact reconstruction fixture failed: {algorithm}, d={d}.");
        }
    }
    var inverse = RestoreActivation(new byte[] { 0 }, new float[] { 2, 6 }, new float[] { 2, 3 }, new float[] { 0.5f }, 1, 2);
    if (!inverse.AsSpan().SequenceEqual(new float[] { 0.5f, 1 })) throw new Exception("Inverse channel/row scaling fixture failed.");
    var identity = new float[] { 1, 0, 0, 1 }; var twice = new float[] { 2, 0, 0, 2 };
    var inputs = Enumerable.Range(0, 256).Select(i => (float)(i % 7 + 1)).ToArray();
    if (RealOutputError(identity, twice, inputs, 2, 2, 2) != 1) throw new Exception("Real output NMSE fixture failed.");
    Console.WriteLine("PASS: SIMD/scalar distances (1536 vectors), tie handling, revival/mass conservation, deterministic finite training, exact reconstruction fixtures, inverse activation scaling and known output NMSE.");
}

static void VerifyPatches(Gguf original, string source, string output, List<ResultRow> results, int seed)
{
    var verified = new List<object>();
    foreach (var group in results.Where(r => r.Seed == seed).GroupBy(r => (r.Dimension, r.Algorithm, r.Epochs)))
    {
        var path = Path.Combine(output, $"reconstructed.d{group.Key.Dimension}.{group.Key.Algorithm}{group.Key.Epochs}.gguf");
        var patched = Gguf.Open(path);
        using var a = File.OpenRead(source); using var b = File.OpenRead(path);
        if (a.Length != b.Length) throw new InvalidDataException("Reconstructed model size changed.");
        long position = 0;
        foreach (var row in group.OrderBy(r => original.Tensors.Single(t => t.Name == r.Tensor).Offset))
        {
            var tensor = original.Tensors.Single(t => t.Name == row.Tensor);
            var expected = ReadPayload(Path.Combine(output, row.Payload));
            if (!expected.AsSpan().SequenceEqual(patched.ReadF16(tensor))) throw new InvalidDataException("Reconstructed GGUF/payload mismatch.");
            CompareRange(a, b, position, tensor.Offset);
            position = tensor.Offset + expected.LongLength * 2;
        }
        CompareRange(a, b, position, a.Length);
        verified.Add(new { file = Path.GetFileName(path), sha256 = Hash(path), unchanged_outside_selected_tensors = true, matches_payloads = true });
    }
    File.WriteAllText(Path.Combine(output, "verification.json"), JsonSerializer.Serialize(verified, new JsonSerializerOptions { WriteIndented = true }));
}

static void CompareRange(Stream a, Stream b, long begin, long end)
{
    a.Position = b.Position = begin;
    var left = new byte[1024 * 1024]; var right = new byte[left.Length];
    while (a.Position < end)
    {
        var n = (int)Math.Min(left.Length, end - a.Position);
        a.ReadExactly(left.AsSpan(0, n)); b.ReadExactly(right.AsSpan(0, n));
        if (!left.AsSpan(0, n).SequenceEqual(right.AsSpan(0, n))) throw new InvalidDataException("Model bytes outside selected tensors changed.");
    }
}

static double Median(IEnumerable<double> values)
{
    var sorted = values.Order().ToArray(); var m = sorted.Length / 2;
    return sorted.Length % 2 == 0 ? (sorted[m - 1] + sorted[m]) / 2 : sorted[m];
}

static void Summarize(string output, List<ResultRow> results)
{
    var summary = results.GroupBy(r => (r.Dimension, r.Algorithm, r.Epochs)).Select(group => {
        var pairs = group.Select(row => (Row: row, Base: results.Single(b => b.Tensor == row.Tensor && b.Dimension == row.Dimension && b.Seed == row.Seed && b.Algorithm == "lloyd"))).ToArray();
        var oneSeed = group.Where(r => r.Seed == group.First().Seed).ToArray();
        return new {
            group.Key.Dimension, group.Key.Algorithm, group.Key.Epochs, cases = group.Count(),
            median_train_ms = Median(group.Select(r => r.TrainMs)),
            median_paired_train_speedup = Median(pairs.Select(p => p.Base.TrainMs / p.Row.TrainMs)),
            median_paired_train_encode_speedup = Median(pairs.Select(p => (p.Base.TrainMs + p.Base.EncodeMs) / (p.Row.TrainMs + p.Row.EncodeMs))),
            median_paired_holdout_mse_ratio = Median(pairs.Select(p => p.Row.HoldoutNmse / p.Base.HoldoutNmse)),
            min_paired_holdout_mse_ratio = pairs.Min(p => p.Row.HoldoutNmse / p.Base.HoldoutNmse),
            max_paired_holdout_mse_ratio = pairs.Max(p => p.Row.HoldoutNmse / p.Base.HoldoutNmse),
            holdout_wins_vs_lloyd25 = pairs.Count(p => p.Row.HoldoutNmse < p.Base.HoldoutNmse),
            median_weight_nrmse = Median(group.Select(r => r.WeightNrmse)),
            median_gaussian_probe_nrmse = Median(group.Select(r => r.GaussianProbeNrmse)),
            selected_original_bytes = oneSeed.Sum(r => r.OriginalF16Bytes), selected_payload_bytes = oneSeed.Sum(r => r.PayloadBytes),
            selected_bpw = oneSeed.Sum(r => r.PayloadBytes) * 8.0 / (oneSeed.Sum(r => r.OriginalF16Bytes) / 2),
            selected_compression_ratio = oneSeed.Sum(r => r.OriginalF16Bytes) * 1.0 / oneSeed.Sum(r => r.PayloadBytes)
        };
    }).OrderBy(x => x.Dimension).ThenBy(x => x.Algorithm).ThenBy(x => x.Epochs).ToArray();
    File.WriteAllText(Path.Combine(output, "summary.json"), JsonSerializer.Serialize(summary, new JsonSerializerOptions { WriteIndented = true }));
}

static int Evaluate(Dictionary<string, string> opts)
{
    string Get(string key, string? fallback = null) => opts.GetValueOrDefault(key) ?? fallback ?? throw new ArgumentException($"Missing {key}");
    var experiment = Path.GetFullPath(Get("--evaluate"));
    var output = Path.GetFullPath(Get("--quality-out"));
    var kinetic = Path.GetFullPath(Get("--kinetic"));
    var corpus = Path.GetFullPath(Get("--corpus"));
    var ctx = int.Parse(Get("--ctx", "512")); var chunks = int.Parse(Get("--chunks", "64"));
    var threads = int.Parse(Get("--threads", "6"));
    if (ctx < 16 || chunks < 1 || threads < 1) throw new ArgumentException("Invalid evaluation sizes.");
    if (!File.Exists(kinetic) || !File.Exists(corpus)) throw new FileNotFoundException("Missing kinetic executable or corpus.");
    if (Directory.Exists(output) || File.Exists(output)) throw new IOException("Quality output already exists; use a fresh directory.");
    using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(experiment, "manifest.json")));
    var model = manifest.RootElement.GetProperty("model").GetString()!;
    if (Hash(model) != manifest.RootElement.GetProperty("source_sha256").GetString()) throw new InvalidDataException("Source model hash mismatch.");
    var patches = Directory.GetFiles(experiment, "reconstructed.*.gguf").Order().ToArray();
    if (patches.Length == 0) throw new InvalidDataException("No reconstructed models to evaluate.");
    using var verification = JsonDocument.Parse(File.ReadAllText(Path.Combine(experiment, "verification.json")));
    foreach (var patch in patches)
    {
        var check = verification.RootElement.EnumerateArray().Single(v => v.GetProperty("file").GetString() == Path.GetFileName(patch));
        if (Hash(patch) != check.GetProperty("sha256").GetString()) throw new InvalidDataException("Reconstructed model hash mismatch.");
    }
    if (bool.Parse(Get("--dry-run", "false")))
    {
        Console.WriteLine($"DRY RUN: evaluate source and {patches.Length} verified reconstructed models using {kinetic}; create logs, baseline probabilities and quality.json in {output}; context={ctx}, chunks={chunks}. No processes started or files written.");
        return 0;
    }
    Directory.CreateDirectory(output);
    var baseLogits = Path.Combine(output, "base.kld");
    string[] BaselineArgs() => new[] { "perplexity", model, corpus, "-c", ctx.ToString(), "--chunks", chunks.ToString(), "-t", threads.ToString(), "--kl-divergence-base", baseLogits };
    var baseline = RunProcess(kinetic, BaselineArgs(), Path.Combine(output, "baseline.log"));
    var basePpl = Number(baseline.Stdout, @"Final estimate: PPL = ([0-9.]+)");
    var comparisons = new List<QualityRow>();
    foreach (var path in new[] { model }.Concat(patches))
    {
        var name = path == model ? "baseline-control" : Path.GetFileNameWithoutExtension(path);
        var arguments = new[] { "perplexity", path, "--kl-divergence", baseLogits, "-t", threads.ToString() };
        var run = RunProcess(kinetic, arguments, Path.Combine(output, name + ".log"));
        var ppl = Number(run.Stdout, @"Mean PPL\(Q\)\s*:\s*([0-9.]+)");
        var row = new QualityRow(name, ppl, (ppl / basePpl - 1) * 100,
            Number(run.Stdout, @"Mean\s+KLD\s*:\s*([-0-9.]+)"),
            Number(run.Stdout, @"Same top p\s*:\s*([0-9.]+)"), run.Milliseconds);
        comparisons.Add(row);
        File.WriteAllText(Path.Combine(output, "quality.json"), JsonSerializer.Serialize(new {
            kinetic, kinetic_sha256 = Hash(kinetic), tool_sha256 = Hash(SourcePath()), corpus, corpus_sha256 = Hash(corpus),
            context = ctx, chunks, scored_tokens = chunks * (ctx / 2 - 1), threads, baseline_ppl = basePpl,
            baseline_ms = baseline.Milliseconds, scope = "Only the selected six matrices are replaced; GGUF copies store reconstructed weights in FP16. These files do not implement compressed inference.",
            comparisons
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"{name,-34} PPL={ppl:F6} delta={row.PplChangePercent:+0.000;-0.000;0}% KLD={row.Kld:F6} top={row.SameTopPercent:F3}%");
    }
    return 0;
}

static double Number(string text, string pattern)
{
    var match = Regex.Match(text, pattern);
    if (!match.Success) throw new InvalidDataException($"Missing metric matching {pattern}");
    return double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture);
}

static (string Stdout, double Milliseconds) RunProcess(string exe, string[] arguments, string log)
{
    var info = new ProcessStartInfo(exe) { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true, CreateNoWindow = true };
    foreach (var argument in arguments) info.ArgumentList.Add(argument);
    var sw = Stopwatch.StartNew();
    using var p = Process.Start(info)!;
    var stdout = p.StandardOutput.ReadToEndAsync(); var stderr = p.StandardError.ReadToEndAsync();
    if (!p.WaitForExit(20 * 60 * 1000)) { p.Kill(true); throw new TimeoutException("Owned evaluation process exceeded 20 minutes."); }
    var result = stdout.GetAwaiter().GetResult(); var errors = stderr.GetAwaiter().GetResult(); sw.Stop();
    File.WriteAllText(log, JsonSerializer.Serialize(new { exe, arguments }) + "\n" + errors + "\n" + result);
    if (p.ExitCode != 0) throw new Exception($"Evaluation failed ({p.ExitCode}): {errors}");
    return (result, sw.Elapsed.TotalMilliseconds);
}

record Training(float[] Centers, int Batch, float Alpha0, int Splits, double Milliseconds);
record ActivationRow(string Tensor, string Algorithm, int Epochs, int Seed, string Recipe, double ChannelExponent, bool RowNormalization,
    int TrainVectors, int CalibrationRows, int ValidationRows, double TrainMs, string TrainSamplesMs, double EncodeMs, long PayloadBytes,
    double WeightNmse, double OutputNmse, string Payload);
record QualityRow(string Model, double Ppl, double PplChangePercent, double Kld, double SameTopPercent, double Milliseconds);
record ResultRow(string Tensor, int Rows, int Cols, int Dimension, int Seed, string Algorithm, int Epochs,
    int TrainVectors, int HoldoutVectors, int BatchSize, float Alpha0, int Revivals, double TrainMs, string TrainSamplesMs,
    double EncodeMs, double PackWriteMs, long PayloadBytes, long OriginalF16Bytes, double BitsPerWeight,
    double CompressionRatio, double WeightNmse, double WeightNrmse, double HoldoutNmse,
    double HoldoutFloatCodebookNmse, double GaussianProbeNrmse, double GaussianProbeCosine,
    int UsedCentroids, string Payload);
record Tensor(string Name, ulong[] Dims, uint Type, long Offset);

sealed class Gguf(string path, List<Tensor> tensors)
{
    public List<Tensor> Tensors => tensors;
    public static Gguf Open(string path)
    {
        using var r = new BinaryReader(File.OpenRead(path));
        if (r.ReadUInt32() != 0x46554747 || r.ReadUInt32() != 3) throw new InvalidDataException("Expected GGUF v3.");
        var nt = r.ReadUInt64(); var nk = r.ReadUInt64();
        if (nt > 100000 || nk > 1000000) throw new InvalidDataException("GGUF header bounds exceeded.");
        var alignment = 32u;
        for (ulong i = 0; i < nk; i++)
        {
            var key = ReadString(r); var type = r.ReadUInt32();
            if (key == "general.alignment" && type == 4) alignment = r.ReadUInt32();
            else Skip(r, type);
        }
        if (alignment == 0 || alignment > 1048576) throw new InvalidDataException("Invalid GGUF alignment.");
        var tensors = new List<Tensor>();
        for (ulong i = 0; i < nt; i++)
        {
            var name = ReadString(r); var nd = r.ReadUInt32();
            if (nd > 4) throw new InvalidDataException("Invalid tensor rank.");
            var dims = new ulong[nd]; for (var j = 0; j < nd; j++) dims[j] = r.ReadUInt64();
            tensors.Add(new(name, dims, r.ReadUInt32(), checked((long)r.ReadUInt64())));
        }
        var start = checked((r.BaseStream.Position + alignment - 1) / alignment * alignment);
        return new(path, tensors.Select(t => t with { Offset = checked(start + t.Offset) }).ToList());
    }
    public float[] ReadF16(Tensor t)
    {
        var n = checked((int)t.Dims.Aggregate(1UL, (a, b) => checked(a * b)));
        using var s = File.OpenRead(path); s.Position = t.Offset;
        var data = new byte[checked(n * 2)]; s.ReadExactly(data);
        var result = new float[n];
        for (var i = 0; i < n; i++) result[i] = (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(data, i * 2));
        if (result.Any(v => !float.IsFinite(v))) throw new InvalidDataException($"Nonfinite weights: {t.Name}");
        return result;
    }
    public void PatchF16(string destination, Tensor t, float[] data)
    {
        if (Path.GetFullPath(destination).Equals(Path.GetFullPath(path), StringComparison.OrdinalIgnoreCase)) throw new IOException("Refusing to patch source.");
        if (data.LongLength != checked((long)t.Dims.Aggregate(1UL, (a, b) => checked(a * b)))) throw new InvalidDataException("Patch shape mismatch.");
        using var stream = new FileStream(destination, FileMode.Open, FileAccess.ReadWrite);
        if (t.Offset < 0 || t.Offset + data.LongLength * 2 > stream.Length) throw new InvalidDataException("Patch exceeds destination.");
        stream.Position = t.Offset;
        using var writer = new BinaryWriter(stream);
        foreach (var value in data) writer.Write(BitConverter.HalfToUInt16Bits((Half)value));
    }
    static string ReadString(BinaryReader r)
    {
        var n = r.ReadUInt64(); if (n > 100000000) throw new InvalidDataException("GGUF string too large.");
        var data = r.ReadBytes((int)n); if (data.Length != (int)n) throw new EndOfStreamException();
        return Encoding.UTF8.GetString(data);
    }
    static void Skip(BinaryReader r, uint type)
    {
        if (type == 8) { ReadString(r); return; }
        if (type == 9)
        {
            var element = r.ReadUInt32(); var n = r.ReadUInt64();
            if (element == 9 || n > 100000000) throw new InvalidDataException("Invalid metadata array.");
            for (ulong i = 0; i < n; i++) Skip(r, element);
            return;
        }
        var size = type switch { 0 or 1 or 7 => 1, 2 or 3 => 2, 4 or 5 or 6 => 4, 10 or 11 or 12 => 8, _ => throw new InvalidDataException("Unknown metadata type.") };
        if (r.BaseStream.Position + size > r.BaseStream.Length) throw new EndOfStreamException();
        r.BaseStream.Position += size;
    }
}
