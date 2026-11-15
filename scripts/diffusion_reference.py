#!/usr/bin/env python3
"""CPU FP32 conformance for the experimental LFM2 masked-diffusion path.

Uses the pinned author's hidden()/greedy picker and a real, scaled-down
Transformers Lfm2MoeForCausalLM. No full 24B weights or remote code are loaded.
The protocol defines equal-confidence commits by ascending position; the
author's torch.topk does not specify a portable ordering for exact ties.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SOURCE_SHA = "bf3e7124106a6dbd92900f908ba1d815cb8084ad73a55bef394b1232e410a8dd"
REVISION = "f0f185c3b8b4da394527e9efbe9a0c87c94b5c01"


def digest(path):
    with Path(path).open("rb") as source:
        h = hashlib.sha256()
        for data in iter(lambda: source.read(1 << 20), b""):
            h.update(data)
        return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def export_gguf(model, destination, gguf):
    c = model.config
    writer = gguf.GGUFWriter(str(destination), "lfm2moe")
    writer.add_name("LFM2 diffusion FP32 conformance fixture (not trained)")
    writer.add_string("kinetic.diffusion.mode", "lfm2-prefix-lm-v1")
    writer.add_context_length(c.max_position_embeddings)
    writer.add_embedding_length(c.hidden_size)
    writer.add_block_count(c.num_hidden_layers)
    writer.add_feed_forward_length(c.intermediate_size)
    writer.add_expert_feed_forward_length(c.moe_intermediate_size)
    writer.add_expert_count(c.num_experts)
    writer.add_expert_used_count(c.num_experts_per_tok)
    writer.add_uint32("lfm2moe.expert_gating_func", 2)
    writer.add_head_count(c.num_attention_heads)
    writer.add_head_count_kv([c.num_key_value_heads if t == "full_attention" else 0 for t in c.layer_types])
    writer.add_layer_norm_rms_eps(c.norm_eps)
    writer.add_rope_freq_base(c.rope_parameters["rope_theta"])
    writer.add_shortconv_l_cache(c.conv_L_cache)
    # Use the standard byte-to-Unicode map, independent of list ordering.
    mapping = {}; extra = 256
    for b in range(256):
        if 33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255:
            mapping[b] = chr(b)
        else:
            mapping[b] = chr(extra); extra += 1
    tokens = [mapping[b] for b in range(256)] + ["<|startoftext|>", "<|im_end|>", "<|pad|>"]
    tokens += [f"fixture-{i}" for i in range(c.vocab_size - len(tokens))]
    tokens[259] = "aa"
    writer.add_tokenizer_model("gpt2"); writer.add_tokenizer_pre("lfm2")
    # GGUFWriter omits empty arrays; provide one valid merge for the toy vocabulary.
    writer.add_token_list(tokens); writer.add_token_merges(["a a"])
    writer.add_token_types([1] * 256 + [3] * 3 + [1] * (c.vocab_size - 259))
    writer.add_bos_token_id(c.bos_token_id); writer.add_eos_token_id(c.eos_token_id)
    writer.add_pad_token_id(c.pad_token_id); writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    tensors = []
    def add(name, tensor):
        tensors.append((name, tensor.detach().float().cpu().contiguous().numpy()))
    add("token_embd.weight", model.model.embed_tokens.weight)
    add("token_embd_norm.weight", model.model.embedding_norm.weight)
    for i, layer in enumerate(model.model.layers):
        p = f"blk.{i}."
        add(p + "attn_norm.weight", layer.operator_norm.weight)
        add(p + "ffn_norm.weight", layer.ffn_norm.weight)
        if layer.is_attention_layer:
            a = layer.self_attn
            for target, source in [("attn_q", a.q_proj), ("attn_k", a.k_proj), ("attn_v", a.v_proj), ("attn_output", a.out_proj)]:
                add(p + target + ".weight", source.weight)
            add(p + "attn_q_norm.weight", a.q_layernorm.weight)
            add(p + "attn_k_norm.weight", a.k_layernorm.weight)
        else:
            add(p + "shortconv.in_proj.weight", layer.conv.in_proj.weight)
            add(p + "shortconv.out_proj.weight", layer.conv.out_proj.weight)
            add(p + "shortconv.conv.weight", layer.conv.conv.weight.squeeze(1))
        f = layer.feed_forward
        if i < c.num_dense_layers:
            for target, source in [("ffn_gate", f.w1), ("ffn_up", f.w3), ("ffn_down", f.w2)]:
                add(p + target + ".weight", source.weight)
        else:
            add(p + "ffn_gate_inp.weight", f.gate.weight)
            add(p + "exp_probs_b.bias", f.expert_bias)
            gate, up = f.experts.gate_up_proj.chunk(2, dim=1)
            add(p + "ffn_gate_exps.weight", gate); add(p + "ffn_up_exps.weight", up)
            add(p + "ffn_down_exps.weight", f.experts.down_proj)
    for name, data in tensors:
        writer.add_tensor(name, data)
    writer.write_header_to_file(); writer.write_kv_data_to_file()
    writer.write_tensors_to_file(); writer.close()


def reference(model, prompt, canvas_size, steps, maximum, window, torch, np):
    generated = []; traces = []; block = 0; done = False
    while len(generated) < maximum and not done:
        count = min(canvas_size, maximum - len(generated))
        prefix = prompt + generated[-window:]
        canvas = [model.pad_id] * count; masked = [True] * count
        for step in range(steps):
            ids = torch.tensor([prefix + canvas], dtype=torch.long)
            mask = torch.tensor([[False] * len(prefix) + masked])
            with torch.inference_mode():
                hidden = model.hidden(ids, mask, len(prefix))[:, len(prefix):]
                raw = model.model.lm_head(hidden[0]).float()
                logits = raw.clone(); logits[:, model.pad_id] = -torch.inf
                candidates, confidence = model._pick(logits, 0.0, 0, 1.0)
            chosen_tokens = candidates.tolist()
            scores = confidence.tolist()
            available = [i for i, active in enumerate(masked) if active]
            if step == steps - 1:
                chosen = available
            else:
                remaining = count * (steps - step - 1) // steps
                chosen = sorted(available, key=lambda i: (-scores[i], i))[:len(available) - remaining]
            before = canvas.copy(); mask_before = masked.copy()
            for i in chosen:
                canvas[i] = chosen_tokens[i]; masked[i] = False
            traces.append(dict(block=block, step=step, prefix=prefix.copy(), canvas_before=before,
                masked_before=mask_before, logits=raw.detach().numpy().copy(), candidates=chosen_tokens,
                log_confidence=[s if active else None for s, active in zip(scores, mask_before)],
                committed=chosen, canvas_after=canvas.copy()))
            if any(t in model.eos_ids and not masked[i] and not any(masked[:i]) for i, t in enumerate(canvas)):
                break
        stop = next((i for i, t in enumerate(canvas) if t in model.eos_ids), None)
        if stop is not None:
            canvas = canvas[:stop]; done = True
        generated += canvas; block += 1
    return generated, traces


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kinetic", type=Path, required=True)
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--gguf-python", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cases", default="1,16,32,64")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    source = a.source_dir / "modeling_diffusion_lfm.py"
    if digest(source) != SOURCE_SHA:
        raise RuntimeError("reference source hash differs from the pinned revision")
    if a.output.exists() and any(a.output.iterdir()):
        raise RuntimeError("comparison output must be new or empty")
    if a.dry_run:
        print(json.dumps({"source_revision": REVISION, "model": str(a.model_dir), "output": str(a.output), "cases": a.cases}))
        return
    import numpy as np
    import torch
    from safetensors.torch import save_file
    import transformers
    from transformers import Lfm2MoeConfig, Lfm2MoeForCausalLM
    sys.path.insert(0, str(a.gguf_python))
    import gguf
    torch.set_num_threads(1); torch.manual_seed(20261112)
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location("pinned_lfm_diffusion_reference", source)
    upstream = importlib.util.module_from_spec(spec); spec.loader.exec_module(upstream)
    config_source = json.loads((a.source_dir / "config.json").read_text())
    config = Lfm2MoeConfig(vocab_size=384, hidden_size=32, intermediate_size=184,
        moe_intermediate_size=24, num_hidden_layers=40, num_dense_layers=2,
        num_experts=64, num_experts_per_tok=4, num_attention_heads=4, num_key_value_heads=2,
        layer_types=config_source["layer_types"], norm_eps=1e-5, conv_L_cache=3,
        bos_token_id=256, eos_token_id=257, pad_token_id=258,
        use_cache=False, use_expert_bias=True, norm_topk_prob=True,
        initializer_range=0.16, rope_parameters={"rope_type": "default", "rope_theta": 10000000.0},
        max_position_embeddings=4096, tie_word_embeddings=True)
    config._attn_implementation = "eager"
    model = Lfm2MoeForCausalLM(config).float().cpu().eval()
    for layer in model.model.layers:
        if hasattr(layer.feed_forward, "expert_bias"):
            with torch.no_grad(): layer.feed_forward.expert_bias.uniform_(-0.003, 0.003)
    mask_embedding = torch.randn(32) * 0.16
    if a.model_dir.exists() and any(a.model_dir.iterdir()):
        raise RuntimeError("fixture model directory must be new or empty")
    a.model_dir.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.model_dir / "hf")
    save_file({"mask_embedding": mask_embedding}, a.model_dir / "mask.safetensors")
    checkpoint = a.model_dir / "lfm2moe-diffusion-fixture-f32.gguf"
    export_gguf(model, checkpoint, gguf)
    oracle = upstream.DiffusionLFM(model, mask_embedding).eval()
    oracle.use_prefix_cache = False
    manifest = dict(reference_repo="summerAIO/diffusion-lfm2.5-24b-a2b", reference_revision=REVISION,
        reference_sha256=SOURCE_SHA, kinetic_sha256=digest(a.kinetic), model_sha256=digest(checkpoint),
        mask_sha256=digest(a.model_dir / "mask.safetensors"), seed=20261112,
        dtype="float32", torch=torch.__version__, transformers=transformers.__version__,
        python=sys.version, numpy=np.__version__, harness_sha256=digest(__file__),
        config_source_sha256=digest(a.source_dir / "config.json"),
        parameters=model.num_parameters(), model_bytes=checkpoint.stat().st_size,
        torch_threads=1, kinetic_threads=6, prefix_reuse=False,
        scale=dict(layers=40, conv=30, attention=10, experts=64, top_k=4, width=32, vocab=384),
        full_24b_weights=False, atol=5e-5, rtol=2e-4)
    save(a.output / "run.json", manifest)
    cases = [(1, 4, 3, 2, 9), (16, 1, 16, 8, 9), (16, 4, 33, 8, 37),
             (32, 3, 39, 8, 9), (64, 4, 71, 8, 37)]
    selected = set(map(int, a.cases.split(",")))
    results = []
    for index, (canvas, steps, maximum, window, length) in enumerate(cases):
        if canvas not in selected: continue
        name = f"case-{index}-c{canvas}-s{steps}"
        folder = a.output / name; folder.mkdir()
        prompt = [256] + [10 + (i * 37) % 240 for i in range(length - 1)]
        prompt_file = folder / "prompt.json"; save(prompt_file, prompt)
        expected, ref_steps = reference(oracle, prompt, canvas, steps, maximum, window, torch, np)
        # Cross-check our traced driver against the author's unchanged sampler.
        with torch.inference_mode():
            original, stats = oracle.generate(torch.tensor([prompt]), max_new_tokens=maximum,
                block_size=canvas, steps=steps, prefix_window=window)
        if original != expected or stats["forward_calls"] != len(ref_steps):
            raise AssertionError("traced fixed sampler differs from the author's original generator (possibly an exact confidence tie)")
        trace_dir = folder / "rust"
        cmd = [str(a.kinetic), "diffuse", str(checkpoint), "--mask-embedding", str(a.model_dir / "mask.safetensors"),
            "--prompt-tokens", str(prompt_file), "--canvas", str(canvas), "--steps", str(steps),
            "--max-new-tokens", str(maximum), "--prefix-window", str(window), "--trace-dir", str(trace_dir), "-t", "6"]
        save(folder / "command.json", cmd)
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=600)
        (folder / "stdout.json").write_text(proc.stdout, encoding="utf-8")
        (folder / "stderr.log").write_text(proc.stderr, encoding="utf-8")
        if proc.returncode: raise RuntimeError(f"Kinetic failed: {proc.stderr}")
        actual = json.loads(proc.stdout)
        rust = json.loads((trace_dir / "trace.json").read_text())
        if len(rust["steps"]) != len(ref_steps): raise AssertionError("forward count differs")
        largest = 0.0; compared = 0
        for number, (r, ref) in enumerate(zip(rust["steps"], ref_steps)):
            raw = np.fromfile(trace_dir / r["logits_file"], dtype="<f4").reshape(r["logits_shape"])
            ref_raw = ref.pop("logits")
            ref_raw.astype("<f4").tofile(folder / f"python-{number:02}.f32")
            largest = max(largest, float(np.max(np.abs(raw - ref_raw))))
            compared += raw.size
            np.testing.assert_allclose(raw, ref_raw, atol=manifest["atol"], rtol=manifest["rtol"])
            for key in ["block", "step", "prefix", "canvas_before", "masked_before", "candidates", "canvas_after"]:
                if r[key] != ref[key]: raise AssertionError(f"step {number}: {key} differs")
            if sorted(r["committed"]) != sorted(ref["committed"]): raise AssertionError(f"step {number}: commit set differs")
            for x, y in zip(r["log_confidence"], ref["log_confidence"]):
                if (x is None) != (y is None): raise AssertionError("confidence mask differs")
                if x is not None: np.testing.assert_allclose(x, y, atol=manifest["atol"], rtol=manifest["rtol"])
            save(folder / f"python-{number:02}.json", ref)
        if actual["tokens"] != expected: raise AssertionError("final token IDs differ")
        result = dict(case=name, status="passed", passes=len(ref_steps), logits_compared=compared,
            max_abs_logit_error=largest, output_tokens=len(expected))
        results.append(result); print(json.dumps(result), flush=True)
        save(a.output / "results.json", results)
    if not results: raise RuntimeError("no comparison cases selected")
    print(f"PASS: {len(results)} cases, {sum(r['logits_compared'] for r in results)} logits")


if __name__ == "__main__":
    main()
