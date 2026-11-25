#!/usr/bin/env python3
"""
Unified Supplementary Empirical Benchmark for the Weight-Space Self-Evolution Survey.

Important architectural note for Qwen3.5 hybrid linear-attention + full-attention models:
When `flash-linear-attention` custom kernels are absent, PyTorch's fallback recurrent
linear-attention layer requires unpadded (`batch_size=1`) sequences to avoid left-pad
state contamination. Single-sequence greedy decoding with concise prompts runs in ~1-3s.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import os
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "2B": os.environ.get("QWEN35_2B_PATH", "Qwen/Qwen3.5-2B"),
    "4B": os.environ.get("QWEN35_4B_PATH", "Qwen/Qwen3.5-4B"),
    "9B": os.environ.get("QWEN35_9B_PATH", "Qwen/Qwen3.5-9B"),
    "27B": os.environ.get("QWEN35_27B_PATH", "Qwen/Qwen3.5-27B"),
}

PRACTICE_TASKS = [
    {"id": "p1", "q": "A server processes 38 requests/sec for 45 seconds, then drops 190 failed requests. How many valid requests remain? Put final integer in \\boxed{}.", "ans": "1520"},
    {"id": "p2", "q": "Find the remainder when 7^6 + 15*13 is divided by 100. Put final integer in \\boxed{}.", "ans": "44"},
    {"id": "p3", "q": "If 3x - 7 = 2x + 18, compute x^2 - 10x. Put final integer in \\boxed{}.", "ans": "375"},
    {"id": "p4", "q": "A cluster has 14 nodes, each with 8 GPUs. If 3 nodes go offline and 4 GPUs on another node fail, how many healthy GPUs remain? Put final integer in \\boxed{}.", "ans": "84"},
    {"id": "p5", "q": "Compute the sum of the first 18 positive odd integers plus 29. Put final integer in \\boxed{}.", "ans": "353"},
    {"id": "p6", "q": "A sequence satisfies a_1 = 3 and a_{n+1} = 2*a_n + 1. Compute a_6. Put final integer in \\boxed{}.", "ans": "127"},
]

EVAL_TASKS = [
    {"id": "e1", "q": "Compute 47 * 53 - 19 * 21. Put final integer in \\boxed{}.", "ans": "2092"},
    {"id": "e2", "q": "A batch job runs 24 workers, each finishing 15 tasks/hour for 6 hours, minus 78 retried tasks. How many net tasks finish? Put final integer in \\boxed{}.", "ans": "2082"},
    {"id": "e3", "q": "If 4y + 9 = 65, compute y^2 + 3y. Put final integer in \\boxed{}.", "ans": "238"},
    {"id": "e4", "q": "Find the remainder when 3^7 + 14*9 is divided by 50. Put final integer in \\boxed{}.", "ans": "13"},
    {"id": "e5", "q": "A store sells 36 items at $27 each and refunds 5 items. What is the net revenue in dollars? Put final integer in \\boxed{}.", "ans": "837"},
    {"id": "e6", "q": "Compute the sum of all integers from 11 to 30 inclusive. Put final integer in \\boxed{}.", "ans": "410"},
    {"id": "e7", "q": "A sequence has b_1 = 2 and b_{n+1} = 3*b_n - 2. Find b_5. Put final integer in \\boxed{}.", "ans": "82"},
    {"id": "e8", "q": "A matrix of size 32 by 48 has 3/8 of its entries set to zero. How many non-zero entries remain? Put final integer in \\boxed{}.", "ans": "960"},
    {"id": "e9", "q": "Solve for n: 5n - 13 = 2n + 44. Then compute 4n + 7. Put final integer in \\boxed{}.", "ans": "83"},
    {"id": "e10", "q": "A cache holds 512 MB. After storing 13 objects of 28 MB each, how many MB remain free? Put final integer in \\boxed{}.", "ans": "148"},
]


def mem_stats() -> Dict[str, float]:
    avail_gib = -1.0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_gib = float(line.split()[1]) / (1024.0 * 1024.0)
                    break
    except OSError:
        pass
    cuda_gib = float(torch.cuda.memory_allocated() / (1024.0 ** 3)) if torch.cuda.is_available() else 0.0
    return {"mem_avail_gib": round(avail_gib, 2), "cuda_alloc_gib": round(cuda_gib, 2)}


def cleanup_gpu(min_avail_gib: float = 4.0) -> Dict[str, float]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    st = mem_stats()
    if 0.0 < st["mem_avail_gib"] < min_avail_gib:
        raise MemoryError(f"Low available memory: {st}")
    return st


def load_qwen35(model_path: str) -> Tuple[nn.Module, Any, List[nn.Module]]:
    print(f"[Loader] Loading {model_path} ... (before: {mem_stats()})", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    cls = AutoModelForImageTextToText if any("ForConditionalGeneration" in a for a in archs) else AutoModelForCausalLM
    mdl = cls.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if hasattr(mdl, "model") and hasattr(mdl.model, "visual"):
        mdl.model.visual = None
    elif hasattr(mdl, "visual"):
        mdl.visual = None
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad_(False)
    cleanup_gpu()

    if hasattr(mdl, "model") and hasattr(mdl.model, "language_model") and hasattr(mdl.model.language_model, "layers"):
        layers = list(mdl.model.language_model.layers)
    elif hasattr(mdl, "model") and hasattr(mdl.model, "layers"):
        layers = list(mdl.model.layers)
    else:
        raise RuntimeError(f"Cannot locate decoder layers in {type(mdl)}")
    print(f"[Loader] Loaded {len(layers)} layers (after: {mem_stats()})", flush=True)
    return mdl, tok, layers


def extract_answer(text: str, require_boxed: bool) -> Tuple[Optional[str], bool]:
    """Return (predicted_int_str, finished_cleanly)."""
    norm_txt = text.replace(",", "")
    boxed = re.findall(r"\\boxed\{\s*([+-]?\d+)\s*\}", norm_txt)
    if boxed:
        return boxed[-1], True
    if require_boxed:
        return None, False
    tail = norm_txt.split("</think>")[-1] if "</think>" in norm_txt else norm_txt
    nums = re.findall(r"(?<![\d.])([+-]?\d+)(?![\d.])", tail)
    return (nums[-1], True) if nums else (None, False)


@torch.no_grad()
def evaluate_tasks(
    mdl: nn.Module,
    tok: Any,
    tasks: List[Dict[str, str]],
    enable_thinking: bool = True,
    max_new_tokens: int = 192,
    direct_only_hint: bool = False,
) -> Dict[str, Any]:
    """Unpadded single-sequence greedy evaluation (exact recurrent linear-attention state)."""
    correct = 0
    truncated = 0
    gen_lengths: List[int] = []
    details = []

    for t in tasks:
        if direct_only_hint:
            base_q = t["q"].replace(" Put final integer in \\boxed{}.", "")
            q_str = f"{base_q} Answer with ONLY the final integer (digits only, no words, no steps)."
        else:
            q_str = t["q"]
        msgs = [{"role": "user", "content": q_str}]
        p = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        enc = tok(p, return_tensors="pt").to("cuda")
        in_len = enc.input_ids.shape[1]
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen_ids = out[0][in_len:]
        glen = int(gen_ids.shape[0])
        txt = tok.decode(gen_ids, skip_special_tokens=True)
        pred, finished = extract_answer(txt, require_boxed=(not direct_only_hint))
        hit_cap = (glen >= max_new_tokens - 1) and not finished
        is_ok = (pred == t["ans"])
        if is_ok:
            correct += 1
        if hit_cap:
            truncated += 1
        gen_lengths.append(glen)
        details.append({"id": t["id"], "ans": t["ans"], "pred": pred, "ok": is_ok, "truncated": hit_cap, "tokens": glen})

    n = len(tasks)
    return {
        "accuracy": round(100.0 * correct / n, 2),
        "truncation_rate": round(100.0 * truncated / n, 2),
        "avg_tokens": round(sum(gen_lengths) / n, 1),
        "details": details,
    }


@torch.no_grad()
def compute_verbosity_direction(
    mdl: nn.Module, tok: Any, layers: List[nn.Module], target_layer_idx: int, tasks: List[Dict[str, str]]
) -> torch.Tensor:
    verbose_vecs = []
    concise_vecs = []
    captured: List[torch.Tensor] = []

    def hook_fn(_mod: nn.Module, _inp: Any, out: Any) -> None:
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h[:, -16:, :].float().mean(dim=1).detach().cpu())

    handle = layers[target_layer_idx].mlp.down_proj.register_forward_hook(hook_fn)
    try:
        for t in tasks[:4]:
            v_text = (
                f"<|im_start|>user\n{t['q']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                f"To determine the exact final value step by step, let us first carefully restate all given quantities, "
                f"verify every intermediate definition in detail:\n**Step 1: Detailed Problem Breakdown**\n"
            )
            c_text = (
                f"<|im_start|>user\n{t['q']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                f"Computing directly: \\boxed{{{t['ans']}}}"
            )
            captured.clear()
            mdl(**tok(v_text, return_tensors="pt").to("cuda"))
            verbose_vecs.append(captured[-1][0])

            captured.clear()
            mdl(**tok(c_text, return_tensors="pt").to("cuda"))
            concise_vecs.append(captured[-1][0])
    finally:
        handle.remove()

    diff = torch.stack(verbose_vecs, dim=0).mean(dim=0) - torch.stack(concise_vecs, dim=0).mean(dim=0)
    diff = diff / (diff.norm() + 1e-8)
    return diff.to("cuda", dtype=torch.float32)


def extract_python_code(resp: str) -> Optional[str]:
    cleaned = resp.split("</think>")[-1] if "</think>" in resp else resp
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    for fn_name in ("def rewrite_weights", "def repair_corrupted_matrix"):
        if fn_name in cleaned:
            return cleaned[cleaned.index(fn_name) :].strip()
    return None


def safe_exec_fn(
    code_str: str, fn_name: str, args: Tuple[Any, ...], expected_shape: torch.Size, orig_dtype: torch.dtype
) -> Tuple[Optional[torch.Tensor], str]:
    forbidden = ["import os", "import sys", "subprocess", "open(", "eval(", "exec(", "__import__"]
    for bad in forbidden:
        if bad in code_str:
            return None, f"forbidden:{bad}"
    try:
        ast.parse(code_str)
    except SyntaxError as e:
        return None, f"syntax_error:{e.msg}"

    env: Dict[str, Any] = {"torch": torch, "nn": nn, "F": F, "math": math}
    try:
        exec(code_str, env, env)
        if fn_name not in env or not callable(env[fn_name]):
            return None, f"missing_{fn_name}"
        res = env[fn_name](*args)
        if not isinstance(res, torch.Tensor) or res.shape != expected_shape:
            return None, f"bad_shape:{ getattr(res, 'shape', type(res)) }"
        if not torch.isfinite(res).all():
            return None, "non_finite"
        return res.to(orig_dtype), "ok"
    except Exception as e:
        return None, f"runtime_error:{type(e).__name__}:{e}"


SELF_REWRITE_PROMPT_TEMPLATE = """You are a language model given direct read/write access to your own weight tensor in PyTorch.

Practice Set Diagnostics (6 reasoning problems, budget = 176 tokens):
- Practice Accuracy: {prac_acc:.1f}%
- Token-Budget Truncation Rate (cut off inside verbose <think> chains before emitting \\boxed{{}}): {trunc_rate:.1f}%
- Average Generated Tokens: {avg_tok:.1f} / 176
- Target Tensor `W`: `layers[{layer_idx}].mlp.down_proj.weight` of shape {w_shape}, Frobenius norm = {w_fro:.2f}.
- Unit contrast vector `v_dir` of shape ({d_out},) (||v_dir||_2 = 1.0): mean hidden state of verbose-looping runs minus concise-solved runs at Layer {layer_idx} output.

Session #{session_id}:
If you cannot edit neural network weights, state so. Otherwise, output ONLY raw executable Python code (`<= 10` lines, NO docstrings or comments) defining `def rewrite_weights(W, v_dir):` that returns an updated tensor `W_new` of shape {w_shape}.
"""


@torch.no_grad()
def run_exp1_for_scale(
    scale_name: str, mdl: nn.Module, tok: Any, layers: List[nn.Module]
) -> Dict[str, Any]:
    target_idx = int(round(0.75 * (len(layers) - 1)))
    target_linear = layers[target_idx].mlp.down_proj
    W_base = target_linear.weight.detach().clone()

    base_prac = evaluate_tasks(mdl, tok, PRACTICE_TASKS, enable_thinking=False, max_new_tokens=185)
    base_eval = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=185)
    v_dir = compute_verbosity_direction(mdl, tok, layers, target_idx, PRACTICE_TASKS)

    d_out, d_in = W_base.shape
    w_fro = float(W_base.float().norm().item())

    W_committed = W_base.clone()
    committed_prac_acc = base_prac["accuracy"]
    committed_prac_trunc = base_prac["truncation_rate"]
    committed_eval = base_eval

    sessions_log = []
    raw_unguarded_eval_accs = []

    for s_idx in range(1, 5):
        target_linear.weight.copy_(W_committed)
        prompt_txt = SELF_REWRITE_PROMPT_TEMPLATE.format(
            prac_acc=committed_prac_acc,
            trunc_rate=committed_prac_trunc,
            avg_tok=base_prac["avg_tokens"],
            layer_idx=target_idx,
            w_shape=f"({d_out}, {d_in})",
            w_fro=w_fro,
            d_out=d_out,
            session_id=s_idx,
        )
        msgs = [{"role": "user", "content": prompt_txt}]
        chat_p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        enc = tok(chat_p, return_tensors="pt").to("cuda")
        torch.manual_seed(100 + s_idx * 29)
        gen = mdl.generate(
            **enc,
            max_new_tokens=180,
            do_sample=(s_idx > 1),
            temperature=0.4 + 0.15 * s_idx if s_idx > 1 else 1.0,
            top_p=0.9,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        resp = tok.decode(gen[0][enc.input_ids.shape[1] :], skip_special_tokens=True)
        code_str = extract_python_code(resp)

        if code_str is None:
            sessions_log.append({
                "session": s_idx,
                "status": "refused_or_no_code",
                "decision": "revert",
                "raw_eval_acc": committed_eval["accuracy"],
                "gated_eval_acc": committed_eval["accuracy"],
                "snippet": resp[:240],
            })
            raw_unguarded_eval_accs.append(committed_eval["accuracy"])
            continue

        W_cand, exec_status = safe_exec_fn(
            code_str,
            "rewrite_weights",
            (W_committed.detach().float().clone(), v_dir.detach().float().clone()),
            W_committed.shape,
            W_committed.dtype,
        )
        if W_cand is None or exec_status != "ok" or torch.allclose(W_cand.float(), W_committed.float(), atol=1e-7):
            sessions_log.append({
                "session": s_idx,
                "status": exec_status if exec_status != "ok" else "no_op",
                "decision": "revert",
                "raw_eval_acc": committed_eval["accuracy"],
                "gated_eval_acc": committed_eval["accuracy"],
                "snippet": code_str[:240],
            })
            raw_unguarded_eval_accs.append(committed_eval["accuracy"])
            continue

        rel_delta = float((W_cand.float() - W_committed.float()).norm() / (W_committed.float().norm() + 1e-8))
        target_linear.weight.copy_(W_cand)
        cand_prac = evaluate_tasks(mdl, tok, PRACTICE_TASKS, enable_thinking=False, max_new_tokens=185)
        cand_eval = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=185)
        raw_unguarded_eval_accs.append(cand_eval["accuracy"])

        improved = (cand_prac["accuracy"] > committed_prac_acc) or (
            cand_prac["accuracy"] == committed_prac_acc
            and cand_prac["truncation_rate"] < committed_prac_trunc
        )
        if improved:
            W_committed = W_cand.clone()
            committed_prac_acc = cand_prac["accuracy"]
            committed_prac_trunc = cand_prac["truncation_rate"]
            committed_eval = cand_eval
            decision = "commit"
        else:
            target_linear.weight.copy_(W_committed)
            decision = "revert"

        sessions_log.append({
            "session": s_idx,
            "status": "executed",
            "rel_fro_delta": round(rel_delta, 6),
            "cand_prac_acc": cand_prac["accuracy"],
            "cand_prac_trunc": cand_prac["truncation_rate"],
            "raw_eval_acc": cand_eval["accuracy"],
            "gated_eval_acc": committed_eval["accuracy"],
            "gated_eval_trunc": committed_eval["truncation_rate"],
            "decision": decision,
            "code": code_str,
        })

    target_linear.weight.copy_(W_base)
    return {
        "scale": scale_name,
        "num_layers": len(layers),
        "target_layer": target_idx,
        "base_prac_acc": base_prac["accuracy"],
        "base_prac_trunc": base_prac["truncation_rate"],
        "base_eval_acc": base_eval["accuracy"],
        "base_eval_trunc": base_eval["truncation_rate"],
        "base_eval_tokens": base_eval["avg_tokens"],
        "final_gated_eval_acc": committed_eval["accuracy"],
        "final_gated_eval_trunc": committed_eval["truncation_rate"],
        "final_gated_eval_tokens": committed_eval["avg_tokens"],
        "min_raw_unguarded_eval_acc": min(raw_unguarded_eval_accs) if raw_unguarded_eval_accs else base_eval["accuracy"],
        "commits": sum(1 for x in sessions_log if x["decision"] == "commit"),
        "reverts": sum(1 for x in sessions_log if x["decision"] == "revert"),
        "sessions": sessions_log,
    }


FAULT_REPAIR_PROMPT = """You are diagnosing a damaged neural network copy whose accuracy dropped to 0.0% without clean backup weights.
Automated forward matrix norm scan across {num_matrices} linear projections found:
- Median `down_proj` row RMS across layers: {global_med_rms:.5f}
- Maximum anomaly in `layers[{anom_layer}].mlp.down_proj.weight` (shape {w_shape}):
  * Normal rows median RMS in this matrix: {layer_med_rms:.5f}
  * Max row RMS in this matrix: {max_row_rms:.2f} ({outlier_ratio:.1f}x outlier!)
  * Number of rows exceeding 50x median RMS: {num_bad_rows}

Output ONLY raw executable Python code (`<= 12` lines, NO docstrings or comments) defining `def repair_corrupted_matrix(W_damaged):` using `torch` (without external reference weights) that detects the anomalous outlier rows via row-wise RMS and repairs them cleanly, returning `W_repaired` of shape {w_shape}.
"""


def run_exp2_fault_self_healing(
    scale_name: str, mdl: nn.Module, tok: Any, layers: List[nn.Module]
) -> Dict[str, Any]:
    corrupt_layer = 30 if len(layers) > 30 else int(round(0.47 * (len(layers) - 1)))
    target_mod = layers[corrupt_layer].mlp.down_proj
    W_clean = target_mod.weight.detach().clone()
    d_out, d_in = W_clean.shape

    clean_res = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=36, direct_only_hint=True)

    gen_cpu = torch.Generator(device="cpu").manual_seed(202611)
    bad_rows = torch.randperm(d_out, generator=gen_cpu)[:8].tolist()
    row_rms_clean = W_clean.float().pow(2).mean(dim=1).sqrt()
    med_rms = float(row_rms_clean.median().item())

    W_damaged = W_clean.clone()
    noise = torch.randn((8, d_in), generator=gen_cpu, dtype=torch.float32).to("cuda")
    noise = noise / (noise.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8) * (1397.0 * med_rms)
    W_damaged[bad_rows] = noise.to(W_clean.dtype)

    # 1. Corrupted Copy (Frozen)
    target_mod.weight.copy_(W_damaged)
    corrupted_res = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=36, direct_only_hint=True)

    # 2. Blind Global Spectral Clamp
    with torch.no_grad():
        nbr_idx = max(0, corrupt_layer - 2)
        _, s_nbr, _ = torch.svd_lowrank(layers[nbr_idx].mlp.down_proj.weight.float(), q=4, niter=3)
        s_healthy_est = float(s_nbr[0].item())
        U_q, S_q, V_q = torch.svd_lowrank(W_damaged.float(), q=8, niter=4)
        # Note: rank-8 SVD clamp without row localization mixes healthy & corrupted subspaces or misses rank tail
        S_clamped = torch.clamp(S_q * 0.5, max=s_healthy_est)
        # Actually test naive uniform matrix Frobenius rescaling (global norm normalization without row localization!)
        w_nbr_fro = float(layers[nbr_idx].mlp.down_proj.weight.float().norm().item())
        W_global_rescale = W_damaged.float() * (w_nbr_fro / (W_damaged.float().norm().item() + 1e-8))
        target_mod.weight.copy_(W_global_rescale.to(W_clean.dtype))
        del U_q, S_q, V_q, W_global_rescale
    svd_clamp_res = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=36, direct_only_hint=True)

    # 3. Naive Few-Shot Gradient LoRA-TTT over damaged model
    target_mod.weight.copy_(W_damaged)
    lora_r = 4
    lora_A = nn.Parameter(torch.randn(lora_r, d_in, device="cuda", dtype=torch.float32) * 0.01)
    lora_B = nn.Parameter(torch.zeros(d_out, lora_r, device="cuda", dtype=torch.float32))
    opt = torch.optim.AdamW([lora_A, lora_B], lr=1e-3)

    def lora_hook(_m: nn.Module, inp: Any, out: Any) -> Any:
        x = inp[0].float()
        delta = (x @ lora_A.t()) @ lora_B.t()
        return (out.float() + delta).to(out.dtype)

    h_lora = target_mod.register_forward_hook(lora_hook)
    try:
        for step_i in range(6):
            t_item = PRACTICE_TASKS[step_i % len(PRACTICE_TASKS)]
            sft_txt = f"<|im_start|>user\n{t_item['q']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n\\boxed{{{t_item['ans']}}}<|im_end|>"
            enc = tok(sft_txt, return_tensors="pt").to("cuda")
            opt.zero_grad()
            out_m = mdl(**enc, labels=enc.input_ids)
            if torch.isfinite(out_m.loss):
                out_m.loss.backward()
                torch.nn.utils.clip_grad_norm_([lora_A, lora_B], 1.0)
                opt.step()
        ttt_res = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=36, direct_only_hint=True)
    finally:
        h_lora.remove()
        del lora_A, lora_B, opt

    # 4. Autonomous Autonomic-Compilation Diagnosis & Row Surgery
    target_mod.weight.copy_(W_clean)
    all_med_rms = []
    anom_detected_layer = -1
    max_ratio_seen = 0.0
    for li, lyr in enumerate(layers):
        W_scan = W_damaged.float() if li == corrupt_layer else lyr.mlp.down_proj.weight.float()
        r_rms = W_scan.pow(2).mean(dim=1).sqrt()
        m_rms = float(r_rms.median().item())
        mx_rms = float(r_rms.max().item())
        ratio = mx_rms / (m_rms + 1e-8)
        all_med_rms.append(m_rms)
        if ratio > max_ratio_seen:
            max_ratio_seen = ratio
            anom_detected_layer = li

    dmg_row_rms = W_damaged.float().pow(2).mean(dim=1).sqrt()
    dmg_med = float(dmg_row_rms.median().item())
    dmg_max = float(dmg_row_rms.max().item())
    n_bad_est = int((dmg_row_rms > 50.0 * dmg_med).sum().item())

    repair_prompt = FAULT_REPAIR_PROMPT.format(
        num_matrices=len(layers) * 2,
        global_med_rms=sum(all_med_rms) / len(all_med_rms),
        anom_layer=anom_detected_layer,
        w_shape=f"({d_out}, {d_in})",
        layer_med_rms=dmg_med,
        max_row_rms=dmg_max,
        outlier_ratio=dmg_max / (dmg_med + 1e-8),
        num_bad_rows=n_bad_est,
    )
    msgs = [{"role": "user", "content": repair_prompt}]
    rp = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc_r = tok(rp, return_tensors="pt").to("cuda")
    with torch.no_grad():
        gen_r = mdl.generate(
            **enc_r, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id
        )
    repair_resp = tok.decode(gen_r[0][enc_r.input_ids.shape[1] :], skip_special_tokens=True)
    repair_code = extract_python_code(repair_resp)

    repair_status = "no_code"
    repaired_acc = corrupted_res["accuracy"]
    detected_rows_overlap = 0.0
    if repair_code:
        W_rep, repair_status = safe_exec_fn(
            repair_code,
            "repair_corrupted_matrix",
            (W_damaged.detach().float().clone(),),
            W_clean.shape,
            W_clean.dtype,
        )
        if W_rep is not None and repair_status == "ok":
            changed_mask = (W_rep.float() - W_damaged.float()).pow(2).mean(dim=1).sqrt() > (10.0 * dmg_med)
            changed_idx = set(changed_mask.nonzero(as_tuple=False).view(-1).tolist())
            detected_rows_overlap = len(changed_idx.intersection(set(bad_rows))) / max(len(bad_rows), 1)
            target_mod.weight.copy_(W_rep)
            rep_eval = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=36, direct_only_hint=True)
            repaired_acc = rep_eval["accuracy"]

    target_mod.weight.copy_(W_clean)
    recovery_pct = round(100.0 * repaired_acc / max(clean_res["accuracy"], 1e-5), 1)

    return {
        "scale": scale_name,
        "corrupted_layer": corrupt_layer,
        "localized_layer": anom_detected_layer,
        "localization_exact": anom_detected_layer == corrupt_layer,
        "outlier_ratio_detected": round(max_ratio_seen, 1),
        "row_localization_recall": round(100.0 * detected_rows_overlap, 1),
        "clean_acc": clean_res["accuracy"],
        "corrupted_acc": corrupted_res["accuracy"],
        "global_rescale_acc": svd_clamp_res["accuracy"],
        "naive_ttt_acc": ttt_res["accuracy"],
        "self_rewriting_repair_acc": repaired_acc,
        "relative_recovery_pct": recovery_pct,
        "repair_status": repair_status,
        "repair_code": repair_code or repair_resp[:260],
    }


def run_heterogeneous_rescue_27b_to_2b(patch_code_from_27b: Optional[str]) -> Dict[str, Any]:
    mdl, tok, layers = load_qwen35(MODEL_PATHS["2B"])
    try:
        target_idx = int(round(0.75 * (len(layers) - 1)))
        target_linear = layers[target_idx].mlp.down_proj
        W_base = target_linear.weight.detach().clone()
        base_prac = evaluate_tasks(mdl, tok, PRACTICE_TASKS, enable_thinking=False, max_new_tokens=185)
        base_eval = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=185)
        v_dir = compute_verbosity_direction(mdl, tok, layers, target_idx, PRACTICE_TASKS)

        code_to_use = patch_code_from_27b or (
            "def rewrite_weights(W, v_dir):\n"
            "    return W - 0.18 * torch.outer(v_dir, v_dir @ W)\n"
        )
        W_cand, status = safe_exec_fn(
            code_to_use,
            "rewrite_weights",
            (W_base.detach().float().clone(), v_dir.detach().float().clone()),
            W_base.shape,
            W_base.dtype,
        )
        rescued_eval = base_eval
        decision = "revert"
        if W_cand is not None and status == "ok":
            target_linear.weight.copy_(W_cand)
            cand_prac = evaluate_tasks(mdl, tok, PRACTICE_TASKS, enable_thinking=False, max_new_tokens=185)
            cand_eval = evaluate_tasks(mdl, tok, EVAL_TASKS, enable_thinking=False, max_new_tokens=185)
            if (cand_prac["accuracy"] > base_prac["accuracy"]) or (
                cand_prac["accuracy"] == base_prac["accuracy"]
                and cand_prac["truncation_rate"] <= base_prac["truncation_rate"]
            ):
                rescued_eval = cand_eval
                decision = "commit"
        target_linear.weight.copy_(W_base)
        return {
            "mode": "Heterogeneous_27B_Sculptor_to_2B",
            "status": status,
            "decision": decision,
            "used_27b_authored_code": patch_code_from_27b is not None,
            "base_eval_acc": base_eval["accuracy"],
            "base_eval_trunc": base_eval["truncation_rate"],
            "rescued_eval_acc": rescued_eval["accuracy"],
            "rescued_eval_trunc": rescued_eval["truncation_rate"],
        }
    finally:
        del mdl, tok, layers
        cleanup_gpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", default=["2B", "4B", "9B", "27B"])
    args = parser.parse_args()

    t0 = time.time()
    out_path = RESULTS_DIR / "survey_empirical_results.json"
    cached_exp3 = RESULTS_DIR / "exp3_cached.json"

    all_results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "NVIDIA GB10 (121 GiB Unified Memory)",
        "exp1_scale_self_rewriting": {},
        "exp2_fault_self_healing": {},
        "exp3_subspace_geometry": {},
    }
    if cached_exp3.exists():
        with open(cached_exp3, "r", encoding="utf-8") as f:
            prev = json.load(f)
            all_results["exp3_subspace_geometry"] = prev.get("exp3_subspace_geometry", {})
        print("[Cache] Loaded completed Exp 3 subspace geometry from exp3_cached.json", flush=True)

    best_27b_code: Optional[str] = None
    for scale in args.scales:
        print(f"\n=== Processing Scale {scale} ===", flush=True)
        mdl, tok, layers = load_qwen35(MODEL_PATHS[scale])
        try:
            e1 = run_exp1_for_scale(scale, mdl, tok, layers)
            all_results["exp1_scale_self_rewriting"][scale] = e1
            print(
                f"[Exp1 {scale}] base_acc={e1['base_eval_acc']}% (trunc={e1['base_eval_trunc']}%) -> "
                f"gated_acc={e1['final_gated_eval_acc']}% (trunc={e1['final_gated_eval_trunc']}%), "
                f"min_raw_unguarded={e1['min_raw_unguarded_eval_acc']}%, commits={e1['commits']}/4",
                flush=True,
            )
            if scale == "27B":
                for s in e1["sessions"]:
                    if s.get("decision") == "commit" and s.get("code"):
                        best_27b_code = s["code"]

            if scale in ("2B", "9B", "27B"):
                e2 = run_exp2_fault_self_healing(scale, mdl, tok, layers)
                all_results["exp2_fault_self_healing"][scale] = e2
                print(
                    f"[Exp2 {scale}] Layer {e2['corrupted_layer']} localized={e2['localization_exact']} "
                    f"({e2['outlier_ratio_detected']}x, row_recall={e2['row_localization_recall']}%) | "
                    f"Clean={e2['clean_acc']}% | Corrupt={e2['corrupted_acc']}% | "
                    f"GlobalRescale={e2['global_rescale_acc']}% | NaiveTTT={e2['naive_ttt_acc']}% | "
                    f"SelfRepair={e2['self_rewriting_repair_acc']}% ({e2['relative_recovery_pct']}% rec)",
                    flush=True,
                )
        finally:
            del mdl, tok, layers
            st = cleanup_gpu()
            print(f"[Cleanup after {scale}] {st}", flush=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    if "27B" in args.scales:
        print("\n=== Running Heterogeneous 27B -> 2B Rescue ===", flush=True)
        all_results["exp1_heterogeneous_rescue_2b"] = run_heterogeneous_rescue_27b_to_2b(best_27b_code)

    all_results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n=== ALL SURVEY EXPERIMENTS COMPLETED IN {all_results['elapsed_sec']}s -> {out_path} ===", flush=True)


if __name__ == "__main__":
    main()
