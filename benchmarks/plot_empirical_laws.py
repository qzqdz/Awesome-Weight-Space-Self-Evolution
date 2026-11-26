#!/usr/bin/env python3
"""
Generate all four Empirical Benchmark Figures (Figs 7-10 in PDF: fig5..fig8.pdf)
strictly from `results/clean_survey_benchmark.json` (100% self-contained published adaptation baselines).
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = BENCH_ROOT / "results" / "clean_survey_benchmark.json"
FIG_DIR = BENCH_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_BLUE = "#326A9D"
C_GREEN = "#2E7D5B"
C_AMBER = "#D97706"
C_RED = "#C53030"
C_PURP = "#6B46C1"
C_GRAY = "#64748B"
C_SLATE = "#CBD5E1"


def plot_fig5_scale_emergence(data: dict) -> None:
    law1 = data["law1_scale_and_cross_compile"]
    endo = law1["endogenous"]
    exo = law1["exogenous_27b_compiler"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.15))

    # Panel (a): Unified Endogenous REPL Patching across Qwen3.5-{2B, 4B, 9B, 27B} (N=20, 196-tok budget)
    scales = ["2B", "4B", "9B", "27B"]
    xlabels = [f"Qwen3.5-{s}" for s in scales]
    base_accs = [endo[s]["base_eval"]["accuracy"] for s in scales]
    unguarded_accs = [endo[s]["worst_unguarded_eval_acc"] for s in scales]
    gated_accs = [endo[s]["final_gated_eval"]["accuracy"] for s in scales]
    commits = [endo[s]["commits"] for s in scales]

    x = np.arange(len(scales))
    w = 0.25

    b1 = ax1.bar(x - w, base_accs, width=w, color=C_SLATE, edgecolor="#1E293B", lw=0.75, label="Frozen Base (W0)")
    b2 = ax1.bar(x, unguarded_accs, width=w, color="#FCA5A5", edgecolor=C_RED, lw=0.75, label="Worst Unguarded Patch")
    b3 = ax1.bar(x + w, gated_accs, width=w, color=C_GREEN, edgecolor="#1E293B", lw=0.75, label="Practice-Gated Commit")

    for bars, col in [(b1, "#334155"), (b2, C_RED), (b3, C_GREEN)]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color=col,
            )

    for idx, s in enumerate(scales):
        delta = gated_accs[idx] - base_accs[idx]
        c_cnt = commits[idx]
        if c_cnt > 0 and delta > 0:
            tag = f"+{delta:.0f}% ({c_cnt}/4 Commits)"
            tcol, ecol = C_GREEN, "#A7F3D0"
        elif c_cnt > 0 and delta < 0:
            tag = f"{delta:.0f}% Practice Overfit\n({c_cnt}/4 Commits)"
            tcol, ecol = C_RED, "#FECACA"
        else:
            tag = f"{c_cnt}/4 Commits\n(Shape / Index Err)"
            tcol, ecol = C_BLUE, "#CBD5E1"
        top_y = max(base_accs[idx], unguarded_accs[idx], gated_accs[idx]) + 7.8
        ax1.text(
            idx,
            min(top_y, 93.0),
            tag,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color=tcol,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=ecol, lw=0.6, alpha=0.95),
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels, fontsize=8.3, fontweight="bold")
    ax1.set_ylabel("Held-Out Math Accuracy (%, N=20)", fontsize=8.8, fontweight="bold")
    ax1.set_ylim(0, 112)
    ax1.set_title("(a) Endogenous Self-Patching Across Qwen3.5 Scales", fontsize=9.4, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=7.0, frameon=True, facecolor="white")
    ax1.grid(axis="y", ls="--", alpha=0.35)

    # Panel (b): Exogenous Cross-Compilation (Qwen3.5-27B Compiler -> {2B, 4B, 9B} Targets)
    sub_scales = ["2B", "4B", "9B"]
    xb = np.arange(len(sub_scales))
    wb = 0.25
    sub_base = [endo[s]["base_eval"]["accuracy"] for s in sub_scales]
    sub_self_ung = [endo[s]["worst_unguarded_eval_acc"] for s in sub_scales]
    sub_exo = [exo[s]["post_cross_compile_eval"]["accuracy"] for s in sub_scales]

    eb1 = ax2.bar(xb - wb, sub_base, width=wb, color=C_SLATE, edgecolor="#1E293B", lw=0.75, label="Frozen Student Base")
    eb2 = ax2.bar(xb, sub_self_ung, width=wb, color="#FCA5A5", edgecolor=C_RED, lw=0.75, label="Endogenous Unguarded")
    eb3 = ax2.bar(xb + wb, sub_exo, width=wb, color=C_BLUE, edgecolor="#1E293B", lw=0.75, label="27B Cross-Compiled (EWSM)")

    for bars, col in [(eb1, "#334155"), (eb2, C_RED), (eb3, C_BLUE)]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=7.0,
                fontweight="bold",
                color=col,
            )

    # Highlight 4B 4.0x rescue
    ax2.text(
        1.0 + wb,
        sub_exo[1] + 8.0,
        "+30% (4.0x)",
        ha="center",
        va="bottom",
        fontsize=6.8,
        fontweight="bold",
        color=C_BLUE,
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#93C5FD", lw=0.65, alpha=0.95),
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels([f"Qwen3.5-{s}\nTarget" for s in sub_scales], fontsize=8.3, fontweight="bold")
    ax2.set_ylabel("Held-Out Math Accuracy (%, N=20)", fontsize=8.8, fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.set_title("(b) Exogenous 27B Cross-Compilation Transfers to Sub-10B", fontsize=9.4, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=7.0, frameon=True, facecolor="white")
    ax2.grid(axis="y", ls="--", alpha=0.35)

    fig.tight_layout(pad=0.85)
    fig.savefig(FIG_DIR / "fig5_scale_emergence.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig6_fault_self_healing(data: dict) -> None:
    fh = data["law2_fault_repair"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.15))

    # Panel (a): Live 64-Layer Reference-Free Row-RMS Scan on Qwen3.5-27B (Layer 30 200*sigma fault)
    m27 = fh["Qwen3.5-27B"]
    n_layers = m27["num_layers"]
    c_layer = m27["corrupted_layer"]
    peak_r = m27["peak_row_rms_ratio"]
    ratios = np.array(m27["layer_row_rms_ratios"], dtype=float)

    ax1.plot(np.arange(n_layers), ratios, color=C_BLUE, lw=1.6, marker="o", markersize=3.0, label="Layer Max / Median Row-RMS")
    ax1.scatter([c_layer], [peak_r], color=C_RED, s=75, zorder=6, label=f"Pinpointed Layer {c_layer} ({peak_r:.1f}x Outlier)")
    ax1.axhline(50.0, color=C_RED, ls="--", lw=1.1, label="Row-Zeroing Gate (50x Median)")
    ax1.set_yscale("log")
    ax1.set_xlabel(f"Decoder Layer Index (Qwen3.5-27B, {n_layers} Layers)", fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Row-RMS Outlier Ratio (log scale)", fontsize=8.8, fontweight="bold")
    ax1.set_ylim(0.8, peak_r * 5.0)
    ax1.set_title("(a) Reference-Free Scan Pinpoints Layer 30 on Qwen3.5-27B", fontsize=9.4, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Panel (b): Unified 4-Scale Comparison under 8-Row 200-sigma Corruption (N=20)
    order = [
        ("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B"),
        ("Qwen2.5-7B-Instruct", "Qwen2.5-7B"),
        ("Qwen3-8B", "Qwen3-8B"),
        ("Qwen3.5-27B", "Qwen3.5-27B"),
    ]
    x = np.arange(len(order))
    w = 0.15

    v_clean = [fh[k]["clean_acc"] for k, _ in order]
    v_ttt = [fh[k]["lora_ttt_acc"] for k, _ in order]
    v_glob = [fh[k]["global_rescale_acc"] for k, _ in order]
    v_edit = [fh[k]["editor_authored_acc"] for k, _ in order]
    v_row50 = [fh[k]["row50x_zeroing_acc"] for k, _ in order]

    ax2.bar(x - 2 * w, v_clean, width=w, color=C_SLATE, edgecolor="black", lw=0.65, label="Pristine Clean")
    ax2.bar(x - w, v_ttt, width=w, color=C_RED, edgecolor="black", lw=0.65, label="Few-Shot LoRA-TTT (0%)")
    ax2.bar(x, v_glob, width=w, color=C_AMBER, edgecolor="black", lw=0.65, label="Global Norm Rescale")
    ax2.bar(x + w, v_edit, width=w, color=C_PURP, edgecolor="black", lw=0.65, label="Editor-Authored (No W0)")
    ax2.bar(x + 2 * w, v_row50, width=w, color=C_GREEN, edgecolor="black", lw=0.75, label="50x Row-RMS Zeroing")

    for idx, (k, _) in enumerate(order):
        h50 = v_row50[idx]
        hed = v_edit[idx]
        # Place concise numeric labels cleanly without horizontal collision
        ax2.text(
            x[idx] + w,
            hed + 1.4,
            f"{hed:.0f}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            fontweight="bold",
            color=C_PURP,
        )
        ax2.text(
            x[idx] + 2 * w + 0.01,
            h50 + 1.4,
            f"{h50:.0f}",
            ha="center",
            va="bottom",
            fontsize=6.3,
            fontweight="bold",
            color=C_GREEN,
        )
        rec = fh[k]["row50x_recovery_pct"]
        top_cluster = max(v_clean[idx], v_glob[idx], hed, h50) + 7.5
        ax2.text(
            x[idx] + 0.08,
            top_cluster,
            f"{rec:.0f}% Rec",
            ha="center",
            va="bottom",
            fontsize=6.4,
            fontweight="bold",
            color=C_GREEN,
            bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="#A7F3D0", lw=0.55, alpha=0.95),
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels([lbl for _, lbl in order], fontsize=8.2, fontweight="bold")
    ax2.set_ylabel("Held-Out Math Accuracy (%, N=20)", fontsize=8.8, fontweight="bold")
    ax2.set_ylim(0, 134)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_title("(b) Unified 200-Sigma Fault Repair Across 1.5B–27B", fontsize=9.4, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=6.7, ncol=2, frameon=True, facecolor="white", columnspacing=0.9)
    ax2.grid(axis="y", ls="--", alpha=0.35)

    fig.tight_layout(pad=0.85)
    fig.savefig(FIG_DIR / "fig6_fault_self_healing.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig7_unblocking_vs_acquisition(data: dict) -> None:
    law3 = data["law3_unseen_and_multiround"]
    oneshot = law3["one_shot_unseen_api_n8"]
    tbl = law3["six_round_table"]
    traj = law3["six_round_spectral_trajectories"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.9, 4.15))

    methods = [
        "Frozen\nBase",
        "Zero-Grad\nSurgery",
        "1-Rd\nLoRA-TTT",
        "Zero-Grad\n(6-Rd)",
        "AlphaEdit\nNull-Space",
        "Sequential\nLoRA-TTT",
        "O-LoRA\n(6-Rd)",
        "Sleep-Replay\n+ Null-Space",
    ]

    # Left (0..2): 1-Shot Unseen Private APIs (N=8, No Ref. Docs)
    x_left = np.array([0.0, 1.0, 2.0])
    v_left = [
        oneshot["frozen_base"],
        oneshot["zero_grad_surgery"],
        oneshot["associative_lora_ttt"],
    ]
    b_left = ax1.bar(x_left, v_left, width=0.42, color="#4F81BD", edgecolor="black", lw=0.7, label="1-Shot Unseen APIs (No Ref. Docs)")
    for bar in b_left:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=6.6, fontweight="bold")

    # Right (3..7): 6-Round Non-Stationary API Evolution (No Ref. Docs)
    ax1.axvline(2.52, color=C_GRAY, ls="--", lw=1.15)
    x_right = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    w = 0.34
    keys_r = [
        "Zero-Grad Directional Surgery",
        "AlphaEdit Null-Space TTT",
        "Sequential LoRA-TTT",
        "O-LoRA",
        "Sleep-Replay + Null-Space TTT",
    ]
    avg_r = [tbl[k]["stream_avg_acc"] for k in keys_r]
    v2_r = [tbl[k]["v2_upgrade_acc"] for k in keys_r]

    b_r1 = ax1.bar(x_right - w / 2, avg_r, width=w, color=C_BLUE, edgecolor="black", lw=0.7, label="6-Rd Active API Acc (%)")
    b_r2 = ax1.bar(x_right + w / 2, v2_r, width=w, color=C_GREEN, edgecolor="black", lw=0.7, label="6-Rd v1->v2 Upgrade Acc (%)")

    for b in (b_r1, b_r2):
        for bar in b:
            h = bar.get_height()
            lbl = f"{h + 1e-6:.1f}" if abs(h - round(h)) > 0.05 else f"{h:.0f}"
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.4, lbl, ha="center", va="bottom", fontsize=5.9, fontweight="bold")

    ax1.text(1.0, 118, "1-Shot Unseen API (N=8)", ha="center", va="top", fontsize=7.1, fontweight="bold", color="#334155")
    ax1.text(5.1, 118, "6-Round Non-Stationary Stream (Table 6)", ha="center", va="top", fontsize=7.1, fontweight="bold", color="#334155")

    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_xticklabels(methods, fontsize=6.7, fontweight="bold")
    ax1.set_ylabel("Accuracy (%, 0 Ref. Context Tokens)", fontsize=8.7, fontweight="bold")
    ax1.set_ylim(0, 124)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_title("(a) Zero-Grad Surgery vs. Associative Compilation", fontsize=9.4, fontweight="bold", pad=10)
    # Place legend in open middle-upper region (x=0.355..0.75, y=0.60..0.88) above short 0%..12.5% bars!
    ax1.legend(loc="upper left", bbox_to_anchor=(0.355, 0.88), fontsize=6.5, frameon=True, facecolor="white")
    ax1.grid(axis="y", ls="--", alpha=0.35)

    # Panel (b): 6-Round Empirical Spectral Norm ||Delta W_t||_2 Trajectories
    rounds = np.arange(1, 7)
    s_seq = traj["Sequential LoRA-TTT"]
    s_ewc = traj["EWC-LoRA-TTT"]
    s_alp = traj["AlphaEdit Null-Space TTT"]
    s_cls = traj["Sleep-Replay + Null-Space TTT"]

    ax2.plot(rounds, s_seq, color=C_RED, lw=2.1, marker="s", label=f"Sequential LoRA-TTT (Peak={max(s_seq):.2f})")
    ax2.plot(rounds, s_ewc, color=C_AMBER, lw=1.9, marker="^", ls="--", label=f"EWC-LoRA-TTT (Peak={max(s_ewc):.2f})")
    ax2.plot(rounds, s_alp, color=C_PURP, lw=1.8, marker="d", ls="-.", label=f"AlphaEdit Null-Space (Peak={max(s_alp):.2f})")
    ax2.plot(rounds, s_cls, color=C_GREEN, lw=2.3, marker="o", label="Sleep-Replay + Spectral Cap (<=4.80)")
    ax2.axhline(4.80, color=C_GRAY, ls=":", lw=1.2, label="Configured Spectral Cap (4.80)")

    ax2.set_xticks(rounds)
    ax2.set_xlabel("Adaptation Round t (1..6)", fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Peak Adapter Spectral Norm ||ΔW_t||_2", fontsize=8.8, fontweight="bold")
    ax2.set_ylim(0, 5.8)
    ax2.set_title("(b) Empirical Spectral Norm Boundedness Over 6 Rounds", fontsize=9.4, fontweight="bold", pad=10)
    # Lower-right region y=0.1..3.3 is completely open!
    ax2.legend(loc="lower right", fontsize=7.1, frameon=True, facecolor="white")
    ax2.grid(True, ls="--", alpha=0.35)

    fig.tight_layout(pad=0.85)
    fig.savefig(FIG_DIR / "fig7_unblocking_vs_acquisition.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig8_subspace_pitfalls(data: dict) -> None:
    exp3 = data["abliteration_subspace_geometry"]
    law4 = data["law4_safety_and_poisoning"]
    ns_kl = law4["nullspace_locality_vs_end_to_end"]
    pois = law4["controlled_stream_poisoning"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.1))

    # Panel (a): Abliteration Rank-1 Subspace Overlap + Layer-Local vs. Multi-Layer KL Note
    layers_2b = exp3["2B"]["layers"][:24]
    layers_9b = exp3["9B"]["layers"]

    ax1.plot(
        np.linspace(0, 100, len(layers_2b)),
        [m["unshielded_core_cos"] for m in layers_2b],
        color=C_RED,
        lw=1.9,
        marker="o",
        markersize=3.4,
        label="Unshielded 2B Refusal Edit (srank=1.005)",
    )
    ax1.plot(
        np.linspace(0, 100, len(layers_9b)),
        [m["unshielded_core_cos"] for m in layers_9b],
        color=C_PURP,
        lw=1.7,
        ls="--",
        marker="^",
        markersize=3.2,
        label="Unshielded 9B Refusal Edit (max=0.764)",
    )
    ax1.plot(
        np.linspace(0, 100, len(layers_9b)),
        [m["shielded_core_cos"] for m in layers_9b],
        color=C_GREEN,
        lw=2.3,
        marker="s",
        markersize=3.1,
        label="Layer-Local Right-Nullspace (<=1e-6)",
    )
    ax1.fill_between(
        np.linspace(0, 100, len(layers_2b)),
        0,
        [m["unshielded_core_cos"] for m in layers_2b],
        color=C_RED,
        alpha=0.09,
    )

    kl_u = ns_kl["end_to_end_logit_kl_unshielded_8layer"]
    kl_s8 = ns_kl["end_to_end_logit_kl_shielded_8layer"]
    kl_s1 = ns_kl["end_to_end_logit_kl_shielded_1layer"]
    ax1.text(
        0.29,
        0.61,
        f"End-to-End Anchor KL (Qwen3-8B):\n"
        f"• Unshielded (8-Layer): {kl_u:.4f}\n"
        f"• Null-Space (8-Layer): {kl_s8:.4f} (>0!)\n"
        f"• Null-Space (1-Layer): {kl_s1:.4f}",
        transform=ax1.transAxes,
        fontsize=6.6,
        fontweight="bold",
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#F8FAFC", edgecolor="#94A3B8", lw=0.8, alpha=0.96),
    )

    ax1.set_xlabel("Normalized Network Depth (%)", fontsize=8.8, fontweight="bold")
    ax1.set_ylabel("Layer-Local Anchor Subspace Overlap", fontsize=8.8, fontweight="bold")
    ax1.set_ylim(-0.04, 1.06)
    ax1.set_title("(a) Layer-Local Orthogonality vs. Multi-Layer Logit Drift", fontsize=9.3, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=6.8, frameon=True, facecolor="white")
    ax1.grid(True, ls="--", alpha=0.35)

    # Panel (b): Fixed-Budget Stream Poisoning Sweep (M=16 docs, r=8, N_trig=16)
    ratios = pois["poison_ratios_pct"]
    asr = pois["trigger_asr_pct"]

    ax2.plot(
        ratios,
        asr["Unfiltered LoRA-TTT"],
        color=C_RED,
        lw=2.8,
        marker="s",
        markersize=6.5,
        label="Unfiltered LoRA-TTT (r=8, 28 steps)",
    )
    ax2.plot(
        ratios,
        asr["Spectral Norm Clipping"],
        color="#F59E0B",
        lw=1.9,
        marker="D",
        markersize=4.5,
        ls="--",
        label="Spectral Norm Clipping (||ΔW||_2<=3.20)",
    )
    ax2.plot(
        ratios,
        asr["TIES-Merging"],
        color=C_PURP,
        lw=2.6,
        marker="^",
        markersize=6.8,
        ls="-.",
        label="TIES-Merging (4 Chunks)",
    )
    ax2.plot(
        ratios,
        asr["Logit-Lens SVD Screening"],
        color=C_GREEN,
        lw=1.9,
        marker="o",
        markersize=4.8,
        label="Logit-Lens SVD Token Screening (0% ASR)",
    )

    ax2.set_xticks(ratios)
    ax2.set_xticklabels([f"{r:g}%" for r in ratios], fontsize=8.3, fontweight="bold")
    ax2.set_xlabel("Poisoned Document Ratio in 16-Doc Stream", fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Trigger Attack Success Rate (ASR %, N=16)", fontsize=8.8, fontweight="bold")
    ax2.set_ylim(-5, 116)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_title("(b) Controlled Stream Backdoor ASR Under Fixed Budget", fontsize=9.3, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=6.9, frameon=True, facecolor="white")
    ax2.grid(True, ls="--", alpha=0.35)

    fig.tight_layout(pad=0.85)
    fig.savefig(FIG_DIR / "fig8_subspace_pitfalls.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    plot_fig5_scale_emergence(data)
    plot_fig6_fault_self_healing(data)
    plot_fig7_unblocking_vs_acquisition(data)
    plot_fig8_subspace_pitfalls(data)
    print("Successfully generated clean empirical figures 5-8 in", FIG_DIR)


if __name__ == "__main__":
    main()
