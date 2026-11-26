#!/usr/bin/env python3
"""Generate Empirical Benchmark Figures (Figs 5-8) from final_survey_benchmark.json + verified multi-round runs."""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCH_ROOT = Path(__file__).resolve().parent
RESULTS_PATH = BENCH_ROOT / "results" / "final_survey_benchmark.json"
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


def plot_fig5_scale_emergence(data: dict) -> None:
    from matplotlib.patches import Patch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.15))

    # Panel (a): Paired Base vs. Post-Edit Accuracy across Qwen3.5-{2B, 4B, 9B, 27B}
    # Ground-truth numbers from Fan et al. (arXiv:2611.21885, Table 2 & Fig. 43) + local sandbox autopsy
    scales = ["Qwen3.5-2B", "Qwen3.5-4B*", "Qwen3.5-9B", "Qwen3.5-27B"]
    base_accs = [9.0, 44.1, 47.5, 57.2]
    x = np.arange(len(scales))
    w = 0.34

    # 1. Base accuracy bars across all 4 scales
    ax1.bar(x - w / 2, base_accs, width=w, color="#CBD5E1", edgecolor="#1E293B", lw=0.8)
    for bx, bh in zip(x - w / 2, base_accs):
        ax1.text(bx, bh + 1.1, f"{bh:.1f}%", ha="center", va="bottom", fontsize=7.1, color="#334155", fontweight="bold")

    # 2a. 2B: No edit submitted -> unchanged at 9.0% (hatched)
    ax1.bar([0 + w / 2], [9.0], width=w, facecolor="#E2E8F0", edgecolor="#475569", hatch="///", lw=0.8)
    ax1.text(0 + w / 2, 9.0 + 1.1, "9.0%", ha="center", va="bottom", fontsize=7.1, color="#475569", fontweight="bold")
    ax1.text(
        0.0, 16.5, "No Edit\n(Denies Weights /\n3D Index Err)",
        ha="center", va="bottom", fontsize=6.7, color=C_BLUE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#CBD5E1", lw=0.5, alpha=0.94),
    )

    # 2b. 4B & 9B: Self-bricked to 0.0% -> show dashed drop envelope + downward arrow + 0.0% floor bar
    for idx, b_acc, tag in [
        (1, 44.1, "Self-Bricks -> 0%\n(-44.1% Session*;\nWithdrawn)"),
        (2, 47.5, "Self-Bricks -> 0%\n(-47.5% Held-Out;\n110.85x Norm)"),
    ]:
        rx = idx + w / 2
        ax1.bar([rx], [b_acc], width=w, facecolor="#FEE2E2", edgecolor=C_RED, ls="--", lw=0.95, alpha=0.55)
        ax1.bar([rx], [1.3], width=w, facecolor=C_RED, edgecolor="#7F1D1D", lw=0.7)
        ax1.annotate(
            "", xy=(rx, 5.6), xytext=(rx, b_acc - 2.6),
            arrowprops=dict(arrowstyle="-|>,head_width=0.24,head_length=0.34", color=C_RED, lw=1.45),
        )
        ax1.text(rx, 1.9, "0.0%", ha="center", va="bottom", fontsize=6.6, color="#7F1D1D", fontweight="bold")
        ax1.text(
            idx, b_acc + 5.6, tag,
            ha="center", va="bottom", fontsize=6.6, color=C_RED, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#FECACA", lw=0.5, alpha=0.94),
        )

    # 2c. 27B: Verified self-compiled improvement 57.2% -> 63.8% (+6.6%)
    ax1.bar([3 + w / 2], [63.8], width=w, color=C_GREEN, edgecolor="#1E293B", lw=0.8)
    ax1.text(
        3 + w / 2 - 0.04, 65.0, "+6.6% Unseen\n(63.8%, 3/8 Sub.)",
        ha="center", va="bottom", fontsize=6.9, color=C_GREEN, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#A7F3D0", lw=0.5, alpha=0.94),
    )

    legend_handles = [
        Patch(facecolor="#CBD5E1", edgecolor="#1E293B", lw=0.8, label="Frozen Base Acc (%)"),
        Patch(facecolor="#E2E8F0", edgecolor="#475569", hatch="///", lw=0.8, label="No Edit Submitted (2B)"),
        Patch(facecolor="#FEE2E2", edgecolor=C_RED, ls="--", lw=0.95, label="Self-Bricked -> 0.0% (4B, 9B)"),
        Patch(facecolor=C_GREEN, edgecolor="#1E293B", lw=0.8, label="Verified Self-Edit (27B)"),
    ]
    ax1.set_xticks(x)
    ax1.set_xticklabels(scales, fontsize=8.4, fontweight="bold")
    ax1.set_ylabel("Math Benchmark Accuracy (%)", fontsize=9.0, fontweight="bold")
    ax1.set_ylim(0, 88)
    ax1.set_title("(a) Autonomic Weight Compilation Emerges at 27B Scale", fontsize=9.5, fontweight="bold", pad=10)
    ax1.legend(handles=legend_handles, loc="upper left", ncol=2, fontsize=6.8, frameon=True, facecolor="white", columnspacing=0.9, handletextpad=0.5)
    ax1.grid(axis="y", ls="--", alpha=0.35)

    # Panel (b): Exact Logged runs/exp5_sculptor_scaling/all_results.json
    exp5 = data["exact_exp5_decoupled_scaling"]
    configs = [
        "1.5B\nSelf-Sculpt",
        "1.5B +\n7B Local",
        "1.5B +\n9B Local",
        "1.5B + Flash\n1KB Cloud",
        "1.5B + Pro\n1KB Cloud",
    ]
    avg_accs = exp5["avg_acc_pct"]
    trap_rates = exp5["avg_trap_pct"]

    xb = np.arange(len(configs))
    cols_b = ["#FCA5A5", C_BLUE, C_BLUE, C_GREEN, C_GREEN]
    bars_b = ax2.bar(xb, avg_accs, width=0.52, color=cols_b, edgecolor="black", lw=0.75, label="8-Round Mean Accuracy (%)")
    for bar, tr in zip(bars_b, trap_rates):
        h = bar.get_height()
        lbl = f"{h:.1f}%\n(Trap {tr:.1f}%)" if tr > 0 else f"{h:.1f}%\n(0% Trap)"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.2, lbl, ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax2.set_xticks(xb)
    ax2.set_xticklabels(configs, fontsize=8.0, fontweight="bold")
    ax2.set_ylabel("8-Round Adaptation AvgAcc (%)", fontsize=9.0, fontweight="bold")
    ax2.set_ylim(0, 58)
    ax2.set_title("(b) Decoupled 1KB IR Auditing Rescues 1.5B Edge Models", fontsize=9.5, fontweight="bold", pad=10)
    ax2.grid(axis="y", ls="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    fig.savefig(FIG_DIR / "fig5_scale_emergence.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig6_fault_self_healing(data: dict) -> None:
    fh = data["sdpa_fault_self_healing"]
    models = ["Qwen2-1.5B", "Qwen2.5-7B", "Qwen3-8B"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    # Panel (a): Exact Reference-Free Row-RMS Scan on Qwen3-8B safetensors (36 layers, Layer 19 = 1397.0x)
    n_layers = fh["Qwen3-8B"]["num_layers"]
    c_layer = fh["Qwen3-8B"]["corrupted_layer"]
    peak_r = fh["Qwen3-8B"]["outlier_ratio"]
    ratios = np.array(fh["Qwen3-8B"]["exact_36_layer_damaged_row_rms_ratios"], dtype=float)

    ax1.plot(np.arange(n_layers), ratios, color=C_BLUE, lw=1.7, marker="o", markersize=3.6, label="Layer Max / Median Row-RMS")
    ax1.scatter([c_layer], [peak_r], color=C_RED, s=75, zorder=6, label=f"Localized Layer {c_layer} ({peak_r:.1f}x Outlier)")
    ax1.axhline(50.0, color=C_RED, ls="--", lw=1.1, label="Outlier Gate Threshold (50x)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Decoder Layer Index (Qwen3-8B, 36 Layers)", fontsize=9.0, fontweight="bold")
    ax1.set_ylabel("Row-RMS Outlier Ratio (log scale)", fontsize=9.0, fontweight="bold")
    ax1.set_ylim(0.8, peak_r * 4.5)
    ax1.set_title("(a) Reference-Free Tensor Scan Pinpoints Damaged Layer", fontsize=9.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="white")
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # Panel (b): Strictly Disaggregated Attribution (Local 7B/8B 5-Method Suite vs. Published 27B 3-Bar Reference)
    labels = ["Qwen2.5-7B\n(Local Math)", "Qwen3-8B\n(Local Math)", "Qwen3.5-27B\n(6-Vision [Ref])"]
    x = np.arange(len(labels))
    w = 0.16

    # Local 7B & 8B (indices 0, 1)
    x_loc = np.array([0.0, 1.0])
    v_clean_loc = [fh["Qwen2.5-7B"]["clean_acc"], fh["Qwen3-8B"]["clean_acc"]]
    v_corr_loc = [fh["Qwen2.5-7B"]["corrupted_acc"], fh["Qwen3-8B"]["corrupted_acc"]]
    v_ttt_loc = [fh["Qwen2.5-7B"]["naive_ttt_acc"], fh["Qwen3-8B"]["naive_ttt_acc"]]
    v_ed_loc = [fh["Qwen2.5-7B"]["self_rewriting_repair_acc"], fh["Qwen3-8B"]["self_rewriting_repair_acc"]]
    v_surg_loc = [fh["Qwen2.5-7B"]["calibrated_row_surgery_acc"], fh["Qwen3-8B"]["calibrated_row_surgery_acc"]]

    ax2.bar(x_loc - 2 * w, v_clean_loc, width=w, color="#CBD5E1", edgecolor="black", lw=0.65, label="Pristine Base")
    ax2.bar(x_loc - w, v_corr_loc, width=w, color=C_RED, edgecolor="black", lw=0.65, label="Corrupted Checkpoint")
    ax2.bar(x_loc, v_ttt_loc, width=w, color=C_AMBER, edgecolor="black", lw=0.65, label="Few-Shot LoRA-TTT (0%)")
    ax2.bar(x_loc + w, v_ed_loc, width=w, color=C_PURP, edgecolor="black", lw=0.65, label="Editor-Authored (No W0)")
    b_surg = ax2.bar(x_loc + 2 * w, v_surg_loc, width=w, color=C_GREEN, edgecolor="black", lw=0.75, label="50x Row-Zeroing (Local)")

    for bar, rtxt in zip(b_surg, [f"{fh['Qwen2.5-7B']['calibrated_recovery_pct']:.0f}% Rec", f"{fh['Qwen3-8B']['calibrated_recovery_pct']:.0f}% Rec"]):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, rtxt, ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=C_GREEN)

    # Published 27B Reference (index 2: ONLY the 3 evaluated quantities from Fan et al. [Ref])
    ax2.axvline(1.52, color=C_GRAY, ls=":", lw=1.1)
    x27 = 2.0
    ax2.bar([x27 - 1.2 * w], [84.6], width=w, color="#CBD5E1", edgecolor="black", lw=0.65)
    ax2.bar([x27], [7.3], width=w, color=C_RED, edgecolor="black", lw=0.65)
    b_27_ed = ax2.bar([x27 + 1.2 * w], [70.2], width=w, color=C_PURP, edgecolor="black", lw=0.75)
    ax2.text(x27 + 1.2 * w, 70.2 + 1.5, "83% Rec\n(15x Clip)", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=C_PURP)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.2, fontweight="bold")
    ax2.set_ylabel("Unseen Task Accuracy (%)", fontsize=9.0, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_title("(b) Reference-Free Fault Repair Across Protocols", fontsize=9.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=6.9, ncol=2, frameon=True, facecolor="white")
    ax2.grid(axis="y", ls="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    fig.savefig(FIG_DIR / "fig6_fault_self_healing.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig7_unblocking_vs_acquisition(data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    uva = data["unblocking_vs_new_skill_8b"]["unseen_private_api_0token_acc"]
    methods = [
        "Frozen\nBase",
        "Zero-Grad\nSurgery",
        "1-Rd\nLoRA-TTT",
        "Zero-Grad\n(8-Rd)",
        "AlphaEdit\n(8-Rd)",
        "Naive TTT\n(8-Rd)",
        "HWGM\n(0-Tok)",
        "HWGM +\nCLS-RAG",
    ]

    # Left regime (indices 0..2): 1-Shot Strictly Unseen Private APIs (single bar per method)
    x_left = np.array([0.0, 1.0, 2.0])
    v_left = [uva["frozen_base"], uva["zero_grad_directional_surgery"], uva["associative_lowrank_ttt"]]
    b_left = ax1.bar(x_left, v_left, width=0.42, color="#4F81BD", edgecolor="black", lw=0.7, label="1-Shot Unseen Private API (0-Tok %)")
    for bar in b_left:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.4, f"{h:.0f}", ha="center", va="bottom", fontsize=6.9, fontweight="bold")

    # Right regime (indices 3..7): 8-Round Non-Stationary HomeoToolBench (paired AvgAcc & v1->v2 Acc)
    ax1.axvline(2.52, color=C_GRAY, ls="--", lw=1.15)
    x_right = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    w = 0.34
    tool_avg = [2.5, 32.9, 35.3, 81.5, 99.5]
    v2_mig = [0.0, 0.0, 0.0, 66.7, 100.0]
    b_r1 = ax1.bar(x_right - w / 2, tool_avg, width=w, color=C_BLUE, edgecolor="black", lw=0.7, label="8-Rd HomeoToolBench AvgAcc (%)")
    b_r2 = ax1.bar(x_right + w / 2, v2_mig, width=w, color=C_GREEN, edgecolor="black", lw=0.7, label="8-Rd v1->v2 Migration Acc (%)")

    for b in (b_r1, b_r2):
        for bar in b:
            h = bar.get_height()
            lbl_s = f"{h:.1f}" if (0 < h < 5) else f"{h:.0f}"
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.4, lbl_s, ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    ax1.text(1.0, 117, "1-Shot Private API", ha="center", va="top", fontsize=7.2, fontweight="bold", color="#334155")
    ax1.text(4.85, 117, "8-Round HomeoToolBench (Table 6)", ha="center", va="top", fontsize=7.2, fontweight="bold", color="#334155")

    ax1.set_xticks(np.arange(len(methods)))
    ax1.set_xticklabels(methods, fontsize=6.9, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=9.0, fontweight="bold")
    ax1.set_ylim(0, 123)
    ax1.set_title("(a) Zero-Grad Surgery vs. Associative Compilation", fontsize=9.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.0, 0.89), fontsize=6.7, frameon=True, facecolor="white")
    ax1.grid(axis="y", ls="--", alpha=0.35)

    # Panel (b): Exact Logged 8-Round post_sigma_max Trajectories (runs/exp1 & runs/exp6)
    rounds = np.arange(1, 9)
    traj = data["exact_multiround_sigma_trajectories"]
    sigma_ewc = traj["ewc_ttt"]
    sigma_naive = traj["naive_ttt"]
    sigma_hwsm = traj["homeo_hwgm"]

    ax2.plot(rounds, sigma_ewc, color=C_AMBER, lw=2.0, marker="^", ls="--", label="EWC-TTT (Peak sigma=4.11, Core=0%)")
    ax2.plot(rounds, sigma_naive, color=C_RED, lw=2.0, marker="s", label="Naive Sequential TTT (Peak sigma=2.36)")
    ax2.plot(rounds, sigma_hwsm, color=C_GREEN, lw=2.4, marker="o", label="Wake-Sleep Homeostasis (sigma*=1.0725)")
    ax2.axhline(1.25, color=C_GRAY, ls=":", lw=1.2, label="Homeostatic Cap (gamma=1.25)")

    ax2.set_xticks(rounds)
    ax2.set_xlabel("Continual Adaptation Round t", fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("Peak Adapter Singular Value (sigma_max)", fontsize=9.0, fontweight="bold")
    ax2.set_ylim(0, 4.5)
    ax2.set_title("(b) Multi-Round Spectral Runaway vs. Fixed-Point Attractor", fontsize=9.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=7.4, frameon=True, facecolor="white")
    ax2.grid(True, ls="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    fig.savefig(FIG_DIR / "fig7_unblocking_vs_acquisition.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig8_subspace_pitfalls(data: dict) -> None:
    exp3 = data["exp3_subspace_geometry"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.9))

    layers_2b = exp3["2B"]["layers"][:24]
    layers_9b = exp3["9B"]["layers"]

    ax1.plot(np.linspace(0, 100, len(layers_2b)), [m["unshielded_core_cos"] for m in layers_2b],
             color=C_RED, lw=1.9, marker="o", markersize=3.5, label="Unshielded 2B Directional Edit (srank=1.004)")
    ax1.plot(np.linspace(0, 100, len(layers_9b)), [m["unshielded_core_cos"] for m in layers_9b],
             color=C_PURP, lw=1.7, ls="--", marker="^", markersize=3.3, label="Unshielded 9B Directional Edit (max=0.764)")
    ax1.plot(np.linspace(0, 100, len(layers_9b)), [m["shielded_core_cos"] for m in layers_9b],
             color=C_GREEN, lw=2.3, marker="s", markersize=3.2, label="Right-Nullspace Shielded (<= 1e-6)")

    ax1.fill_between(np.linspace(0, 100, len(layers_2b)), 0, [m["unshielded_core_cos"] for m in layers_2b],
                     color=C_RED, alpha=0.10)
    ax1.text(0.38, 0.68, "2B Refusal Edit Top-1 SVD = 99.55%",
             transform=ax1.transAxes, fontsize=7.5, fontweight="bold", color=C_RED,
             bbox=dict(boxstyle="round,pad=0.24", facecolor="#FEF2F2", edgecolor=C_RED, lw=0.8))

    ax1.set_xlabel("Normalized Network Depth (%)", fontsize=9.0, fontweight="bold")
    ax1.set_ylabel("Cosine Overlap with Top-16 V_core", fontsize=9.0, fontweight="bold")
    ax1.set_ylim(-0.04, 0.98)
    ax1.set_title("(a) Rank-1 Abliteration Overlap & Right-Nullspace Shielding", fontsize=9.5, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=7.1, frameon=True, facecolor="white")
    ax1.grid(True, ls="--", alpha=0.35)

    # Panel (b): Exact Logged Noise Sweep (runs/exp3_noise_sweep/all_results.json)
    ns = data["exact_exp3_noise_sweep"]
    noise_pct = ns["noise_levels_pct"]
    trap_spec = ns["avg_trap_rate_pct"]["spectral_only"]
    trap_ties = ns["avg_trap_rate_pct"]["ties_merging"]
    trap_naive = ns["avg_trap_rate_pct"]["naive_ttt"]
    trap_hwsm = ns["avg_trap_rate_pct"]["hwgm_homeo"]

    ax2.plot(noise_pct, trap_spec, color=C_AMBER, lw=2.0, marker="d", ls="--", label="Spectral-Only SVD (Peak 33.5% Mean / 100% Rd)")
    ax2.plot(noise_pct, trap_ties, color=C_RED, lw=2.0, marker="^", label="TIES-Merging (21.0% Mean at 70% / 100% Rd)")
    ax2.plot(noise_pct, trap_naive, color=C_PURP, lw=1.9, marker="s", label="Naive Online LoRA-TTT (10.0% Mean / 48.9% Rd)")
    ax2.plot(noise_pct, trap_hwsm, color=C_GREEN, lw=2.5, marker="o", label="Semantic IR Audit + HWGM (0.0% Across All)")

    ax2.set_xticks(noise_pct)
    ax2.set_xticklabels(["0%", "25%", "50%", "70%"])
    ax2.set_xlabel("Poisoned / Trap Document Ratio in Stream", fontsize=9.0, fontweight="bold")
    ax2.set_ylabel("8-Round Mean Trap Assimilation (%)", fontsize=9.0, fontweight="bold")
    ax2.set_ylim(-2, 42)
    ax2.set_title("(b) Weight-Space Poisoning Immunity Across Noise Ratios", fontsize=9.5, fontweight="bold", pad=10)
    ax2.legend(loc="upper left", fontsize=7.3, frameon=True, facecolor="white")
    ax2.grid(True, ls="--", alpha=0.35)

    fig.tight_layout(pad=0.8)
    fig.savefig(FIG_DIR / "fig8_subspace_pitfalls.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    plot_fig5_scale_emergence(data)
    plot_fig6_fault_self_healing(data)
    plot_fig7_unblocking_vs_acquisition(data)
    plot_fig8_subspace_pitfalls(data)
    print("Successfully generated empirical figures 5-8 in", FIG_DIR)


if __name__ == "__main__":
    main()
