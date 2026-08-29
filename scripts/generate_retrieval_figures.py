#!/usr/bin/env python3
"""Generate publication-ready 300 DPI figures for the Retrieval Ablation Study."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "results/retrieval/retrieval_ablation_summary_table.csv"
FIGURES_DIR = PROJECT_ROOT / "results/retrieval/figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    if not CSV_PATH.exists():
        print(f"❌ File {CSV_PATH} not found!")
        return

    df = pd.read_csv(CSV_PATH)
    print(f"📊 Loaded {len(df)} rows from {CSV_PATH}")

    # =========================================================================
    # Figure 1: Retriever Comparison (BM25 vs Dense vs Hybrid)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    df_k10 = df[(df["Top_K"] == 10) & (df["Top_N"] == 5)].copy()
    df_k10["Method"] = (
        df_k10["Retriever"].str.upper()
        + " ("
        + df_k10["Reranker"].replace({"noop": "No-op", "cross-encoder": "Cross-Encoder"})
        + ")"
    )
    df_k10 = df_k10.sort_values(by="MRR@5", ascending=False)

    colors = ["#2b5c8f" if "Cross-Encoder" in m else "#8fa9c4" for m in df_k10["Method"]]
    bars1 = ax1.barh(df_k10["Method"], df_k10["MRR@5"], color=colors, edgecolor="black", alpha=0.85)
    ax1.set_xlabel("MRR@5 (Mean Reciprocal Rank)", fontsize=11, fontweight="bold")
    ax1.set_title("So Sánh MRR@5 Giữa Các Phương Pháp (K=10)", fontsize=12, fontweight="bold")
    ax1.set_xlim(0, 0.40)
    for bar in bars1:
        ax1.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{bar.get_width():.4f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    df_k10_plot = df_k10.melt(
        id_vars=["Method"], value_vars=["Hit@1", "Hit@5"], var_name="Metric", value_name="Score"
    )
    sns.barplot(
        data=df_k10_plot,
        y="Method",
        x="Score",
        hue="Metric",
        palette=["#e76f51", "#2a9d8f"],
        ax=ax2,
        alpha=0.9,
        edgecolor="black",
    )
    ax2.set_xlabel("Hit Rate Score", fontsize=11, fontweight="bold")
    ax2.set_title("Hit Rate@1 & Hit Rate@5 (K=10)", fontsize=12, fontweight="bold")
    ax2.set_xlim(0, 0.50)
    ax2.set_ylabel("")
    ax2.legend(title="Chỉ số", frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_retriever_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ Created fig1_retriever_comparison.png")

    # =========================================================================
    # Figure 2: Reranker Impact (Before vs After)
    # =========================================================================
    df_pivot = (
        df[df["Top_N"] == 5]
        .pivot_table(index=["Retriever", "Top_K"], columns="Reranker", values="MRR@5")
        .reset_index()
    )
    df_pivot["Gain_Percent"] = (
        (df_pivot["cross-encoder"] - df_pivot["noop"]) / df_pivot["noop"]
    ) * 100
    df_pivot["Label"] = df_pivot["Retriever"].str.upper() + " (K=" + df_pivot["Top_K"].astype(str) + ")"
    df_pivot = df_pivot.sort_values(by="cross-encoder", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(df_pivot))

    for i, row in df_pivot.reset_index().iterrows():
        ax.plot([row["noop"], row["cross-encoder"]], [i, i], color="#999999", linewidth=2, zorder=1)
        gain_text = f"+{row['Gain_Percent']:.1f}%"
        ax.text(
            row["cross-encoder"] + 0.008,
            i,
            gain_text,
            va="center",
            fontweight="bold",
            color="#1b4965",
        )

    ax.scatter(
        df_pivot["noop"],
        y_pos,
        color="#e63946",
        s=120,
        label="Trước Rerank (No-op)",
        zorder=2,
        edgecolors="black",
    )
    ax.scatter(
        df_pivot["cross-encoder"],
        y_pos,
        color="#2a9d8f",
        s=140,
        label="Sau Rerank (Cross-Encoder)",
        zorder=2,
        edgecolors="black",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_pivot["Label"], fontsize=11, fontweight="bold")
    ax.set_xlabel("MRR@5 Score", fontsize=12, fontweight="bold")
    ax.set_title("Mức Tăng Trưởng MRR@5 Nhờ Cross-Encoder Reranker", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlim(0.12, 0.40)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_reranker_impact.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ Created fig2_reranker_impact.png")

    # =========================================================================
    # Figure 3: Top-K Sensitivity (K=5, 10, 20)
    # =========================================================================
    df_ce = df[(df["Reranker"] == "cross-encoder") & (df["Top_N"] == 5)].sort_values(by="Top_K")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"bm25": "#e76f51", "hybrid": "#2a9d8f", "dense": "#457b9d"}

    for ret in ["bm25", "hybrid", "dense"]:
        sub = df_ce[df_ce["Retriever"] == ret]
        ax1.plot(
            sub["Top_K"],
            sub["MRR@5"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=ret.upper(),
            color=palette[ret],
        )
        for _, row in sub.iterrows():
            ax1.annotate(
                f"{row['MRR@5']:.4f}",
                (row["Top_K"], row["MRR@5"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax1.set_title("MRR@5 Theo Độ Sâu Top-K Ban Đầu", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Initial Top-K", fontsize=11, fontweight="bold")
    ax1.set_ylabel("MRR@5", fontsize=11, fontweight="bold")
    ax1.set_xticks([5, 10, 20])
    ax1.set_ylim(0.22, 0.36)
    ax1.legend(title="Retriever", frameon=True)

    for ret in ["bm25", "hybrid", "dense"]:
        sub = df_ce[df_ce["Retriever"] == ret]
        ax2.plot(
            sub["Top_K"],
            sub["Hit@5"],
            marker="s",
            linewidth=2.5,
            markersize=8,
            label=ret.upper(),
            color=palette[ret],
        )
        for _, row in sub.iterrows():
            ax2.annotate(
                f"{row['Hit@5']*100:.1f}%",
                (row["Top_K"], row["Hit@5"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    ax2.set_title("Hit Rate@5 (%) Theo Độ Sâu Top-K Ban Đầu", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Initial Top-K", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Hit Rate@5", fontsize=11, fontweight="bold")
    ax2.set_xticks([5, 10, 20])
    ax2.set_ylim(0.25, 0.45)
    ax2.legend(title="Retriever", frameon=True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_topk_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ Created fig3_topk_sensitivity.png")

    # =========================================================================
    # Figure 4: Pareto Frontier (Accuracy vs Latency)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(11, 7))
    df_p = df[df["Top_N"] == 5].copy()
    ret_colors = {"bm25": "#e76f51", "hybrid": "#2a9d8f", "dense": "#457b9d"}
    rerank_markers = {"noop": "o", "cross-encoder": "*"}

    for _, row in df_p.iterrows():
        color = ret_colors[row["Retriever"]]
        marker = rerank_markers[row["Reranker"]]
        size = 280 if row["Reranker"] == "cross-encoder" else 120
        ax.scatter(
            row["P50_Latency_ms"],
            row["MRR@5"],
            color=color,
            marker=marker,
            s=size,
            edgecolors="black",
            alpha=0.85,
            zorder=3,
        )

    pareto_points = [
        df_p[(df_p["Retriever"] == "dense") & (df_p["Reranker"] == "noop") & (df_p["Top_K"] == 10)].iloc[0],
        df_p[(df_p["Retriever"] == "bm25") & (df_p["Reranker"] == "noop") & (df_p["Top_K"] == 10)].iloc[0],
        df_p[(df_p["Retriever"] == "bm25") & (df_p["Reranker"] == "cross-encoder") & (df_p["Top_K"] == 10)].iloc[0],
        df_p[(df_p["Retriever"] == "bm25") & (df_p["Reranker"] == "cross-encoder") & (df_p["Top_K"] == 20)].iloc[0],
    ]
    pareto_lat = [p["P50_Latency_ms"] for p in pareto_points]
    pareto_mrr = [p["MRR@5"] for p in pareto_points]
    ax.plot(
        pareto_lat,
        pareto_mrr,
        color="#d90429",
        linestyle="--",
        linewidth=2.5,
        label="Đường biên Pareto Tối ưu (Pareto Frontier)",
        zorder=2,
    )

    ax.annotate(
        "Ultra-Fast Tier\nDense No-op (9.2ms, MRR: 0.1895)",
        xy=(9.2, 0.1895),
        xytext=(20, 0.16),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontweight="bold",
        fontsize=9,
    )
    ax.annotate(
        "Balanced Tier\nBM25 No-op (47.3ms, MRR: 0.2425)",
        xy=(47.3, 0.2425),
        xytext=(55, 0.21),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontweight="bold",
        fontsize=9,
    )
    ax.annotate(
        "High Accuracy Tier\nBM25 + CE K10 (117ms, MRR: 0.3067)",
        xy=(117.2, 0.3067),
        xytext=(80, 0.325),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontweight="bold",
        fontsize=9,
    )
    ax.annotate(
        "🏆 GOLDEN RETRIEVER\nBM25 + CE K20 (182ms, MRR: 0.3365)",
        xy=(182.4, 0.3365),
        xytext=(130, 0.355),
        arrowprops=dict(arrowstyle="->", color="#d90429", lw=1.8),
        fontweight="bold",
        fontsize=10,
        color="#d90429",
    )

    ax.set_title("Biểu Đồ Đường Biên Pareto: Chất Lượng (MRR@5) vs Độ Trễ P50 (ms)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Độ Trễ P50 (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Độ Chính Xác MRR@5", fontsize=12, fontweight="bold")
    ax.set_xlim(-5, 230)
    ax.set_ylim(0.15, 0.38)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_pareto_frontier.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ Created fig4_pareto_frontier.png")

    # =========================================================================
    # Figure 5: Latency Breakdown (Retrieve vs Rerank)
    # =========================================================================
    df_lat = df[(df["Reranker"] == "cross-encoder") & (df["Top_N"] == 5)].copy()
    df_lat["Label"] = df_lat["Retriever"].str.upper() + " (K=" + df_lat["Top_K"].astype(str) + ")"
    df_lat = df_lat.sort_values(by="P50_Latency_ms", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    y_pos = range(len(df_lat))
    bars1 = ax.barh(
        df_lat["Label"],
        df_lat["Retrieve_Mean_ms"],
        color="#457b9d",
        edgecolor="black",
        label="Thời gian Truy xuất (Retrieve ms)",
        alpha=0.85,
    )
    bars2 = ax.barh(
        df_lat["Label"],
        df_lat["Rerank_Mean_ms"],
        left=df_lat["Retrieve_Mean_ms"],
        color="#e76f51",
        edgecolor="black",
        label="Thời gian Rerank (Cross-Encoder ms)",
        alpha=0.85,
    )

    ax.set_xlabel("Thời Gian Thực Thi Trung Bình (ms)", fontsize=11, fontweight="bold")
    ax.set_title("Phân Bổ Thời Gian Giữa Bước Truy Xuất & Bước Rerank", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 240)
    ax.legend(loc="lower right", frameon=True)

    for i, row in df_lat.reset_index().iterrows():
        total = row["Retrieve_Mean_ms"] + row["Rerank_Mean_ms"]
        p50 = row["P50_Latency_ms"]
        ax.text(total + 3, i, f"{total:.1f} ms (P50: {p50} ms)", va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_latency_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ Created fig5_latency_breakdown.png")

    print(f"\n🎉 Toàn bộ 5 biểu đồ chất lượng cao 300 DPI đã được lưu vào: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
