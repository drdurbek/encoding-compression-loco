

"""Create a publication figure from NestedLOCO_component_results.csv."""

# Configure paths via tools/run_analysis.py or NADES_DATA_ROOT.
if "data_path" not in globals():
    import os as _path_os
    import sys as _path_sys
    from pathlib import Path as _Path
    _path_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "tools"))
    from project_paths import bind_paths as _bind_paths
    if not _path_os.environ.get("NADES_DATA_ROOT"):
        raise RuntimeError("Run through tools/run_analysis.py --data-root <Curated>, or set NADES_DATA_ROOT.")
    data_path = _bind_paths(_path_os.environ["NADES_DATA_ROOT"])


from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = data_path('')
INPUT = data_path('NestedLOCO_Final60/NestedLOCO_component_results.csv')
OUTPUT_STEM = data_path('NestedLOCO_Final60/NestedLOCO_componentwise_validation')

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.7,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
TEAL, ORANGE, NAVY, GREY, LIGHT_GREY = "#298C8C", "#E07A5F", "#2A6FBB", "#8A8A8A", "#E8E8E8"


def clean_name(value: str) -> str:
    replacements = {
        "Cholinechloride": "Choline chloride", "Citricacid": "Citric acid",
        "Decanoicacid": "Decanoic acid", "Dodecanoicacid": "Dodecanoic acid",
        "Malicacid": "Malic acid", "Malonicacid": "Malonic acid",
        "TrimethylamineN-oxide": "Trimethylamine N-oxide",
    }
    return replacements.get(value, value)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Nested LOCO results not found: {INPUT}")
    df = pd.read_csv(INPUT)
    required = {"held_out_component", "n_test", "n_test_non_NADES", "ROC_AUC", "AP_non_NADES", "balanced_accuracy", "MCC"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    df = df.dropna(subset=["ROC_AUC", "AP_non_NADES", "balanced_accuracy", "MCC"]).copy()
    df = df.sort_values(["ROC_AUC", "held_out_component"]).reset_index(drop=True)
    df["label"] = [f"{clean_name(x)}  (n={n})" for x, n in zip(df["held_out_component"], df["n_test"])]
    df["pr_baseline"] = df["n_test_non_NADES"] / df["n_test"]
    y = np.arange(len(df))

    fig, axes = plt.subplots(
        1, 3, figsize=(11.0, 9.4), sharey=True,
        gridspec_kw={"width_ratios": [1.08, 1.08, 1.28], "wspace": 0.08},
    )
    ax_roc, ax_ap, ax_op = axes

    ax_roc.barh(y, df["ROC_AUC"], color=TEAL, height=0.72, edgecolor="none")
    ax_roc.axvline(0.50, color=GREY, lw=1.0, ls="--")
    ax_roc.axvline(df["ROC_AUC"].mean(), color=NAVY, lw=1.0)
    ax_roc.set(xlim=(0, 1), xlabel="ROC–AUC", yticks=y, yticklabels=df["label"])
    ax_roc.invert_yaxis()
    ax_roc.set_title("a  Ranking transfer", loc="left", fontweight="bold", pad=8)

    ax_ap.barh(y, df["AP_non_NADES"], color=ORANGE, height=0.72, edgecolor="none")
    ax_ap.scatter(df["pr_baseline"], y, color="black", s=11, zorder=3, label="Class-prevalence baseline")
    ax_ap.axvline(df["AP_non_NADES"].mean(), color=NAVY, lw=1.0)
    ax_ap.set(xlim=(0, 1), xlabel="AP, non-NADES class")
    ax_ap.set_title("b  Minority-class discrimination", loc="left", fontweight="bold", pad=8)
    ax_ap.legend(frameon=False, fontsize=7, loc="lower right", handletextpad=0.4)

    ax_op.barh(y, df["balanced_accuracy"], color="#91D1C2", height=0.72, edgecolor="none")
    ax_op.axvline(0.50, color=GREY, lw=1.0, ls="--")
    ax_op.axvline(df["balanced_accuracy"].mean(), color=NAVY, lw=1.0)
    ax_op.set(xlim=(0, 1), xlabel="Balanced accuracy")
    ax_op.set_title("c  Thresholded transfer", loc="left", fontweight="bold", pad=8)
    ax_mcc = ax_op.twiny()
    ax_mcc.scatter(df["MCC"], y, color="#333333", s=14, zorder=4)
    ax_mcc.set(xlim=(-1, 1), xlabel="Matthews correlation coefficient")
    ax_mcc.xaxis.set_label_position("top")
    ax_mcc.xaxis.tick_top()
    ax_mcc.axvline(0, color="#333333", lw=0.65, ls=":")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color=LIGHT_GREY, lw=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
    for side in ("bottom", "left", "right"):
        ax_mcc.spines[side].set_visible(False)
    ax_mcc.tick_params(axis="y", left=False, right=False, labelleft=False, labelright=False)
    handles = [
        mpl.patches.Patch(facecolor="#91D1C2", edgecolor="none", label="Balanced accuracy"),
        mpl.lines.Line2D([], [], color="#333333", marker="o", linestyle="None", markersize=4, label="MCC"),
    ]
    ax_op.legend(handles=handles, frameon=False, fontsize=7, loc="lower right", handletextpad=0.4)
    fig.subplots_adjust(left=0.29, right=0.98, bottom=0.08, top=0.93, wspace=0.08)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(OUTPUT_STEM.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved nested-LOCO figure: {OUTPUT_STEM}")


if __name__ == "__main__":
    main()


