"""Summarize descriptor-family and individual-feature stability in nested LOCO.

Reads the selected 60 descriptors from every outer fold and writes a compact
engineering figure plus auditable CSV tables. No labels or model outcomes are
used to calculate selection frequency.
"""

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
OUTDIR = data_path('NestedLOCO_Final60')
FEATURES_PATH = data_path('NestedLOCO_Final60/NestedLOCO_fold_selected_features.csv')
RESULTS_PATH = data_path('NestedLOCO_Final60/NestedLOCO_component_results.csv')
FIGURE_STEM = data_path('NestedLOCO_Final60/NestedLOCO_representation_stability')

mpl.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

FAMILY_ORDER = ["A–B interactions", "Component B", "Weighted mixture", "Component A", "Arithmetic mean"]
COLORS = {
    "A–B interactions": "#2F5597", "Component B": "#6BAED6",
    "Weighted mixture": "#298C8C", "Component A": "#91D1C2",
    "Arithmetic mean": "#E7A94B", 
}


def family(feature: str) -> str:
    if feature.startswith("A_"):
        return "Component A"
    if feature.startswith("B_"):
        return "Component B"
    if feature.startswith("C_"):
        return "Component C"
    if feature.startswith("D_"):
        return "Component D"
    if feature.startswith("mix_"):
        return "Weighted mixture"
    if feature.startswith("avg_"):
        return "Arithmetic mean"
    if feature.startswith("AB_"):
        return "A–B interactions"
    return "Composition"


def main() -> None:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    selected = pd.read_csv(FEATURES_PATH)
    required = {"held_out_component", "rank_in_selected_set", "feature"}
    if required - set(selected.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(selected.columns))}")
    selected["family"] = selected["feature"].map(family)
    n_folds = selected["held_out_component"].nunique()

    # Individual descriptor recurrence.
    feature_frequency = selected["feature"].value_counts().rename_axis("feature").reset_index(name="n_folds_selected")
    feature_frequency["selection_rate"] = feature_frequency["n_folds_selected"] / n_folds
    feature_frequency["family"] = feature_frequency["feature"].map(family)
    feature_frequency.to_csv(data_path('NestedLOCO_Final60/NestedLOCO_descriptor_selection_frequency.csv'), index=False)

    # Family composition in every independently selected Final60 subset.
    family_counts = (
        selected.groupby(["held_out_component", "family"]).size().unstack(fill_value=0)
        .reindex(columns=FAMILY_ORDER, fill_value=0)
    )
    if RESULTS_PATH.exists():
        results = pd.read_csv(RESULTS_PATH).set_index("held_out_component")
        family_counts = family_counts.join(results[["ROC_AUC"]], how="left").sort_values("ROC_AUC")
    else:
        family_counts = family_counts.sort_index()
    family_counts.to_csv(data_path('NestedLOCO_Final60/NestedLOCO_family_composition.csv'))

    # Publication figure: each row is an independently selected Final60 set;
    # panel b reports recurrence of individual descriptors across all folds.
    y = np.arange(len(family_counts))
    labels = [f"{component}" for component in family_counts.index]
    fig, (ax_family, ax_feature) = plt.subplots(
        1, 2, figsize=(10, 6.5), gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.5}
    )
    left = np.zeros(len(family_counts))
    for name in FAMILY_ORDER:
        values = family_counts[name].to_numpy()
        ax_family.barh(y, values, left=left, height=0.72, color=COLORS[name], edgecolor="none", label=name)
        left += values
    ax_family.set(yticks=y, yticklabels=labels, xlim=(0, 60), xlabel="Descriptors in independently selected Final60 subset")
    ax_family.invert_yaxis()
    ax_family.set_title("a", loc="left", fontweight="bold", pad=8)
    
    top = feature_frequency.head(20).sort_values(["n_folds_selected", "feature"], ascending=[True, True])
    colors = [COLORS[f] for f in top["family"]]
    ax_feature.barh(np.arange(len(top)), top["selection_rate"] * 100, color=colors, height=0.70, edgecolor="none")
    ax_feature.set(yticks=np.arange(len(top)), yticklabels=top["feature"], xlim=(0, 35), xlabel="Nested folds selecting descriptor (%)")
    ax_feature.set_title("b", loc="left", fontweight="bold", pad=8)
    ax_feature.axvline(100 / n_folds, color="#8A8A8A", lw=0.8, ls=":")

    for ax in (ax_family, ax_feature):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#E8E8E8", lw=0.6, zorder=0)
        ax.tick_params(axis="y", length=0)
    legend_handles = [
        mpl.patches.Patch(facecolor=COLORS[name], edgecolor="none", label=name)
        for name in FAMILY_ORDER
    ]
    fig.legend(
        handles=legend_handles, frameon=False, fontsize=8, ncol=len(FAMILY_ORDER),
        loc="center", bbox_to_anchor=(0.6, 0.0),
        handlelength=1.0, columnspacing=1.0, handletextpad=0.4,
    )
    fig.subplots_adjust(left=0.27, right=0.98, bottom=0.08, top=0.94)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(FIGURE_STEM.with_suffix(suffix), dpi=600 if suffix == ".png" else None, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Saved: {data_path('NestedLOCO_Final60/NestedLOCO_representation_stability.png')}")


if __name__ == "__main__":
    main()


