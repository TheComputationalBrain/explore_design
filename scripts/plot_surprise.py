# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exd.events import get_run_events
from exd.models.ideal_observer import SurpriseEstimator

derivatives_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024/derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0"
)
events = get_run_events(derivatives_dir, sub="01", ses=2, run=2, onset="outcome")
estimator = SurpriseEstimator(
    latent_levels=[20, 40, 60, 80],
    sd=events["SD"].unique()[0],
    learning_params={"vol": 0.0416},
    option_cols=["obsA", "obsB"],
    outcome_range=(1, 100),
    unobserved_value=np.nan,
)
surprise_df = estimator.fit_predict(events)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.2,
        "figure.dpi": 300,
    }
)

COL_A = "#E87840"  # option A — warm orange
COL_B = "#3A86C8"  # option B — mid blue
ALPHA_S = 0.75  # stem marker alpha

# ── Data ───────────────────────────────────────────────────────────────────
surp_A = surprise_df["Surprise__obsA"] * 2
surp_B = surprise_df["Surprise__obsB"] * 2
mean_A = events["A_mean"]
mean_B = events["B_mean"]
trials = np.arange(len(surp_A))

# ── Figure ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 3))


def _stem(ax, x, y, color, label, zorder=2):
    mask = ~np.isnan(y)
    markerline, stemlines, baseline = ax.stem(
        x[mask], y[mask], linefmt=color, markerfmt="o", basefmt=" ", label=label
    )
    plt.setp(stemlines, linewidth=0.9, alpha=ALPHA_S, color=color)
    plt.setp(markerline, markersize=4, alpha=ALPHA_S, color=color, zorder=zorder)


_stem(ax, trials, surp_A.values, COL_A, "Surprise — option A")
_stem(ax, trials, surp_B.values, COL_B, "Surprise — option B", zorder=3)

ax.plot(
    trials, mean_A, color=COL_A, lw=1.6, ls="--", label="True mean — option A", zorder=4
)
ax.plot(
    trials, mean_B, color=COL_B, lw=1.6, ls="--", label="True mean — option B", zorder=4
)

# ── Legend ─────────────────────────────────────────────────────────────────
leg = ax.legend(
    frameon=False,
    fontsize=8,
    ncol=2,
    loc="upper center",  # anchor point of the legend
    bbox_to_anchor=(0.5, -0.15),  # (x, y) relative to axes
)

fig.tight_layout()
fig.savefig(
    "/home/plbarbarant/repos/explore_design/outputs/surprise.png", bbox_inches="tight"
)
plt.show()
