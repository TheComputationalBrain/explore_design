# %%
"""Generalization of a functional template to new subjects.

Subjects are split into two groups. A template is learned with fmralign on
the first group, using one session as alignment data. Each subject of the
second group is then aligned to this template with its own alignment session,
and its maps of the other session are predicted from the first group
(``GroupAlignment.predict_subject``). The prediction is scored at each vertex
by the correlation, across quantities, with the subject's actual maps.

Groups (2-fold split of the subjects) and sessions (alignment / evaluation)
are swapped, which gives 4 splits.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fmralign import GroupAlignment
from fmralign.embeddings.whole_brain import get_laplacian_embedding
from fmralign.methods import SpectralOT
from fmralign.metrics import score_voxelwise
from nilearn.datasets import (
    fetch_atlas_surf_destrieux,
    load_fsaverage,
    load_fsaverage_data,
)
from nilearn.plotting import plot_surf_stat_map, show
from nilearn.surface import InMemoryMesh, load_surf_data
from sklearn.model_selection import KFold

# KeOps (geomloss backend) needs the CUDA headers of the environment
os.environ.setdefault("CUDA_HOME", sys.prefix)

OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs")
# METHODS = ["identity"]
METHODS = ["identity", "procrustes", "spectral_ot_geometric", "spectral_ot"]
MESH_NAME = "fsaverage5"  # mesh of the first-level maps
N_JOBS = 10

# SpectralOT (whole hemisphere); the other methods are piecewise on Destrieux
SPECTRAL_REG = 1e-4
SPECTRAL_ALPHA = 0.5
N_EIGENMODES = 3
SPECTRAL_DEVICE = "cuda:0"

HEMIS = {"L": "left", "R": "right"}


def _sessions(subject_dir: Path) -> list[Path]:
    """First-level session folders (aligned outputs excluded), in order."""
    sessions = [
        p for p in subject_dir.glob("ses-*") if "_aligned-" not in p.name
    ]
    return sorted(sessions, key=lambda p: int(p.name.split("-")[1]))


def _load_session(ses_dir: Path, quantities, hemi) -> np.ndarray:
    """Stack effect sizes of all quantities -> (n_quantities, n_vertices)."""
    return np.vstack(
        [
            load_surf_data(
                ses_dir / f"quantity-{q}_stat-effect_size_hemi-{hemi}.gii"
            )
            for q in quantities
        ]
    )


def _make_aligner(method, hemi, mesh, labels) -> GroupAlignment:
    if method in ("spectral_ot", "spectral_ot_geometric"):
        part = mesh.parts[HEMIS[hemi]]
        evecs = get_laplacian_embedding(
            InMemoryMesh(coordinates=part.coordinates, faces=part.faces),
            k=N_EIGENMODES,
        )
        method = SpectralOT(
            evecs=evecs,
            reg=SPECTRAL_REG,
            device=SPECTRAL_DEVICE,
            alpha=1.0 if method == "spectral_ot_geometric" else SPECTRAL_ALPHA,
            backend="geomloss",
        )
        labels = None
    return GroupAlignment(
        method=method, labels=labels, scale_template=True, n_jobs=N_JOBS
    )


# %%
mesh = load_fsaverage(MESH_NAME)["pial"]
destrieux = fetch_atlas_surf_destrieux()  # fsaverage5 parcels
labels = {"L": destrieux["map_left"], "R": destrieux["map_right"]}

session_dirs = {
    d.name: _sessions(d)[:2]
    for d in sorted(OUTPUT_ROOT.glob("sub-*"))
    if len(_sessions(d)) >= 2
}
subjects = np.array(list(session_dirs))
quantities = sorted(
    f.name.removeprefix("quantity-").split("_stat-")[0]
    for f in session_dirs[subjects[0]][0].glob(
        "quantity-*_stat-effect_size_hemi-L.gii"
    )
)
print(f"{len(subjects)} subjects, quantities: {quantities}")

# data[session index][hemi][subject] -> (n_quantities, n_vertices)
data = [
    {
        hemi: {
            s: _load_session(session_dirs[s][i], quantities, hemi)
            for s in subjects
        }
        for hemi in HEMIS
    }
    for i in range(2)
]

# %%
# scores[method][hemi] -> (n_splits * n_test_subjects, n_vertices)
scores = {method: {hemi: [] for hemi in HEMIS} for method in METHODS}
kfold = KFold(n_splits=2, shuffle=True, random_state=0)
for align_ses in (0, 1):
    eval_ses = 1 - align_ses
    for train_idx, test_idx in kfold.split(subjects):
        train, test = subjects[train_idx], subjects[test_idx]
        for method in METHODS:
            for hemi in HEMIS:
                aligner = _make_aligner(method, hemi, mesh, labels[hemi])
                aligner.fit(
                    {s: data[align_ses][hemi][s] for s in train}, y="template"
                )
                X_eval = {s: data[eval_ses][hemi][s] for s in train}
                for s in test:
                    y_pred = aligner.predict_subject(
                        X_eval, data[align_ses][hemi][s]
                    )
                    scores[method][hemi].append(
                        score_voxelwise(
                            data[eval_ses][hemi][s], y_pred, loss="corr"
                        )
                    )
            print(f"Done: align on session {align_ses}, {method}")

# %%
mean_corr = {}
for method in METHODS:
    # Mean over vertices, then over (split, test subject) pairs
    per_subject = np.nanmean(
        np.hstack([np.array(scores[method][h]) for h in HEMIS]), axis=1
    )
    print(
        f"{method}: mean correlation {per_subject.mean():.3f} "
        f"± {per_subject.std():.3f} (sd over {len(per_subject)} predictions)"
    )
    mean_corr[method] = per_subject.mean()

# %%
sulcal = load_fsaverage_data(MESH_NAME, data_type="sulcal", mesh_type="pial")
inflated = load_fsaverage(MESH_NAME)["inflated"]
fig, axes = plt.subplots(
    len(METHODS),
    len(HEMIS),
    figsize=(8, 3 * len(METHODS)),
    subplot_kw={"projection": "3d"},
    squeeze=False,
)
for row, method in enumerate(METHODS):
    for col, (hemi, hemi_name) in enumerate(HEMIS.items()):
        hemi_scores = np.array(scores[method][hemi])
        plot_surf_stat_map(
            inflated.parts[hemi_name],
            stat_map=np.nanmean(hemi_scores, axis=0),
            hemi=hemi_name,
            view="lateral",
            bg_map=sulcal.data.parts[hemi_name],
            bg_on_data=True,
            vmax=1,
            cmap="cold_hot",
            title=f"{method} (mean r = {mean_corr[method]:.3f})\n"
            f"hemi {hemi}: mean r = {np.nanmean(hemi_scores):.3f}",
            axes=axes[row, col],
            figure=fig,
        )
# fig.suptitle("Correlation between predicted and actual maps, per vertex")
fig.savefig(
    OUTPUT_ROOT / "alignment" / "generalization_correlation_per_vertex.png",
    dpi=300,
)
show()
