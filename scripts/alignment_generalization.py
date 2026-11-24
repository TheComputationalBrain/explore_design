# %%
"""Generalization of a functional template to new subjects.

Subjects are split into two groups. A template is learned with fmralign on
the first group, using one session as alignment data. Each subject of the
second group is then aligned to this template with its own alignment session,
and its maps of the other session are predicted from the first group
(``GroupAlignment.predict_subject``).

For each quantity, a one-sample second-level GLM (as in glm_second_level.py)
is fit on the predicted maps of the second group, and another one on their
actual maps (ground truth). The thresholded group z-maps are plotted side by
side, one figure per split.

Groups (2-fold split of the subjects) and sessions (alignment / evaluation)
are swapped, which gives 4 splits.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from fmralign import GroupAlignment
from fmralign.embeddings.whole_brain import get_laplacian_embedding
from fmralign.methods import SpectralOT
from nilearn.datasets import (
    fetch_atlas_surf_destrieux,
    load_fsaverage,
)
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import smooth_img
from nilearn.plotting import plot_surf_contours, plot_surf_stat_map, show
from nilearn.surface import InMemoryMesh, SurfaceImage, load_surf_data
from sklearn.model_selection import KFold

# KeOps (geomloss backend) needs the CUDA headers of the environment
os.environ.setdefault("CUDA_HOME", sys.prefix)

# METHODS = ["identity"]
METHODS = [
    # "identity",
    # "identity_smoothed",
    "procrustes",
    # "spectral_ot_geometric",
    # "spectral_ot",
]
# "identity_smoothed": identity, with the maps used for the prediction smoothed
# with SMOOTHING_FWHM (the predicted subject's actual maps are not smoothed)
# METHODS = ["identity", "identity_smoothed"]
SMOOTHING_FWHM = 6.0
MESH_NAME = "fsaverage5"  # mesh of the first-level maps
OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs") / MESH_NAME
N_JOBS = 10
# Second-level thresholding
ALPHA = 0.05
HEIGHT_CONTROL = "fpr"
# Display
HEMI = "right"
VIEW = "lateral"
OUTLINE_COLOR = (0, 0, 0, 0.5)
# Glasser HCP-MMP1 annotations on fsaverage; fsaverage5 vertices are the first
# 10242 fsaverage vertices
ATLAS_DIR = Path("/home/plbarbarant/repos/explore_design/data")

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
    elif method == "identity_smoothed":
        method = "identity"
    labels = None
    return GroupAlignment(
        method=method,
        labels=labels,
        scale_template=True,
        n_jobs=N_JOBS,
        n_iter=1,
    )


def z_map_and_threshold(maps, q_idx):
    """One-sample z-map of quantity q_idx and its threshold at ALPHA.

    maps: {hemi: {subject: (n_quantities, n_vertices)}}
    """
    imgs = [
        SurfaceImage(
            mesh=mesh, data={HEMIS[h]: maps[h][s][q_idx] for h in HEMIS}
        )
        for s in maps["L"]
    ]
    design_matrix = pd.DataFrame({"intercept": np.ones(len(imgs))})
    z_map = (
        SecondLevelModel()
        .fit(imgs, design_matrix=design_matrix)
        .compute_contrast("intercept", output_type="z_score")
    )
    _, threshold = threshold_stats_img(
        z_map, alpha=ALPHA, height_control=HEIGHT_CONTROL, two_sided=True
    )
    return z_map, threshold


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
if "identity_smoothed" in METHODS:
    smoothed = []
    for session in data:
        imgs = smooth_img(
            [
                SurfaceImage(
                    mesh=mesh,
                    data={HEMIS[h]: session[h][s].T for h in HEMIS},
                )
                for s in subjects
            ],
            SMOOTHING_FWHM,
        )
        smoothed.append(
            {
                h: {
                    s: np.asarray(img.data.parts[HEMIS[h]]).T
                    for s, img in zip(subjects, imgs)
                }
                for h in HEMIS
            }
        )

# %%
# z_maps[split, condition, quantity] -> (unthresholded z-map, threshold)
# conditions: "ground_truth" and every method
z_maps = {}
kfold = KFold(n_splits=2, shuffle=True, random_state=0)
for align_ses in (0, 1):
    eval_ses = 1 - align_ses
    for fold, (train_idx, test_idx) in enumerate(kfold.split(subjects)):
        split = f"align ses {align_ses}, fold {fold + 1} / 2"
        train, test = subjects[train_idx], subjects[test_idx]
        ground_truth = {
            h: {s: data[eval_ses][h][s] for s in test} for h in HEMIS
        }
        predictions = {}
        for method in METHODS:
            predictions[method] = {hemi: {} for hemi in HEMIS}
            for hemi in HEMIS:
                aligner = _make_aligner(method, hemi, mesh, labels[hemi])
                aligner.fit(
                    {s: data[align_ses][hemi][s] for s in train}, y="template"
                )
                eval_data = smoothed if method == "identity_smoothed" else data
                X_eval = {s: eval_data[eval_ses][hemi][s] for s in train}
                for s in test:
                    predictions[method][hemi][s] = aligner.predict_subject(
                        X_eval, data[align_ses][hemi][s]
                    )
        for condition, maps in {
            "ground_truth": ground_truth,
            **predictions,
        }.items():
            for q_idx, quantity in enumerate(quantities):
                z_maps[split, condition, quantity] = z_map_and_threshold(
                    maps, q_idx
                )
        print(f"Done: {split}")

# %%
splits = list(dict.fromkeys(key[0] for key in z_maps))

# %%
# Thresholded group z-maps of each split, as in glm_second_level.py
inflated = load_fsaverage(MESH_NAME)["inflated"]
n_vertices = inflated.parts[HEMI].n_vertices
glasser = SurfaceImage(
    mesh=inflated,
    data={
        hemi: nib.freesurfer.read_annot(
            ATLAS_DIR / f"{h.lower()}h.HCP-MMP1.annot"
        )[0][:n_vertices]
        for h, hemi in HEMIS.items()
    },
)
# All parcels of the displayed hemisphere (0 is the medial wall)
glasser_levels = [
    int(label) for label in np.unique(glasser.data.parts[HEMI]) if label > 0
]

conditions = ["ground_truth", *METHODS]
figures_dir = OUTPUT_ROOT / "alignment"
figures_dir.mkdir(parents=True, exist_ok=True)
for i, split in enumerate(splits):
    fig, axes = plt.subplots(
        len(quantities),
        len(conditions),
        figsize=(4 * len(conditions), 3 * len(quantities)),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )
    for col, condition in enumerate(conditions):
        for row, quantity in enumerate(quantities):
            z_map, threshold = z_maps[split, condition, quantity]
            plot_surf_stat_map(
                inflated,
                stat_map=z_map,
                hemi=HEMI,
                view=VIEW,
                bg_map=np.full(n_vertices, 0.5),
                bg_on_data=False,
                threshold=threshold,
                cmap="cold_hot",
                colorbar=False,
                title=f"{quantity} - {condition} (z > {threshold:.2f})",
                axes=axes[row, col],
                figure=fig,
            )
            plot_surf_contours(
                inflated,
                roi_map=glasser,
                hemi=HEMI,
                levels=glasser_levels,
                colors=[OUTLINE_COLOR] * len(glasser_levels),
                axes=axes[row, col],
                figure=fig,
            )
    fig.suptitle(f"Second-level z-maps ({split}), {HEIGHT_CONTROL} < {ALPHA}")
    fig.savefig(
        figures_dir / f"generalization_zmaps_split-{i + 1}.png", dpi=150
    )
show()
