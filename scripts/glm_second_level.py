# %%
"""Compare functional alignment methods at the group level.

For each method (and smoothing level), subject maps are the effect sizes
averaged over the aligned sessions, tested against 0 with a one-sample
second-level GLM; z-maps are thresholded with a false discovery rate.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_surf_contours, plot_surf_stat_map, show
from nilearn.surface import SurfaceImage, load_surf_data

OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs")
# Glasser HCP-MMP1 annotations on fsaverage; fsaverage5 vertices are the first
# 10242 fsaverage vertices
ATLAS_DIR = Path("/home/plbarbarant/repos/explore_design/data")
N_VERTICES = 10242
# (alignment method, smoothing FWHM in mm)
METHODS = [
    ("identity", None),
    ("identity", 5.0),
    ("spectral_ot_geometric", None),
    ("spectral_ot", None),
    ("procrustes", None),
]
MESH_NAME = "fsaverage5"  # mesh of the first-level maps
ALPHA = 0.05
HEIGHT_CONTROL = "fpr"
# Display
HEMI = "right"
VIEW = "lateral"
OUTLINE_COLOR = (0, 0, 0, 0.5)

HEMIS = {"L": "left", "R": "right"}


def load_subject_maps(method, quantity, mesh) -> list[SurfaceImage]:
    """One image per subject: effect sizes averaged over aligned sessions."""
    imgs = []
    for subject_dir in sorted(OUTPUT_ROOT.glob("sub-*")):
        ses_dirs = sorted(subject_dir.glob(f"ses-*_aligned-{method}"))
        if not ses_dirs:
            continue
        data = {
            hemi: np.mean(
                [
                    load_surf_data(
                        d / f"quantity-{quantity}_stat-effect_size_hemi-{h}.gii"
                    )
                    for d in ses_dirs
                ],
                axis=0,
            )
            for h, hemi in HEMIS.items()
        }
        imgs.append(SurfaceImage(mesh=mesh, data=data))
    return imgs


def thresholded_z_map(imgs, smoothing_fwhm):
    """One-sample z-map, thresholded with HEIGHT_CONTROL at ALPHA."""
    design_matrix = pd.DataFrame({"intercept": np.ones(len(imgs))})
    z_map = (
        SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
        .fit(imgs, design_matrix=design_matrix)
        .compute_contrast("intercept", output_type="z_score")
    )
    return threshold_stats_img(
        z_map, alpha=ALPHA, height_control=HEIGHT_CONTROL, two_sided=True
    )


# %%
fsaverage = load_fsaverage(MESH_NAME)
sulcal = load_fsaverage_data(MESH_NAME, data_type="sulcal", mesh_type="pial")

glasser = SurfaceImage(
    mesh=fsaverage["inflated"],
    data={
        hemi: nib.freesurfer.read_annot(
            ATLAS_DIR / f"{h.lower()}h.HCP-MMP1.annot"
        )[0][:N_VERTICES]
        for h, hemi in HEMIS.items()
    },
)
# All parcels of the displayed hemisphere (0 is the medial wall)
glasser_levels = [
    int(label) for label in np.unique(glasser.data.parts[HEMI]) if label > 0
]

reference_dir = next(OUTPUT_ROOT.glob(f"sub-*/ses-*_aligned-{METHODS[0][0]}"))
quantities = sorted(
    f.name.removeprefix("quantity-").split("_stat-")[0]
    for f in reference_dir.glob("quantity-*_stat-effect_size_hemi-L.gii")
)

results = {}
for method, fwhm in METHODS:
    for quantity in quantities:
        print(f"Computing {method} {fwhm}mm, quantity {quantity}")
        imgs = load_subject_maps(method, quantity, fsaverage["pial"])
        results[method, fwhm, quantity] = thresholded_z_map(imgs, fwhm)

# %%
fig, axes = plt.subplots(
    len(quantities),
    len(METHODS),
    figsize=(4 * len(METHODS), 3 * len(quantities)),
    subplot_kw={"projection": "3d"},
)
for col, (method, fwhm) in enumerate(METHODS):
    label = method if fwhm is None else f"{method} {fwhm:g}mm"
    for row, quantity in enumerate(quantities):
        z_map, threshold = results[method, fwhm, quantity]
        plot_surf_stat_map(
            fsaverage["inflated"],
            stat_map=z_map,
            hemi=HEMI,
            view=VIEW,
            bg_map=np.full(N_VERTICES, 0.5),
            bg_on_data=False,
            threshold=threshold,
            cmap="cold_hot",
            colorbar=False,
            title=f"{quantity} - {label} (z > {threshold:.2f})",
            axes=axes[row, col],
            figure=fig,
        )
        plot_surf_contours(
            fsaverage["inflated"],
            roi_map=glasser,
            hemi=HEMI,
            levels=glasser_levels,
            colors=[OUTLINE_COLOR] * len(glasser_levels),
            axes=axes[row, col],
            figure=fig,
        )
fig.suptitle(f"Second-level z-maps, {HEIGHT_CONTROL} < {ALPHA}")
show()
