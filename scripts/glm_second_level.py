# %%
"""Compare functional alignment methods at the group level.

For each method (and smoothing level), subject maps are the effect sizes
averaged over the aligned sessions, tested against 0 with a sign-flipping
permutation test (FWE-corrected, max-statistic procedure).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_surf_stat_map, show
from nilearn.surface import SurfaceImage, load_surf_data

OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs")
# (alignment method, smoothing FWHM in mm)
CONDITIONS = [
    ("identity", None),
    ("identity", 6.0),
    ("identity", 12.0),
    ("procrustes", None),
    ("spectral_ot", None),
]
MESH_NAME = "fsaverage5"  # mesh of the first-level maps
ALPHA = 0.05
N_PERM = 10000
N_JOBS = 10
# Display
HEMI = "right"
VIEW = "lateral"

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


def signed_neg_log_pvals(imgs, smoothing_fwhm) -> SurfaceImage:
    """FWE-corrected -log10 p-values, signed by the effect direction."""
    design_matrix = pd.DataFrame({"intercept": np.ones(len(imgs))})
    neg_log_pvals = non_parametric_inference(
        imgs,
        design_matrix=design_matrix,
        smoothing_fwhm=smoothing_fwhm,
        n_perm=N_PERM,
        two_sided_test=True,
        n_jobs=N_JOBS,
    )
    z_map = (
        SecondLevelModel(smoothing_fwhm=smoothing_fwhm)
        .fit(imgs, design_matrix=design_matrix)
        .compute_contrast("intercept", output_type="z_score")
    )
    data = {
        hemi: np.sign(z_map.data.parts[hemi]) * neg_log_pvals.data.parts[hemi]
        for hemi in HEMIS.values()
    }
    return SurfaceImage(mesh=z_map.mesh, data=data)


# %%
fsaverage = load_fsaverage(MESH_NAME)
sulcal = load_fsaverage_data(MESH_NAME, data_type="sulcal", mesh_type="pial")

reference_dir = next(
    OUTPUT_ROOT.glob(f"sub-*/ses-*_aligned-{CONDITIONS[0][0]}")
)
quantities = sorted(
    f.name.removeprefix("quantity-").split("_stat-")[0]
    for f in reference_dir.glob("quantity-*_stat-effect_size_hemi-L.gii")
)

results = {}
for method, fwhm in CONDITIONS:
    for quantity in quantities:
        print(f"Computing {method} {fwhm}mm, quantity {quantity}")
        imgs = load_subject_maps(method, quantity, fsaverage["pial"])
        results[method, fwhm, quantity] = signed_neg_log_pvals(imgs, fwhm)

# %%
fig, axes = plt.subplots(
    len(quantities),
    len(CONDITIONS),
    figsize=(4 * len(CONDITIONS), 3 * len(quantities)),
    subplot_kw={"projection": "3d"},
)
for col, (method, fwhm) in enumerate(CONDITIONS):
    label = method if fwhm is None else f"{method} {fwhm:g}mm"
    for row, quantity in enumerate(quantities):
        plot_surf_stat_map(
            fsaverage["inflated"],
            stat_map=results[method, fwhm, quantity],
            hemi=HEMI,
            view=VIEW,
            bg_map=sulcal,
            bg_on_data=True,
            threshold=-np.log10(ALPHA),
            cmap="cold_hot",
            colorbar=False,
            title=f"{quantity} - {label}",
            axes=axes[row, col],
            figure=fig,
        )
fig.suptitle(f"Signed -log10 p (FWE, {N_PERM} perm.), threshold p < {ALPHA}")
show()
