# %%
from collections import defaultdict
from pathlib import Path

import numpy as np
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm import threshold_stats_img
from nilearn.glm.second_level import (
    SecondLevelModel,
    make_second_level_design_matrix,
)
from nilearn.plotting import plot_design_matrix, plot_surf_stat_map, show
from nilearn.surface import SurfaceImage, load_surf_data

data_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
outputs_dir = Path("/home/plbarbarant/repos/explore_design/outputs/")
quantity_name = "motor-LminusR"
imgs_L = sorted(
    outputs_dir.rglob(
        f"sub-*/ses-*/quantity-{quantity_name}_stat-effect_size_hemi-L*"
    )
)
imgs_R = sorted(
    outputs_dir.rglob(
        f"sub-*/ses-*/quantity-{quantity_name}_stat-effect_size_hemi-R*"
    )
)
# %%
# group (L, R) file pairs by subject, across all their sessions
per_subject = defaultdict(list)
for img_L, img_R in zip(imgs_L, imgs_R):
    subject = next(p for p in img_L.parts if p.startswith("sub-"))
    per_subject[subject].append((img_L, img_R))

fsaverage_meshes = load_fsaverage()

# average sessions per subject -> one SurfaceImage per subject
subjects_label = sorted(per_subject)

imgs = []
for subject in subjects_label:
    files = per_subject[subject]
    data_L = np.mean([load_surf_data(str(f_L)) for f_L, _ in files], axis=0)
    data_R = np.mean([load_surf_data(str(f_R)) for _, f_R in files], axis=0)
    imgs.append(
        SurfaceImage(
            mesh=fsaverage_meshes["pial"],
            data={"left": data_L, "right": data_R},
        )
    )

dmtx = make_second_level_design_matrix(
    subjects_label=subjects_label,
)
plot_design_matrix(dmtx)
show()
# %%
model = SecondLevelModel(n_jobs=10, verbose=10)
model.fit(imgs, design_matrix=dmtx)
stat_map = model.compute_contrast("intercept", output_type="z_score")

alpha = 0.25
height_control = "fdr"
thresholded_map, threshold = threshold_stats_img(
    stat_map, alpha=alpha, height_control=height_control
)
fsaverage_sulcal = load_fsaverage_data(data_type="sulcal", mesh_type="pial")

# %%
plot_surf_stat_map(
    fsaverage_meshes["inflated"],
    stat_map=thresholded_map,
    hemi="right",
    view="lateral",
    bg_map=fsaverage_sulcal,
    bg_on_data=True,
    threshold=threshold,
    cmap="cold_hot",
    title=f"Second Level z-score {height_control} {alpha} - {quantity_name}",
)
show()
