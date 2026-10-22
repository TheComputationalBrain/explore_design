# %%
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_surf_stat_map, show
from nilearn.surface import SurfaceImage

data_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
task_label = "ExplorePlus"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
for quantity_name in ["motor_first_half", "motor_second_half"]:
    imgs_L = sorted(
        list(
            Path("/home/plbarbarant/repos/explore_design/outputs/").rglob(
                f"sub-*/ses-*/quantity_{quantity_name}_stat-z_score_hemi-L*"
            )
        )
    )
    imgs_R = sorted(
        list(
            Path("/home/plbarbarant/repos/explore_design/outputs/").rglob(
                f"sub-*/ses-*/quantity_{quantity_name}_stat-z_score_hemi-R*"
            )
        )
    )
    imgs = []
    subjects_label = []
    sessions = []
    fsaverage_meshes = load_fsaverage()
    for img_L, img_R in zip(imgs_L, imgs_R):
        data = {
            "left": img_L,
            "right": img_R,
        }
        subjects_label.append("sub-" + str(img_L)[51:53])
        sessions.append(str(img_L)[54:59])
        imgs.append(SurfaceImage(mesh=fsaverage_meshes["pial"], data=data))

    def threshold_stat_map(stat_map_img, alpha=0.05, height_control="fdr"):
        """Threshold a stat map using FDR correction."""
        thresholded_map, threshold = threshold_stats_img(
            stat_map_img, alpha=alpha, height_control=height_control
        )
        return thresholded_map, threshold

    # Group images by subject: {subject_label: [img_ses1, img_ses2]}
    subject_to_imgs = {}
    subject_to_sessions = {}
    for sub, ses, img in zip(subjects_label, sessions, imgs):
        subject_to_imgs.setdefault(sub, []).append(img)
        subject_to_sessions.setdefault(sub, []).append(ses)

    # Debug: only keep the first two subjects
    debug_subjects = list(subject_to_imgs.keys())

    # For each subject, threshold both sessions and build a red/blue/green
    # overlap map per hemisphere: 1 = session 1 only, 2 = session 2 only,
    # 3 = overlap. 0 (masked out) = active in neither session.
    sulcal_data = load_fsaverage_data(mesh_type="inflated", data_type="sulcal")
    views = ["lateral", "medial"]
    hemis = ["left", "right"]
    alpha = 1.0

    n_subjects = len(debug_subjects)
    n_cols = 2 * len(hemis) * len(views)

    # Single figure for all subjects: one row per subject.
    fig, axes = plt.subplots(
        n_subjects,
        n_cols,
        figsize=(4 * n_cols, 4 * n_subjects),
        subplot_kw={"projection": "3d"},
        squeeze=False,
    )

    for row, sub in enumerate(debug_subjects):
        print(f"Processing subject {sub} ({row + 1}/{n_subjects})")
        img_ses1, img_ses2 = subject_to_imgs[sub][-2:]
        ses1, ses2 = subject_to_sessions[sub][-2:]

        thresholded_ses1, threshold1 = threshold_stat_map(img_ses1, alpha=alpha)
        thresholded_ses2, threshold2 = threshold_stat_map(img_ses2, alpha=alpha)

        vmax = 5
        vmin = -vmax

        i = 0
        for hemi in hemis:
            for view in views:
                plot_surf_stat_map(
                    fsaverage_meshes["inflated"],
                    stat_map=thresholded_ses1.data.parts[hemi],
                    figure=fig,
                    hemi=hemi,
                    view=view,
                    axes=axes[row, i],
                    threshold=threshold1,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="cold_hot",
                    bg_map=sulcal_data.data.parts[hemi],
                    title=f"{sub}-{ses1}-{hemi}-{view}",
                )
                plot_surf_stat_map(
                    fsaverage_meshes["inflated"],
                    stat_map=thresholded_ses2.data.parts[hemi],
                    figure=fig,
                    hemi=hemi,
                    view=view,
                    axes=axes[row, i + 1],
                    threshold=threshold2,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="cold_hot",
                    bg_map=sulcal_data.data.parts[hemi],
                    title=f"{sub}-{ses2}-{hemi}-{view}",
                )
                i += 2

    fig.suptitle(
        f"{quantity_name} - FDR {alpha}",
        fontsize=14,
    )
    fig.savefig(
        f"/home/plbarbarant/repos/explore_design/outputs/qc/{quantity_name}.png",
        dpi=100,
        bbox_inches="tight",
    )
    show()
