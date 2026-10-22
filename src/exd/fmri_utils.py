from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import load_fsaverage
from nilearn.glm.first_level import (
    compute_regressor,
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.plotting import plot_design_matrix
from nilearn.surface import SurfaceImage, load_surf_data
from tqdm import tqdm

from exd.events import get_run_events


def first_level_analysis(
    data_dir,
    derivatives_folder,
    task_label,
    subjects,
    quantity_name,
    onset,
    contrast_name,
    dmtx_functor,
):
    derivatives_dir = str(Path(data_dir) / derivatives_folder)
    (
        models_all_subjects,
        run_imgs_all_subjects,
        _,
        confounds_all_subjects,
    ) = first_level_from_bids(
        data_dir,
        task_label,
        space_label=None,
        sub_labels=subjects,
        smoothing_fwhm=3.0,
        high_pass=1 / 128,
        t_r=1.71,
        hrf_model="spm + derivative",
        derivatives_folder=derivatives_folder,
        confounds_strategy=["motion", "global_signal"],
        n_jobs=10,
        verbose=10,
    )
    for i, subject in tqdm(enumerate(subjects), desc="Processing subject: "):
        model = models_all_subjects[i]
        run_imgs = run_imgs_all_subjects[i]
        confounds = confounds_all_subjects[i]

        fmri_sessions = get_fmri_sessions(derivatives_dir, subject, task_label)
        run_map = {
            (
                int(p.split("ses-")[1].split("/")[0]),
                int(p.split("run-")[1].split("_")[0]),
            ): (p, c)
            for p, c in zip(run_imgs, confounds)
        }

        for ses in fmri_sessions:
            output_path_ses = (
                Path("/home/plbarbarant/repos/explore_design/outputs/")
                / f"sub-{subject}"
                / f"ses-{ses}"
            )
            output_path_ses.mkdir(exist_ok=True, parents=True)
            beh_runs = get_subject_session_runs_beh(
                derivatives_dir, subject, ses, task_label
            )
            # Keep only ses, runs with behavioral data attached
            keys = sorted(
                k for k in run_map if k[0] == ses and k[1] in beh_runs
            )

            selected_imgs = [run_map[k][0] for k in keys]
            selected_confounds = [run_map[k][1] for k in keys]
            selected_run_ids = [k[1] for k in keys]

            design_matrices = []
            for run_id, confounds in zip(selected_run_ids, selected_confounds):
                output_path_run = (
                    output_path_ses
                    / f"res_quantity-{quantity_name}_run-{run_id}"
                )
                output_path_run.mkdir(exist_ok=True, parents=True)
                run_events = get_run_events(
                    derivatives_dir,
                    sub=subject,
                    ses=ses,
                    run=run_id,
                    onset=onset,
                )
                dmtx = dmtx_functor(model, run_events, confounds)
                design_matrices.append(dmtx)
                plot_design_matrix(
                    dmtx,
                    rescale=True,
                    output_file=str(output_path_run / "dmtx.png"),
                )

            # Load surface images
            surf_imgs = []
            mesh = load_fsaverage("fsaverage5")["pial"]
            for path in selected_imgs:
                path_surf = path.replace("/func/", "/freesurfer/").replace(
                    "_space-MNI152NLin2009cAsym", ""
                )
                path_lh = path_surf.replace(".nii.gz", "_fsaverage5_lh.gii")
                path_rh = path_surf.replace(".nii.gz", "_fsaverage5_rh.gii")
                left_data = load_surf_data(path_lh).T
                right_data = load_surf_data(path_rh).T
                data = {"left": left_data, "right": right_data}
                surf_imgs.append(SurfaceImage(mesh=mesh, data=data))

            model.fit(run_imgs=surf_imgs, design_matrices=design_matrices)
            for output_type in ["effect_size", "effect_variance", "z_score"]:
                stat_map = model.compute_contrast(
                    contrast_name, output_type=output_type
                )
                stat_map.data.to_filename(
                    output_path_ses
                    / f"quantity_{quantity_name}_stat-{output_type}.gii",
                )


def get_subject_session_runs_beh(data_dir, subject, ses, task_label) -> set:
    """Glob BIDS files to find available run indices for a subject/session."""
    pattern = f"sub-{subject}/ses-{ses}/beh/*task-{task_label}*_run-*_beh.tsv"
    run_files = sorted(Path(data_dir).glob(pattern))
    return set([int(f.stem.split("run-")[1].split("_")[0]) for f in run_files])


def get_fmri_sessions(derivatives_dir, subject, task_label) -> set:
    """Get available fMRI sessions for a subject."""
    pattern = f"sub-{subject}/ses-*/func"
    func_dirs = sorted(Path(derivatives_dir).glob(pattern))
    return set([int(f.parent.name.split("-")[1]) for f in func_dirs])


def make_modulated_dtmx(model, events, confounds, quantity_name):
    # Frame times
    n_scans = len(confounds)
    start_time = model.slice_time_ref * model.t_r
    end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
    frame_times = np.linspace(start_time, end_time, n_scans)

    # Events without modulation
    condition = np.array(
        [
            events["onset"].values,
            events["duration"].values,
            np.ones(len(events)),
        ]
    )
    quantity_no_mod, _ = compute_regressor(
        exp_condition=condition,
        hrf_model=model.hrf_model,
        frame_times=frame_times,
        min_onset=model.min_onset,
    )
    confounds = confounds.copy()
    confounds.insert(0, f"{quantity_name}_no_modulation", quantity_no_mod[:, 0])

    # Modulation of events by the quantity of interest
    modulated_events = pd.DataFrame(
        {
            "onset": events["onset"].values,
            "duration": events["duration"].values,
            "trial_type": f"{quantity_name}",
            "modulation": events[f"{quantity_name}"].values,
        }
    )

    # Replace missing values in confounds
    regs = confounds.fillna(0).to_numpy()

    dmtx = make_first_level_design_matrix(
        frame_times=frame_times,
        events=modulated_events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        fir_delays=model.fir_delays,
        add_regs=regs,
        add_reg_names=confounds.columns.tolist(),
        min_onset=model.min_onset,
    )

    # Remove dummy columns
    dmtx = dmtx.loc[:, ~dmtx.columns.str.startswith("dummy")]
    return dmtx
