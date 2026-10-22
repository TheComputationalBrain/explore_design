# %%
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import load_fsaverage
from nilearn.glm.first_level import (
    make_first_level_design_matrix,
)
from nilearn.glm.first_level.first_level import _list_valid_subjects
from nilearn.plotting import plot_design_matrix
from nilearn.surface import SurfaceImage, load_surf_data
from tqdm import tqdm

from exd.events import get_run_events
from exd.fmri_utils import (
    first_level_from_bids,
    get_fmri_sessions,
    get_subject_session_runs_beh,
)
from exd.models.ideal_observer import (
    ConfidenceEstimator,
    IdealObserverEstimator,
    PredictionErrorEstimator,
    SurpriseEstimator,
    UncertaintyEstimator,
)
from exd.models.motor import MotorEstimator

# GLOBALS
OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs")

LATENT_LEVELS = [20, 40, 60, 80]
LEARNING_PARAMS = {"vol": 0.0416}
OPTION_COLS = ["obsA", "obsB"]
OUTCOME_RANGE = (1, 100)
MESH_NAME = "fsaverage5"

# Onsets the regressors lock to; loaded once per run.
ONSETS = ["cue", "outcome", "RT"]

# Contrasts computed from every fit. Keys become output filenames.
CONTRASTS = {
    "Pe": "Pe",
    "Surprise": "Surprise",
    "Confidence": "Confidence",
    "Er": "Er",
    "Eu_ch": "Eu_ch",
    "motor-LminusR": "L - R",
}


# Maps (estimator predictions, events) -> one modulation value per trial.
ModulationFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class Regressor:
    """One regressor family: which events it locks to, and how it is modulated.

    `modulation=None` means a categorical regressor: the estimator returns trial
    labels (e.g. "L"/"R"), which become one column each.
    """

    name: str
    onset: str
    estimator: object
    modulation: ModulationFn | None = None


def sum_over_options(prefix: str) -> ModulationFn:
    """Sum an estimator output over options, e.g. Pe__obsA + Pe__obsB."""
    cols = [f"{prefix}__{opt}" for opt in OPTION_COLS]
    return lambda pred, events: pred[cols].sum(axis=1).to_numpy()


def signed_option_difference(
    prefix: str,
    choice_col: str = "arm_choice",
) -> ModulationFn:
    """Difference between options, signed so it is always (chosen - unchosen)."""

    def fn(pred: pd.DataFrame, events: pd.DataFrame) -> np.ndarray:
        diff = (
            pred[f"{prefix}__{OPTION_COLS[0]}"]
            - pred[f"{prefix}__{OPTION_COLS[1]}"]
        )
        sign = np.where(events[choice_col].eq("A"), 1, -1)
        return diff.to_numpy() * sign

    return fn


def make_regressors(sd: float) -> list[Regressor]:
    shared = {
        "latent_levels": LATENT_LEVELS,
        "sd": sd,
        "learning_params": LEARNING_PARAMS,
        "option_cols": OPTION_COLS,
    }
    return [
        Regressor(
            "Pe",
            "outcome",
            PredictionErrorEstimator(**shared),
            sum_over_options("Pe"),
        ),
        Regressor(
            "Surprise",
            "outcome",
            SurpriseEstimator(**shared, outcome_range=OUTCOME_RANGE),
            sum_over_options("Surprise"),
        ),
        Regressor(
            "Confidence",
            "outcome",
            ConfidenceEstimator(**shared),
            sum_over_options("Confidence"),
        ),
        Regressor(
            "Er",
            "cue",
            IdealObserverEstimator(**shared, outcome_range=OUTCOME_RANGE),
            signed_option_difference("IO"),
        ),
        Regressor(
            "Eu_ch",
            "outcome",
            UncertaintyEstimator(**shared),
            sum_over_options("Eu_ch"),
        ),
        Regressor("Motor", "RT", MotorEstimator(option_cols=OPTION_COLS)),
    ]


def regressor_events(
    regressor: Regressor,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Expand one Regressor into nilearn-style event rows."""
    base = events[["onset", "duration"]]

    if regressor.modulation is None:
        return base.assign(
            trial_type=regressor.estimator.fit_predict(events),
            modulation=1.0,
        )

    pred = regressor.estimator.fit_predict(events)
    return pd.concat(
        [
            base.assign(
                trial_type=f"{regressor.name}_no_modulation",
                modulation=1.0,
            ),
            base.assign(
                trial_type=regressor.name,
                modulation=regressor.modulation(pred, events),
            ),
        ],
    )


def make_full_design_matrix(
    model,
    confounds: pd.DataFrame,
    derivatives_dir: str,
    subject: str,
    ses: int,
    run: int,
) -> pd.DataFrame:
    """Build one design matrix containing every quantity for a single run."""
    events_by_onset = {
        onset: get_run_events(
            derivatives_dir,
            sub=subject,
            ses=ses,
            run=run,
            onset=onset,
        )
        for onset in ONSETS
    }

    sd = events_by_onset["outcome"]["SD"].unique()[0]
    regressors = make_regressors(sd)

    events = pd.concat(
        [regressor_events(r, events_by_onset[r.onset]) for r in regressors],
        ignore_index=True,
    )

    n_scans = len(confounds)
    frame_times = (np.arange(n_scans) + model.slice_time_ref) * model.t_r
    confounds = confounds.fillna(0)

    return make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        fir_delays=model.fir_delays,
        add_regs=confounds.to_numpy(),
        add_reg_names=confounds.columns.tolist(),
        min_onset=model.min_onset,
    )


def parse_ses_run(path: str) -> tuple[int, int]:
    return (
        int(path.split("ses-")[1].split("/", maxsplit=1)[0]),
        int(path.split("run-")[1].split("_", maxsplit=1)[0]),
    )


def load_surface_run(path: str, mesh) -> SurfaceImage:
    """Load the fsaverage5 surface pair matching a volumetric preproc path."""
    surf = path.replace("/func/", "/freesurfer/").replace(
        "_space-MNI152NLin2009cAsym",
        "",
    )
    data = {
        "left": load_surf_data(
            surf.replace(".nii.gz", f"_{MESH_NAME}_lh.gii"),
        ).T,
        "right": load_surf_data(
            surf.replace(".nii.gz", f"_{MESH_NAME}_rh.gii"),
        ).T,
    }
    return SurfaceImage(mesh=mesh, data=data)


def first_level_analysis(
    data_dir,
    derivatives_folder,
    task_label,
    subjects,
    contrasts=CONTRASTS,
    output_root=OUTPUT_ROOT,
):
    derivatives_dir = str(Path(data_dir) / derivatives_folder)
    models, imgs_all, _, confounds_all = first_level_from_bids(
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
    mesh = load_fsaverage(MESH_NAME)["pial"]

    for model, run_imgs, run_confounds, subject in zip(
        tqdm(models, desc="subject"),
        imgs_all,
        confounds_all,
        subjects,
    ):
        # (ses, run) -> (image path, confounds table)
        run_map = {
            parse_ses_run(p): (p, c) for p, c in zip(run_imgs, run_confounds)
        }

        for ses in sorted(
            get_fmri_sessions(derivatives_dir, subject),
        ):
            beh_runs = get_subject_session_runs_beh(
                derivatives_dir,
                subject,
                ses,
                task_label,
            )
            # Keep only runs that have both images and behavioural data
            keys = sorted(
                k for k in run_map if k[0] == ses and k[1] in beh_runs
            )
            if not keys:
                continue

            output_ses = output_root / f"sub-{subject}" / f"ses-{ses}"
            output_ses.mkdir(parents=True, exist_ok=True)

            design_matrices, surf_imgs = [], []
            for _, run in keys:
                img_path, confounds = run_map[(ses, run)]
                dmtx = make_full_design_matrix(
                    model,
                    confounds,
                    derivatives_dir,
                    subject,
                    ses,
                    run,
                )

                output_run = output_ses / f"run-{run}"
                output_run.mkdir(parents=True, exist_ok=True)
                dmtx.to_csv(output_run / "design_matrix.tsv", sep="\t")
                plot_design_matrix(
                    dmtx,
                    rescale=True,
                    output_file=str(output_run / "design_matrix.png"),
                )

                design_matrices.append(dmtx)
                surf_imgs.append(load_surface_run(img_path, mesh))

            model.fit(run_imgs=surf_imgs, design_matrices=design_matrices)

            # Save statistical maps
            for name, expression in contrasts.items():
                for output_type in [
                    "effect_size",
                    "effect_variance",
                    "z_score",
                ]:
                    stat_map = model.compute_contrast(
                        expression,
                        output_type=output_type,
                    )
                    stat_map.data.to_filename(
                        output_ses / f"quantity-{name}_stat-{output_type}.gii",
                    )


if __name__ == "__main__":
    data_dir = Path(
        "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024",
    )
    derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0"
    subjects = _list_valid_subjects(str(data_dir / derivatives_folder), None)
    first_level_analysis(
        data_dir=data_dir,
        derivatives_folder="derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0",
        task_label="ExplorePlus",
        subjects=subjects,
    )
