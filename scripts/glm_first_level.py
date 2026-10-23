# %%
"""First-level GLM of the ExplorePlus task on the fsaverage surface.

Three stages, run per subject and session:

1. events: ideal-observer quantities -> nilearn events table per run
2. design matrix: events + confounds -> design matrix per run
3. GLM: fit all runs of the session, save the contrast maps

Generic fMRI helpers (run selection, surface loading, design matrix, saving)
live in :mod:`exd.fmri_utils`; this script holds the task-specific parts.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import load_fsaverage
from nilearn.glm.first_level.first_level import _list_valid_subjects
from tqdm import tqdm

from exd.events import get_run_events
from exd.fmri_utils import (
    first_level_from_bids,
    get_session_runs,
    load_surface_run,
    make_design_matrix,
    save_contrasts,
    save_design_matrix,
)
from exd.models.ideal_observer import (
    ConfidenceEstimator,
    IdealObserverEstimator,
    PredictionErrorEstimator,
    SurpriseEstimator,
    UnchosenConfidenceEstimator,
)
from exd.models.motor import MotorEstimator

# Paths
DATA_DIR = Path(
    "/home/plbarbarant/nasShare/projects/protocols/"
    "ExplorePlus_MeynielPaunovRaglio_2024"
)
DERIVATIVES_FOLDER = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0"
OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs")
TASK_LABEL = "ExplorePlus"

# Ideal observer
LATENT_LEVELS = [20, 40, 60, 80]
LEARNING_PARAMS = {"vol": 0.0416}
OPTION_COLS = ["obsA", "obsB"]
OUTCOME_RANGE = (1, 100)
CHOICE_COL = "arm_choice"

# GLM
T_R = 1.71
SMOOTHING_FWHM = 3.0
HIGH_PASS = 1 / 128
HRF_MODEL = "spm + derivative"
CONFOUNDS_STRATEGY = ["motion", "global_signal"]
N_JOBS = 10
VERBOSE = 10

# Surface
MESH_NAME = "fsaverage5"
VOLUME_SPACE = "MNI152NLin2009cAsym"

# Onsets the regressors lock to; loaded once per run.
ONSETS = ["cue", "outcome", "RT"]
# Onsets that get an unmodulated regressor. RT is left out: responses follow
# the cue closely, so its regressor is highly collinear with the cue one.
UNMODULATED_ONSETS = ["cue", "outcome"]

# Contrasts computed from every fit. Keys become output filenames.
CONTRASTS = {
    "Pe": "Pe",
    "Surprise": "Surprise",
    "Conf_ch": "Conf_ch",
    "Conf_unch": "Conf_unch",
    "Er": "Er",
    "motor-LminusR": "Motor",
}
OUTPUT_TYPES = ("effect_size", "effect_variance", "z_score")


# %%
# Events

# Maps (estimator predictions, events) -> one modulation value per trial.
ModulationFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


@dataclass(frozen=True)
class Regressor:
    """Parametric regressor: the onset it locks to and its modulation."""

    name: str
    onset: str
    estimator: object
    modulation: ModulationFn


def _sum_over_options(prefix: str) -> ModulationFn:
    """Sum an estimator output over options, e.g. Pe__obsA + Pe__obsB."""
    cols = [f"{prefix}__{opt}" for opt in OPTION_COLS]
    return lambda pred, events: pred[cols].sum(axis=1).to_numpy()


def _signed_option_difference(prefix: str) -> ModulationFn:
    """Difference between options, signed as (chosen - unchosen)."""
    col_a, col_b = (f"{prefix}__{opt}" for opt in OPTION_COLS)

    def fn(pred, events):
        sign = np.where(events[CHOICE_COL].eq("A"), 1, -1)
        return (pred[col_a] - pred[col_b]).to_numpy() * sign

    return fn


def _signed_motor_response(pred, events) -> np.ndarray:
    """+1 for a left response, -1 for a right response."""
    return pred["trial_type"].map({"L": 1.0, "R": -1.0}).fillna(0).to_numpy()


def _make_regressors(sd: float) -> list[Regressor]:
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
            _sum_over_options("Pe"),
        ),
        Regressor(
            "Surprise",
            "outcome",
            SurpriseEstimator(**shared, outcome_range=OUTCOME_RANGE),
            _sum_over_options("Surprise"),
        ),
        Regressor(
            "Conf_ch",
            "outcome",
            ConfidenceEstimator(**shared),
            _sum_over_options("Conf_ch"),
        ),
        Regressor(
            "Conf_unch",
            "outcome",
            UnchosenConfidenceEstimator(**shared),
            _sum_over_options("Conf_unch"),
        ),
        Regressor(
            "Er",
            "cue",
            IdealObserverEstimator(**shared, outcome_range=OUTCOME_RANGE),
            _signed_option_difference("IO"),
        ),
        # Signed rather than categorical L/R: L + R would rebuild the RT onset
        # regressor, which is collinear with the cue one
        Regressor(
            "Motor",
            "RT",
            MotorEstimator(option_cols=OPTION_COLS),
            _signed_motor_response,
        ),
    ]


def _onset_events(onset: str, events: pd.DataFrame) -> pd.DataFrame:
    """Unmodulated regressor shared by every quantity locked to `onset`."""
    base = events[["onset", "duration"]]
    return base.assign(trial_type=onset, modulation=1.0)


def _regressor_events(
    regressor: Regressor, events: pd.DataFrame
) -> pd.DataFrame:
    """Modulated rows of one regressor.

    The modulation is mean-centred within the run (nilearn does not do it).
    Otherwise its mean acts as a scaled copy of the unmodulated onset
    regressor, making the two strongly collinear and the onset beta absorb
    part of the parametric effect.
    """
    pred = regressor.estimator.fit_predict(events)
    modulation = regressor.modulation(pred, events)
    return events[["onset", "duration"]].assign(
        trial_type=regressor.name,
        modulation=modulation - modulation.mean(),
    )


def make_events(derivatives_dir, subject, ses, run) -> pd.DataFrame:
    """Build the nilearn events table of one run.

    Parameters
    ----------
    derivatives_dir : str or Path
        Folder holding the formatted behavioural files.
    subject : str
        Subject label, without the "sub-" prefix.
    ses, run : int
        Session and run indices.

    Returns
    -------
    pd.DataFrame
        Columns ``onset``, ``duration``, ``trial_type`` and ``modulation``:
        one unmodulated regressor per onset in ``UNMODULATED_ONSETS`` and one
        mean-centred parametric regressor per quantity.
    """
    # Trial table time-locked to each onset
    trials = {}
    for onset in ONSETS:
        trials[onset] = get_run_events(
            derivatives_dir, sub=subject, ses=ses, run=run, onset=onset
        )
    sd = trials["outcome"]["SD"].unique()[0]

    events = []
    for onset in UNMODULATED_ONSETS:
        events.append(_onset_events(onset, trials[onset]))
    for regressor in _make_regressors(sd):
        events.append(_regressor_events(regressor, trials[regressor.onset]))
    return pd.concat(events, ignore_index=True)


# %%
# GLM


def first_level_analysis(
    subjects, contrasts=CONTRASTS, output_root=OUTPUT_ROOT
):
    """Fit one first-level model per subject and session, save contrasts.

    Parameters
    ----------
    subjects : list of str
        Subject labels, without the "sub-" prefix.
    contrasts : dict, default=CONTRASTS
        Output name -> contrast expression.
    output_root : Path, default=OUTPUT_ROOT
        Maps go to ``sub-<label>/ses-<ses>/``, design matrices to
        ``sub-<label>/ses-<ses>/run-<run>/``.
    """
    derivatives_dir = DATA_DIR / DERIVATIVES_FOLDER
    models, imgs_all, _, confounds_all = first_level_from_bids(
        DATA_DIR,
        TASK_LABEL,
        space_label=VOLUME_SPACE,
        sub_labels=subjects,
        smoothing_fwhm=SMOOTHING_FWHM,
        high_pass=HIGH_PASS,
        t_r=T_R,
        hrf_model=HRF_MODEL,
        derivatives_folder=DERIVATIVES_FOLDER,
        confounds_strategy=CONFOUNDS_STRATEGY,
        n_jobs=N_JOBS,
        verbose=VERBOSE,
    )
    mesh = load_fsaverage(MESH_NAME)["pial"]

    for model, imgs, confounds, subject in zip(
        tqdm(models, desc="subject"), imgs_all, confounds_all, subjects
    ):
        session_runs = get_session_runs(
            imgs, confounds, derivatives_dir, subject, TASK_LABEL
        )
        for ses, runs in session_runs.items():
            output_ses = output_root / f"sub-{subject}" / f"ses-{ses}"

            design_matrices, surf_imgs = [], []
            for run, img_path, run_confounds in runs:
                events = make_events(derivatives_dir, subject, ses, run)
                dmtx = make_design_matrix(events, run_confounds, model)
                save_design_matrix(dmtx, output_ses / f"run-{run}")
                design_matrices.append(dmtx)
                surf_imgs.append(
                    load_surface_run(img_path, mesh, MESH_NAME, VOLUME_SPACE)
                )

            model.fit(run_imgs=surf_imgs, design_matrices=design_matrices)
            save_contrasts(model, contrasts, output_ses, OUTPUT_TYPES)


if __name__ == "__main__":
    derivatives_dir = str(DATA_DIR / DERIVATIVES_FOLDER)
    subjects = _list_valid_subjects(derivatives_dir, None)[:1]
    first_level_analysis(subjects)
