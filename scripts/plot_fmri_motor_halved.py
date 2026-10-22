# %%
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import _list_valid_subjects
from sklearn.base import clone

from exd.fmri_utils import first_level_analysis
from exd.models.motor import MotorEstimator


def make_dmtx_one_run_motor(model, events, confounds, estimator) -> pd.DataFrame:
    algo = clone(estimator)
    events["trial_type"] = algo.fit_predict(events)
    # Create design matrix
    # Replace missing values in confounds
    regs = confounds.fillna(0).to_numpy()
    # Create the design matrix
    n_scans = len(confounds)
    start_time = model.slice_time_ref * model.t_r
    end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
    frame_times = np.linspace(start_time, end_time, n_scans)
    dmtx = make_first_level_design_matrix(
        frame_times=frame_times[n_scans // 2 :],
        events=events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        fir_delays=model.fir_delays,
        add_regs=regs[n_scans // 2 :],
        add_reg_names=confounds.columns.tolist(),
        min_onset=model.min_onset,
    )
    # Remove the dummy* columns
    dtmx = dmtx.loc[:, ~dmtx.columns.str.startswith("dummy")]
    return dtmx


if __name__ == "__main__":
    data_dir = Path(
        "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
    )
    task_label = "ExplorePlus"
    derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
    subjects = _list_valid_subjects(str(data_dir / derivatives_folder), None)
    estimator = MotorEstimator(
        option_cols=["obsA", "obsB"],
    )
    first_level_analysis(
        data_dir=str(data_dir),
        task_label=task_label,
        derivatives_folder=derivatives_folder,
        subjects=subjects,
        quantity_name="motor_second_half",
        onset="RT",
        contrast_name="R - L",
        dmtx_functor=partial(make_dmtx_one_run_motor, estimator=estimator),
    )
