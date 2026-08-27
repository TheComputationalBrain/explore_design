# %%
from pathlib import Path

from nilearn.glm.first_level.first_level import _list_valid_subjects

from exd.fmri_utils import first_level_analysis, make_modulated_dtmx
from exd.models.ideal_observer import ConfidenceEstimator


def make_dmtx_one_run_Conf(model, events, confounds):
    estimator = ConfidenceEstimator(
        latent_levels=[20, 40, 60, 80],
        sd=events["SD"].unique()[0],
        learning_params={"vol": 0.0416},
        option_cols=["obsA", "obsB"],
    )
    pred = estimator.fit_predict(events)
    events["Conf"] = pred["Conf__obsA"] + pred["Conf__obsB"]

    return make_modulated_dtmx(
        model=model, events=events, confounds=confounds, quantity_name="Conf"
    )


if __name__ == "__main__":
    data_dir = Path(
        "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
    )
    task_label = "ExplorePlus"
    derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
    subjects = _list_valid_subjects(str(data_dir / derivatives_folder), None)
    first_level_analysis(
        data_dir=str(data_dir),
        task_label=task_label,
        derivatives_folder=derivatives_folder,
        subjects=subjects,
        quantity_name="Conf",
        onset="outcome",
        contrast_name="Conf",
        dmtx_functor=make_dmtx_one_run_Conf,
    )
