# %%
from pathlib import Path

import pandas as pd


def make_events(
    features: pd.DataFrame,
    onset: str,
):
    if onset == "RT":
        onset = (features.trial_start + features.RT) / 1000
    elif onset == "outcome":
        onset = (features.outcome_start) / 1000
    elif onset == "cue":
        onset = features.trial_start / 1000
    duration = onset * 0  # Dirac event
    features = features.drop(columns=["Unnamed: 0"])
    events = pd.concat(
        [onset.rename("onset"), duration.rename("duration"), features],
        axis=1,
    )
    # Drop rows with NaN onset
    events = events.dropna(subset=["onset"])
    return events


def get_run_events(derivatives_dir, sub, ses, run, onset):
    suffix = f"/sub-{sub}/ses-{ses}/beh/sub-{sub}_ses-{ses}_task-ExplorePlus_run-{run:02d}_desc-formatted_beh.tsv"
    path_beh = str(Path(derivatives_dir)) + suffix
    features = pd.read_csv(path_beh, sep="\t", index_col=None)
    events = make_events(features, onset)
    return events
