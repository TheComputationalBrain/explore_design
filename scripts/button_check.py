# %%
from pathlib import Path

import pandas as pd


def check_buttons_flip(derivatives_dir, sub, ses, run):
    suffix = f"/sub-{sub}/ses-{ses}/beh/sub-{sub}_ses-{ses}_task-ExplorePlus_run-{run:02d}_desc-formatted_beh.tsv"
    path_beh = str(Path(derivatives_dir)) + suffix
    features = pd.read_csv(path_beh, sep="\t", index_col=None)
    # if the column arm_choice is B and the column "Key" is "y"
    # return 1
    # otherwise return 0
    return (
        features[(features["arm_choice"] == "B") & (features["Key"] == "y")].shape[0]
        > 0
    )


path = "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024/derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0"

# Glob for subjects
SUBJECTS = sorted(list(Path(path).glob("sub-*")))
SUBJECTS = [path.name[-2:] for path in SUBJECTS]

for sub in SUBJECTS:
    # Glob for fmri sessions
    fmri_sessions = list(Path(path).glob(f"sub-{sub}/ses-*/func"))
    # Get the session number
    session_numbers = [int(sess.parent.name.split("-")[1]) for sess in fmri_sessions]
    for ses_num in session_numbers:
        # Glob for runs
        runs = list(
            Path(path).glob(
                f"sub-{sub}/ses-{ses_num}/beh/sub-{sub}_ses-{ses_num}_task-ExplorePlus_run-*_desc-formatted_beh.tsv"
            )
        )
        for run in runs:
            res = check_buttons_flip(
                path, sub, ses_num, int(run.name.split("_")[3].split("-")[1])
            )
            if res:
                print(
                    f"Button flip for sub-{sub}/ses-{ses_num}/run-{int(run.name.split('_')[3].split('-')[1]):02d}: {res}"
                )
## SUB-01 SES-1 ALL RUNS ARE FLIPPED
