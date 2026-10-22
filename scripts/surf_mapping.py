import os
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn.glm.first_level.first_level import _list_valid_subjects

from exd.fmri_utils import get_fmri_sessions

FSH = os.environ["FREESURFER_HOME"] = "/usr/local/freesurfer/7.4.1"
os.environ["FS_LICENSE"] = "/home/plbarbarant/freesurfer/license.txt"
data_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
derivatives_dir = Path("derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/")
subjects = _list_valid_subjects(str(data_dir / derivatives_dir), None)


def collapse_gifti(path):
    """Collapse freesurfer output in a single darray"""
    img = nib.load(path)
    data = img.agg_data("time series")  # (n_vertices, n_timepoints)
    new_darray = nib.gifti.GiftiDataArray(data=data.astype(np.float32))
    new_img = nib.gifti.GiftiImage(darrays=[new_darray])
    nib.save(new_img, path)


def project_volume(subject, data_dir, derivatives_dir):
    work_dir = data_dir / derivatives_dir
    fmri_sessions = list(get_fmri_sessions(str(work_dir), subject, "ExplorePlus"))
    os.environ["SUBJECTS_DIR"] = str(work_dir / "freesurfer")

    for ses in fmri_sessions:
        subject_dir = work_dir / f"sub-{subject}" / f"ses-{ses}"
        fs_dir = subject_dir / "freesurfer"
        fs_dir.mkdir(exist_ok=True)

        runs = sorted(
            (subject_dir / "func").glob("sub-*_run-0[0-9]_desc-preproc_bold.nii.gz")
        )

        for img in runs:
            basename = img.stem.replace(".nii", "")

            # Native
            left_fmri_tex = str(fs_dir / (basename + "_individual_lh.gii"))
            right_fmri_tex = str(fs_dir / (basename + "_individual_rh.gii"))

            subprocess.run(
                [
                    f"{FSH}/bin/mri_vol2surf",
                    "--src",
                    str(img),
                    "--o",
                    left_fmri_tex,
                    "--out_type",
                    "gii",
                    "--regheader",
                    f"sub-{subject}",
                    "--hemi",
                    "lh",
                    "--projfrac-avg",
                    "0",
                    "1",
                    "0.2",
                ],
                check=True,
            )

            subprocess.run(
                [
                    f"{FSH}/bin/mri_vol2surf",
                    "--src",
                    str(img),
                    "--o",
                    right_fmri_tex,
                    "--out_type",
                    "gii",
                    "--regheader",
                    f"sub-{subject}",
                    "--hemi",
                    "rh",
                    "--projfrac-avg",
                    "0",
                    "1",
                    "0.2",
                ],
                check=True,
            )

            # fsaverage7
            left_fsaverage7_fmri_tex = str(fs_dir / (basename + "_fsaverage7_lh.gii"))
            right_fsaverage7_fmri_tex = str(fs_dir / (basename + "_fsaverage7_rh.gii"))

            subprocess.run(
                [
                    f"{FSH}/bin/mri_surf2surf",
                    "--srcsubject",
                    f"sub-{subject}",
                    "--srcsurfval",
                    left_fmri_tex,
                    "--trgsubject",
                    "ico",
                    "--trgicoorder",
                    "7",
                    "--trgsurfval",
                    left_fsaverage7_fmri_tex,
                    "--hemi",
                    "lh",
                ],
                check=True,
            )

            subprocess.run(
                [
                    f"{FSH}/bin/mri_surf2surf",
                    "--srcsubject",
                    f"sub-{subject}",
                    "--srcsurfval",
                    right_fmri_tex,
                    "--trgsubject",
                    "ico",
                    "--trgicoorder",
                    "7",
                    "--trgsurfval",
                    right_fsaverage7_fmri_tex,
                    "--hemi",
                    "rh",
                ],
                check=True,
            )

            # fsaverage5
            left_fsaverage5_fmri_tex = str(fs_dir / (basename + "_fsaverage5_lh.gii"))
            right_fsaverage5_fmri_tex = str(fs_dir / (basename + "_fsaverage5_rh.gii"))

            subprocess.run(
                [
                    f"{FSH}/bin/mri_surf2surf",
                    "--srcsubject",
                    f"sub-{subject}",
                    "--srcsurfval",
                    left_fmri_tex,
                    "--trgsubject",
                    "ico",
                    "--trgicoorder",
                    "5",
                    "--trgsurfval",
                    left_fsaverage5_fmri_tex,
                    "--hemi",
                    "lh",
                ],
                check=True,
            )

            subprocess.run(
                [
                    f"{FSH}/bin/mri_surf2surf",
                    "--srcsubject",
                    f"sub-{subject}",
                    "--srcsurfval",
                    right_fmri_tex,
                    "--trgsubject",
                    "ico",
                    "--trgicoorder",
                    "5",
                    "--trgsurfval",
                    right_fsaverage5_fmri_tex,
                    "--hemi",
                    "rh",
                ],
                check=True,
            )

            # Save the Giftis as a single darray
            collapse_gifti(left_fmri_tex)
            collapse_gifti(right_fmri_tex)
            collapse_gifti(left_fsaverage7_fmri_tex)
            collapse_gifti(right_fsaverage7_fmri_tex)
            collapse_gifti(left_fsaverage5_fmri_tex)
            collapse_gifti(right_fsaverage5_fmri_tex)


for subject in subjects:
    project_volume(subject, data_dir, derivatives_dir)
