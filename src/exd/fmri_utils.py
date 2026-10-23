from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Memory
from nilearn._utils.logger import find_stack_level
from nilearn._utils.param_validation import (
    check_params,
)
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import (
    FirstLevelModel,
    _check_args_first_level_from_bids,
    _check_kwargs_load_confounds,
    _check_repetition_time,
    _check_slice_time_ref,
    _get_confounds,
    _get_masks_files,
    _get_processed_imgs,
    _list_valid_subjects,
    _make_bids_files_filter,
)
from nilearn.interfaces.bids import parse_bids_filename
from nilearn.interfaces.bids.query import (
    infer_repetition_time_from_dataset,
    infer_slice_timing_start_time_from_dataset,
)
from nilearn.interfaces.bids.utils import bids_entities
from nilearn.masking import intersect_masks
from nilearn.plotting import plot_design_matrix
from nilearn.surface import SurfaceImage, load_surf_data


def first_level_from_bids(
    dataset_path,
    task_label,
    space_label=None,
    sub_labels=None,
    exclude_subjects=None,
    img_filters=None,
    t_r=None,
    slice_time_ref=None,
    hrf_model="glover",
    drift_model="cosine",
    high_pass=0.01,
    drift_order=1,
    fir_delays=None,
    min_onset=-24,
    mask_img=None,
    target_affine=None,
    target_shape=None,
    smoothing_fwhm=None,
    memory=None,
    memory_level=1,
    standardize=False,
    signal_scaling=0,
    noise_model="ar1",
    verbose=0,
    n_jobs=1,
    minimize_memory=True,
    derivatives_folder="derivatives",
    **kwargs,
):
    """Adapted version of nilearn.glm.first_level.first_level_from_bids\
    without event loading."""
    check_params(locals())
    if memory is None:
        memory = Memory(None)
    if space_label is None:
        space_label = "MNI152NLin2009cAsym"

    sub_labels = sub_labels or []
    img_filters = img_filters or []
    exclude_subjects = exclude_subjects or []

    _check_args_first_level_from_bids(
        dataset_path=dataset_path,
        task_label=task_label,
        space_label=space_label,
        sub_labels=sub_labels,
        exclude_subjects=exclude_subjects,
        img_filters=img_filters,
        derivatives_folder=derivatives_folder,
    )

    dataset_path = Path(dataset_path).absolute()

    kwargs_load_confounds, remaining_kwargs = _check_kwargs_load_confounds(
        **kwargs
    )

    if len(remaining_kwargs) > 0:
        raise RuntimeError(
            "Unknown keyword arguments. Keyword arguments should start with "
            f"`confounds_` prefix: {remaining_kwargs}"
        )

    if (
        drift_model is not None
        and kwargs_load_confounds is not None
        and "high_pass" in kwargs_load_confounds.get("strategy", [])
    ):
        if drift_model == "cosine":
            verb = "duplicate"
        if drift_model == "polynomial":
            verb = "conflict with"

        warn(
            f"""Confounds will contain a high pass filter,
 that may {verb} the {drift_model} one used in the model.
 Remember to visualize your design matrix before fitting your model
 to check that your model is not overspecified.""",
            UserWarning,
            stacklevel=find_stack_level(),
        )

    derivatives_path = Path(dataset_path) / derivatives_folder
    derivatives_path = derivatives_path.absolute()

    # Get metadata for models.
    #
    # We do it once and assume all subjects and runs
    # have the same value.

    # Repetition time
    #
    # Try to find a t_r value in the bids datasets
    # If the parameter information is not found in the derivatives folder,
    # a search is done in the raw data folder.
    filters = _make_bids_files_filter(
        task_label=task_label,
        space_label=space_label,
        supported_filters=[
            *bids_entities()["raw"],
            *bids_entities()["derivatives"],
        ],
        extra_filter=img_filters,
        verbose=verbose,
    )
    inferred_t_r = infer_repetition_time_from_dataset(
        bids_path=derivatives_path, filters=filters, verbose=verbose
    )
    if inferred_t_r is None:
        filters = _make_bids_files_filter(
            task_label=task_label,
            supported_filters=[*bids_entities()["raw"]],
            extra_filter=[x for x in img_filters if x[0] != "desc"],
            verbose=verbose,
        )
        inferred_t_r = infer_repetition_time_from_dataset(
            bids_path=dataset_path, filters=filters, verbose=verbose
        )

    if t_r is None and inferred_t_r is not None:
        t_r = inferred_t_r
    if t_r is not None and t_r != inferred_t_r:
        warn(
            f"\n't_r' provided ({t_r}) is different "
            f"from the value found in the BIDS dataset ({inferred_t_r}).\n"
            "Note this may lead to the wrong model specification.",
            stacklevel=find_stack_level(),
            category=RuntimeWarning,
        )
    if t_r is not None:
        _check_repetition_time(t_r)
    else:
        warn(
            "\n't_r' not provided and cannot be inferred from BIDS metadata.\n"
            "It will need to be set manually in the list of models, "
            "otherwise their fit will throw an exception.",
            stacklevel=find_stack_level(),
            category=RuntimeWarning,
        )

    # Slice time correction reference time
    #
    # Try to infer a slice_time_ref value in the bids derivatives dataset.
    #
    # If no value can be inferred, the default value of 0.0 is used.
    filters = _make_bids_files_filter(
        task_label=task_label,
        space_label=space_label,
        supported_filters=[
            *bids_entities()["raw"],
            *bids_entities()["derivatives"],
        ],
        extra_filter=img_filters,
        verbose=verbose,
    )
    StartTime = infer_slice_timing_start_time_from_dataset(
        bids_path=derivatives_path, filters=filters, verbose=verbose
    )
    if StartTime is not None and t_r is not None:
        assert StartTime < t_r
        inferred_slice_time_ref = StartTime / t_r
    else:
        if slice_time_ref is None:
            warn(
                "'slice_time_ref' not provided "
                "and cannot be inferred from metadata.\n"
                "It will be assumed that the slice timing reference "
                "is 0.0 percent of the repetition time.\n"
                "If it is not the case it will need to "
                "be set manually in the generated list of models.",
                stacklevel=find_stack_level(),
            )
        inferred_slice_time_ref = 0.0

    if slice_time_ref is None and inferred_slice_time_ref is not None:
        slice_time_ref = inferred_slice_time_ref
    if slice_time_ref is not None and slice_time_ref != inferred_slice_time_ref:
        warn(
            f"'slice_time_ref' provided ({slice_time_ref}) is different "
            f"from the value found in the BIDS dataset "
            f"({inferred_slice_time_ref}).\n"
            "Note this may lead to the wrong model specification.",
            stacklevel=find_stack_level(),
        )
    if slice_time_ref is not None:
        _check_slice_time_ref(slice_time_ref)

    # Build fit_kwargs dictionaries to pass to their respective models fit
    # Events and confounds files must match number of imgs (runs)
    models = []
    models_run_imgs = []
    models_events = []
    models_confounds = []

    sub_labels = _list_valid_subjects(derivatives_path, sub_labels)
    if len(sub_labels) == 0:
        raise RuntimeError(f"\nNo subject found in:\n {derivatives_path}")

    sub_labels = sorted(set(sub_labels) - set(exclude_subjects))

    for sub_label_ in sub_labels:
        # Create model
        model = FirstLevelModel(
            t_r=t_r,
            slice_time_ref=slice_time_ref,
            hrf_model=hrf_model,
            drift_model=drift_model,
            high_pass=high_pass,
            drift_order=drift_order,
            fir_delays=fir_delays,
            min_onset=min_onset,
            mask_img=None if mask_img == "derivatives" else mask_img,
            target_affine=target_affine,
            target_shape=target_shape,
            smoothing_fwhm=smoothing_fwhm,
            memory=memory,
            memory_level=memory_level,
            standardize=standardize,
            signal_scaling=signal_scaling,
            noise_model=noise_model,
            verbose=verbose,
            n_jobs=n_jobs,
            minimize_memory=minimize_memory,
            subject_label=sub_label_,
        )
        models.append(model)

        imgs, files_to_check = _get_processed_imgs(
            derivatives_path=derivatives_path,
            sub_label=sub_label_,
            task_label=task_label,
            space_label=space_label,
            img_filters=img_filters,
            verbose=verbose,
        )
        models_run_imgs.append(imgs)

        confounds = _get_confounds(
            derivatives_path=derivatives_path,
            sub_label=sub_label_,
            task_label=task_label,
            img_filters=img_filters,
            imgs=files_to_check,
            verbose=verbose,
            kwargs_load_confounds=kwargs_load_confounds,
        )
        models_confounds.append(confounds)

        if mask_img == "derivatives":
            masks = _get_masks_files(
                derivatives_path=derivatives_path,
                sub_label=sub_label_,
                task_label=task_label,
                space_label=space_label,
                img_filters=img_filters,
                imgs=files_to_check,
                verbose=verbose,
            )
            if masks:
                mask_img = intersect_masks(
                    masks, threshold=0.5, connected=False
                )
                model.mask_img = mask_img

    return models, models_run_imgs, models_events, models_confounds


def get_subject_session_runs_beh(data_dir, subject, ses, task_label) -> set:
    """Glob BIDS files to find available run indices for a subject/session."""
    pattern = f"sub-{subject}/ses-{ses}/beh/*task-{task_label}*_run-*_beh.tsv"
    run_files = sorted(Path(data_dir).glob(pattern))
    return {int(f.stem.split("run-")[1].split("_")[0]) for f in run_files}


def get_fmri_sessions(derivatives_dir, subject) -> set:
    """Get available fMRI sessions for a subject."""
    pattern = f"sub-{subject}/ses-*/func"
    func_dirs = sorted(Path(derivatives_dir).glob(pattern))
    return {int(f.parent.name.split("-")[1]) for f in func_dirs}


def get_session_runs(imgs, confounds, derivatives_dir, subject, task_label):
    """Group runs by session, keeping those with both images and behaviour.

    Parameters
    ----------
    imgs : list of str
        Preprocessed BOLD paths of the subject.
    confounds : list of pd.DataFrame
        Confounds of each image, in the same order.
    derivatives_dir : str or Path
        Folder holding the fMRI and behavioural derivatives.
    subject : str
        Subject label, without the "sub-" prefix.
    task_label : str
        Task label of the behavioural files.

    Returns
    -------
    dict
        Session -> sorted list of ``(run, img, confounds)``. Sessions without
        any valid run are left out.
    """
    by_ses_run = {}
    for img, conf in zip(imgs, confounds):
        entities = parse_bids_filename(img)["entities"]
        by_ses_run[int(entities["ses"]), int(entities["run"])] = (img, conf)

    session_runs = {}
    for ses in sorted(get_fmri_sessions(derivatives_dir, subject)):
        beh_runs = get_subject_session_runs_beh(
            derivatives_dir, subject, ses, task_label
        )
        runs = [
            (run, *by_ses_run[s, run])
            for s, run in sorted(by_ses_run)
            if s == ses and run in beh_runs
        ]
        if runs:
            session_runs[ses] = runs
    return session_runs


def load_surface_run(
    path, mesh, mesh_name="fsaverage5", space_label="MNI152NLin2009cAsym"
) -> SurfaceImage:
    """Load the surface pair matching a volumetric preprocessed image path.

    The ``freesurfer`` files store data as (time, vertices), hence the
    transpose.
    """
    surf = path.replace("/func/", "/freesurfer/").replace(
        f"_space-{space_label}", ""
    )
    data = {
        hemi: load_surf_data(
            surf.replace(".nii.gz", f"_{mesh_name}_{h}.gii")
        ).T
        for hemi, h in [("left", "lh"), ("right", "rh")]
    }
    return SurfaceImage(mesh=mesh, data=data)


def make_design_matrix(events, confounds, model) -> pd.DataFrame:
    """Build the design matrix of one run.

    Parameters
    ----------
    events : pd.DataFrame
        nilearn events table (``onset``, ``duration``, ``trial_type``, and
        optionally ``modulation``).
    confounds : pd.DataFrame
        One row per scan; missing values are set to 0.
    model : FirstLevelModel
        Provides the timing (``t_r``, ``slice_time_ref``), HRF and drift
        parameters.

    Returns
    -------
    pd.DataFrame
        Design matrix, one row per scan.
    """
    scans = np.arange(len(confounds))
    frame_times = (scans + model.slice_time_ref) * model.t_r
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


def save_design_matrix(design_matrix, output_dir):
    """Save a design matrix as ``design_matrix.tsv`` and ``.png``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    design_matrix.to_csv(output_dir / "design_matrix.tsv", sep="\t")
    plot_design_matrix(
        design_matrix,
        rescale=True,
        output_file=str(output_dir / "design_matrix.png"),
    )


def save_contrasts(
    model,
    contrasts,
    output_dir,
    output_types=("effect_size", "effect_variance", "z_score"),
):
    """Save each contrast map as ``quantity-<name>_stat-<type>.gii``."""
    for name, expression in contrasts.items():
        for output_type in output_types:
            stat_map = model.compute_contrast(
                expression, output_type=output_type
            )
            stat_map.data.to_filename(
                Path(output_dir) / f"quantity-{name}_stat-{output_type}.gii"
            )
