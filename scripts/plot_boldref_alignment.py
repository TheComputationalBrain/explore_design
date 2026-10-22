# %%
from pathlib import Path

from matplotlib.lines import Line2D
from nilearn.datasets import load_mni152_template
from nilearn.plotting import plot_anat

data_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
task_label = "ExplorePlus"
out_dir = Path("/home/plbarbarant/repos/explore_design/outputs/qc/epi_alignment")

mni_template = load_mni152_template()
deriv_dir = data_dir / derivatives_folder
subject_dirs = sorted(deriv_dir.glob("sub-*"))

run_colors = ["r", "g", "b", "y", "m", "c"]  # cycle if more runs than colors

for subject_dir in subject_dirs:
    subject = subject_dir.name.split("-")[1]
    session_dirs = sorted(subject_dir.glob("ses-*"))

    for session_dir in session_dirs:
        session = session_dir.name.split("-")[1]
        boldref_files = sorted(
            session_dir.rglob(f"*task-{task_label}*space-{space}_boldref.nii.gz")
        )
        if not boldref_files:
            continue

        display = plot_anat(mni_template, title=f"sub-{subject} ses-{session}")

        legend_handles = []
        for run_idx, boldref_path in enumerate(boldref_files):
            parts = boldref_path.name.split("_")
            run_label = next(
                (p for p in parts if p.startswith("run-")), f"run-{run_idx}"
            )
            run_label = next(
                (p for p in parts if p.startswith("run-")), f"run-{run_idx}"
            )
            color = run_colors[run_idx % len(run_colors)]
            display.add_edges(boldref_path, color=color)
            legend_handles.append(Line2D([0], [0], color=color, lw=2, label=run_label))

        legend = display.frame_axes.figure.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
        )
        for text in legend.get_texts():
            text.set_color("white")

        display.savefig(out_dir / f"sub-{subject}_ses-{session}_align.png")
        display.close()
