# %%
"""Functional alignment of first-level maps across subjects with fmralign.

For each subject, the alignment is learned on one session and applied to the
other one. Aligned effect sizes of all quantities are saved, per method, to
``sub-*/ses-*_aligned-<method>/quantity-*_stat-effect_size_hemi-*.gii``.
"""

from pathlib import Path

import numpy as np
from fmralign import GroupAlignment
from fmralign.embeddings.whole_brain import get_laplacian_embedding
from fmralign.methods import SpectralOT
from nilearn.datasets import fetch_atlas_surf_destrieux, load_fsaverage
from nilearn.surface import InMemoryMesh, SurfaceImage, load_surf_data

OUTPUT_ROOT = Path("/home/plbarbarant/repos/explore_design/outputs")
METHODS = ["identity", "procrustes", "spectral_ot"]
# Index of the session used to learn the alignment; the other one is aligned
ALIGNMENT_SESSION = 0
MESH_NAME = "fsaverage5"  # mesh of the first-level maps
N_JOBS = 10

# SpectralOT (whole hemisphere); the other methods are piecewise on Destrieux
SPECTRAL_REG = 1e-4
SPECTRAL_ALPHA = 0.5
N_EIGENMODES = 3
SPECTRAL_DEVICE = "cuda:0"

HEMIS = {"L": "left", "R": "right"}


def _sessions(subject_dir: Path) -> list[Path]:
    """First-level session folders (aligned outputs excluded), in order."""
    sessions = [
        p for p in subject_dir.glob("ses-*") if "_aligned-" not in p.name
    ]
    return sorted(sessions, key=lambda p: int(p.name.split("-")[1]))


def _load_session(ses_dir: Path, quantities, hemi) -> np.ndarray:
    """Stack effect sizes of all quantities -> (n_quantities, n_vertices)."""
    return np.vstack(
        [
            load_surf_data(
                ses_dir / f"quantity-{q}_stat-effect_size_hemi-{hemi}.gii"
            )
            for q in quantities
        ]
    )


def _make_aligner(method, hemi, mesh, labels) -> GroupAlignment:
    if method == "spectral_ot":
        part = mesh.parts[HEMIS[hemi]]
        evecs = get_laplacian_embedding(
            InMemoryMesh(coordinates=part.coordinates, faces=part.faces),
            k=N_EIGENMODES,
        )
        method = SpectralOT(
            evecs=evecs,
            reg=SPECTRAL_REG,
            device=SPECTRAL_DEVICE,
            alpha=SPECTRAL_ALPHA,
            backend="geomloss",
        )
        labels = None
    return GroupAlignment(
        method=method, labels=labels, scale_template=True, n_jobs=N_JOBS
    )


# %%
mesh = load_fsaverage(MESH_NAME)["pial"]
destrieux = fetch_atlas_surf_destrieux()  # fsaverage5 parcels
labels = {"L": destrieux["map_left"], "R": destrieux["map_right"]}

# subject -> (session to learn the alignment on, session to align)
train_test = {}
for subject_dir in sorted(OUTPUT_ROOT.glob("sub-*")):
    sessions = _sessions(subject_dir)
    if len(sessions) < 2:
        print(f"Skipping {subject_dir.name}: sessions {sessions}")
        continue
    train_test[subject_dir.name] = (
        sessions[ALIGNMENT_SESSION],
        sessions[1 - ALIGNMENT_SESSION],
    )

first_test_dir = next(iter(train_test.values()))[1]
quantities = sorted(
    f.name.removeprefix("quantity-").split("_stat-")[0]
    for f in first_test_dir.glob("quantity-*_stat-effect_size_hemi-L.gii")
)

X_train, X_test = {}, {}
for hemi in HEMIS:
    X_train[hemi] = {
        s: _load_session(train, quantities, hemi)
        for s, (train, _) in train_test.items()
    }
    X_test[hemi] = {
        s: _load_session(test, quantities, hemi)
        for s, (_, test) in train_test.items()
    }

# %%
for method in METHODS:
    aligned = {}
    for hemi in HEMIS:
        aligner = _make_aligner(method, hemi, mesh, labels[hemi])
        aligner.fit(X_train[hemi], y="template")
        aligned[hemi] = aligner.transform(X_test[hemi])

    for subject, (_, test_dir) in train_test.items():
        out_dir = test_dir.with_name(f"{test_dir.name}_aligned-{method}")
        out_dir.mkdir(exist_ok=True)
        for i, q in enumerate(quantities):
            data = {h: aligned[hemi][subject][i] for hemi, h in HEMIS.items()}
            SurfaceImage(mesh=mesh, data=data).data.to_filename(
                out_dir / f"quantity-{q}_stat-effect_size.gii"
            )
