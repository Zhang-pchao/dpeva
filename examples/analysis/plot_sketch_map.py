"""Create a sketch map of training data using descriptors from a DP model."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from ase import Atoms
from ase.io import write
from deepmd.infer.deep_pot import DeepPot
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from .common import evaluate_descriptors, load_deepmd_dataset


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a PCA-based sketch map of a DeepMD training set using the "
            "descriptors from a frozen Deep Potential model."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the DeepMD dataset directory (contains set.* folders).",
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Path to the frozen_model.pb file used for inference.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory used to store the generated sketch map and samples.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Optional head name if the graph exposes multiple outputs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames evaluated in parallel when computing descriptors.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=4,
        help="Number of representative structures to export as XYZ files.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for the PCA algorithm.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate all results even if they already exist in the output directory.",
    )
    return parser.parse_args()


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        logging.warning(
            "Output directory %s is not empty. Existing files may be reused.", path
        )


def compute_pca_embedding(data: np.ndarray, random_state: int) -> np.ndarray:
    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(data)


def export_samples(
    dataset_coords: np.ndarray,
    dataset_cells: np.ndarray,
    atom_symbols: Sequence[str],
    atom_types: Sequence[int],
    frame_indices: Sequence[Tuple[str, int]],
    destination: Path,
) -> None:
    """Write representative structures to ``destination`` as XYZ files."""

    destination.mkdir(parents=True, exist_ok=True)
    symbols = [atom_symbols[idx] for idx in atom_types]

    for label, frame_index in frame_indices:
        atoms = Atoms(
            symbols=symbols,
            positions=dataset_coords[frame_index],
            cell=dataset_cells[frame_index],
            pbc=True,
        )
        file_path = destination / f"sample_{label}_{frame_index:05d}.xyz"
        write(file_path, atoms)
        logging.info("Exported %s", file_path)


def select_representative_frames(embedding: np.ndarray, count: int) -> list[tuple[str, int]]:
    """Select frames from extreme regions of the embedding."""

    indices = {
        "pc1_min": int(np.argmin(embedding[:, 0])),
        "pc1_max": int(np.argmax(embedding[:, 0])),
        "pc2_min": int(np.argmin(embedding[:, 1])),
        "pc2_max": int(np.argmax(embedding[:, 1])),
    }

    selected: list[tuple[str, int]] = list(indices.items())

    remaining = count - len(selected)
    if remaining > 0:
        additional = np.linspace(0, embedding.shape[0] - 1, remaining, dtype=int)
        for idx, frame in enumerate(additional):
            selected.append((f"extra{idx}", int(frame)))

    seen = set()
    unique_selection = []
    for label, frame_index in selected:
        if frame_index in seen:
            continue
        seen.add(frame_index)
        unique_selection.append((label, frame_index))
    return unique_selection[:count]


def create_sketch_map(
    dataset_dir: Path,
    model_path: Path,
    output_dir: Path,
    head: str | None,
    batch_size: int,
    sample_count: int,
    random_state: int,
    overwrite: bool,
) -> None:
    logging.info("Loading dataset from %s", dataset_dir)
    dataset = load_deepmd_dataset(dataset_dir)

    logging.info("Loading model graph from %s", model_path)
    model = DeepPot(str(model_path), head=head)

    descriptor_cache = output_dir / "descriptor_frames.npy"
    if descriptor_cache.exists() and not overwrite:
        logging.info("Reusing cached descriptors from %s", descriptor_cache)
        descriptor_frames = np.load(descriptor_cache)
    else:
        logging.info("Evaluating descriptors (batch size: %d)", batch_size)
        descriptor_frames = evaluate_descriptors(dataset, model, batch_size=batch_size)
        np.save(descriptor_cache, descriptor_frames)
        logging.info("Descriptor tensor saved to %s", descriptor_cache)

    mean_descriptor = descriptor_frames.mean(axis=1)
    energy_per_atom = dataset.energies / dataset.n_atoms

    logging.info("Performing PCA to obtain the sketch map")
    embedding = compute_pca_embedding(mean_descriptor, random_state=random_state)

    df = pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "energy_per_atom": energy_per_atom,
            "frame_index": np.arange(dataset.n_frames),
        }
    )

    csv_path = output_dir / "sketch_map_points.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Stored PCA embedding coordinates in %s", csv_path)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df,
        x="pc1",
        y="pc2",
        hue="energy_per_atom",
        palette="viridis",
        s=60,
        alpha=0.8,
    )
    plt.title("Sketch map of training structures")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Energy per atom", loc="best")
    plt.tight_layout()

    figure_path = output_dir / "sketch_map.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    logging.info("Sketch map saved to %s", figure_path)

    sample_dir = output_dir / "structures"
    selection = select_representative_frames(embedding, sample_count)
    export_samples(
        dataset.coords,
        dataset.cells,
        dataset.type_symbols,
        dataset.atom_types,
        selection,
        sample_dir,
    )

    selection_path = output_dir / "selected_frames.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json_dump = {label: int(index) for label, index in selection}
        json.dump(json_dump, handle, indent=2)
    logging.info("Sample metadata stored in %s", selection_path)


if __name__ == "__main__":
    args = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    prepare_output_directory(args.output, args.overwrite)
    create_sketch_map(
        dataset_dir=args.dataset,
        model_path=args.model,
        output_dir=args.output,
        head=args.head,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        random_state=args.random_state,
        overwrite=args.overwrite,
    )
