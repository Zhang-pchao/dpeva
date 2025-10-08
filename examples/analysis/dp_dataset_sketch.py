"""Deep Potential dataset sketch-map workflow packaged in a single script.

This script combines all functionality required to inspect a DeepMD training
set with a frozen Deep Potential model. It performs the following steps:

1. Load a DeepMD-format dataset (``set.*`` folders with ``coord.npy``,
   ``force.npy``, etc.).
2. Load a ``frozen_model.pb`` graph produced by ``deepmd-kit``.
3. Evaluate the descriptor tensor for each frame in the dataset.
4. Compute a PCA-based sketch map using the mean descriptor per frame.
5. Export a scatter plot colored by energy per atom and store representative
   atomic structures from extreme regions of the map.

Example usage
-------------
.. code-block:: bash

    python dp_dataset_sketch.py \
        /path/to/dataset \
        /path/to/frozen_model.pb \
        ./sketch_map_output \
        --batch-size 16 \
        --sample-count 6 \
        --random-state 13

Dependencies: ``deepmd-kit``, ``dpdata``, ``numpy``, ``ase``, ``matplotlib``,
``seaborn``, and ``scikit-learn``.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import dpdata
import numpy as np
import pandas as pd
import seaborn as sns
from ase import Atoms
from ase.io import write
from deepmd.infer.deep_pot import DeepPot
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


@dataclass
class DatasetBundle:
    """Container for arrays and metadata required during analysis."""

    system: dpdata.LabeledSystem
    coords: np.ndarray
    cells: np.ndarray
    energies: np.ndarray
    atom_types: List[int]
    type_symbols: List[str]

    @property
    def n_frames(self) -> int:
        """Total number of frames contained in the dataset."""

        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        """Number of atoms per frame."""

        return int(self.coords.shape[1])


# ---------------------------------------------------------------------------
# Dataset and model helpers
# ---------------------------------------------------------------------------

def load_deepmd_dataset(dataset_path: Path | str) -> DatasetBundle:
    """Load a DeepMD dataset stored in numpy format."""

    dataset = dpdata.LabeledSystem(str(dataset_path), fmt="deepmd/npy")
    coords = np.asarray(dataset.data["coords"], dtype=np.float64)
    cells = np.asarray(dataset.data["cells"], dtype=np.float64)
    energies = np.asarray(dataset.data["energies"], dtype=np.float64)

    atom_types = list(map(int, dataset.get_atom_types()))
    type_symbols = list(dataset.get_atom_names())

    return DatasetBundle(
        system=dataset,
        coords=coords,
        cells=cells,
        energies=energies,
        atom_types=atom_types,
        type_symbols=type_symbols,
    )


def build_model_type_indices(dataset: DatasetBundle, model: DeepPot) -> List[int]:
    """Map dataset atom types to the ordering expected by the model graph."""

    model_type_map = list(model.get_type_map())
    index_lookup = {symbol: model_type_map.index(symbol) for symbol in dataset.type_symbols}
    return [index_lookup[dataset.type_symbols[idx]] for idx in dataset.atom_types]


def batched(total_frames: int, batch_size: int) -> Iterable[tuple[int, int]]:
    """Yield (start, stop) indices that split a range into batches."""

    for start in range(0, total_frames, batch_size):
        stop = min(total_frames, start + batch_size)
        yield start, stop


def evaluate_descriptors(
    dataset: DatasetBundle,
    model: DeepPot,
    batch_size: int,
) -> np.ndarray:
    """Evaluate descriptor tensors for every frame in the dataset."""

    mapped_types = build_model_type_indices(dataset, model)

    descriptor_frames: list[np.ndarray] = []
    for start, stop in batched(dataset.n_frames, batch_size):
        cells = dataset.cells[start:stop]
        if cells.ndim == 3 and cells.shape[-2:] == (3, 3):
            cells_batch = cells
        else:
            cells_batch = cells.reshape(-1, 3, 3)

        descriptor = model.eval_descriptor(
            dataset.coords[start:stop],
            cells_batch,
            mapped_types,
        )
        descriptor_frames.append(descriptor)

    return np.concatenate(descriptor_frames, axis=0)


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def compute_pca_embedding(data: np.ndarray, random_state: int) -> np.ndarray:
    """Reduce descriptor features to two principal components."""

    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(data)


def select_representative_frames(embedding: np.ndarray, count: int) -> list[tuple[str, int]]:
    """Pick frames from extreme regions of the PCA embedding."""

    indices = {
        "pc1_min": int(np.argmin(embedding[:, 0])),
        "pc1_max": int(np.argmax(embedding[:, 0])),
        "pc2_min": int(np.argmin(embedding[:, 1])),
        "pc2_max": int(np.argmax(embedding[:, 1])),
    }

    selection: list[tuple[str, int]] = list(indices.items())
    remaining = max(count - len(selection), 0)

    if remaining > 0:
        additional = np.linspace(0, embedding.shape[0] - 1, remaining, dtype=int)
        for idx, frame in enumerate(additional):
            selection.append((f"extra{idx}", int(frame)))

    unique_frames: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, frame_idx in selection:
        if frame_idx in seen:
            continue
        seen.add(frame_idx)
        unique_frames.append((label, frame_idx))
    return unique_frames[:count]


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


def save_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    """Persist metadata about the analysis for reproducibility."""

    metadata_path = output_dir / "descriptor_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the script."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze a DeepMD training set with a frozen Deep Potential model "
            "and create a PCA sketch map colored by energy per atom."
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
        help="Path to the frozen_model.pb file produced by deepmd-kit.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Directory where descriptors, plots, and samples will be written.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Optional name of the output head when the graph exposes multiple heads.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames evaluated together when calling the model.",
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
        help="Random seed supplied to the PCA algorithm.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate descriptors and plots even if they already exist.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output.mkdir(parents=True, exist_ok=True)
    descriptor_cache = args.output / "descriptor_frames.npy"

    logging.info("Loading dataset from %s", args.dataset)
    dataset = load_deepmd_dataset(args.dataset)

    logging.info("Loading model graph from %s", args.model)
    model = DeepPot(str(args.model), head=args.head)

    if descriptor_cache.exists() and not args.overwrite:
        logging.info("Reusing cached descriptors from %s", descriptor_cache)
        descriptor_frames = np.load(descriptor_cache)
    else:
        logging.info("Evaluating descriptors in batches of %d", args.batch_size)
        descriptor_frames = evaluate_descriptors(dataset, model, batch_size=args.batch_size)
        np.save(descriptor_cache, descriptor_frames)
        logging.info("Descriptor tensor stored in %s", descriptor_cache)

    mean_descriptor = descriptor_frames.mean(axis=1)
    energies_per_atom = dataset.energies / dataset.n_atoms

    logging.info("Computing PCA embedding for the sketch map")
    embedding = compute_pca_embedding(mean_descriptor, random_state=args.random_state)

    df = pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "energy_per_atom": energies_per_atom,
            "frame_index": np.arange(dataset.n_frames),
        }
    )

    csv_path = args.output / "sketch_map_points.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Stored embedding coordinates in %s", csv_path)

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

    figure_path = args.output / "sketch_map.png"
    plt.savefig(figure_path, dpi=300)
    plt.close()
    logging.info("Sketch map saved to %s", figure_path)

    sample_dir = args.output / "structures"
    selection = select_representative_frames(embedding, args.sample_count)
    export_samples(
        dataset.coords,
        dataset.cells,
        dataset.type_symbols,
        dataset.atom_types,
        selection,
        sample_dir,
    )

    selection_path = args.output / "selected_frames.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump({label: int(index) for label, index in selection}, handle, indent=2)
    logging.info("Recorded representative frames in %s", selection_path)

    metadata = {
        "dataset": str(args.dataset.resolve()),
        "model": str(args.model.resolve()),
        "n_frames": dataset.n_frames,
        "n_atoms": dataset.n_atoms,
        "descriptor_shape": descriptor_frames.shape,
        "head": args.head,
        "batch_size": args.batch_size,
        "sample_count": args.sample_count,
        "random_state": args.random_state,
    }
    save_metadata(args.output, metadata)
    logging.info("Metadata written to descriptor_metadata.json")


if __name__ == "__main__":
    main()
