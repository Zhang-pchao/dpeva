"""Utility helpers for analyzing Deep Potential datasets with dpeva."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import dpdata
from deepmd.infer.deep_pot import DeepPot


@dataclass
class DatasetBundle:
    """Container for the raw arrays used in descriptor analysis."""

    system: dpdata.LabeledSystem
    coords: np.ndarray
    cells: np.ndarray
    energies: np.ndarray
    atom_types: List[int]
    type_symbols: List[str]

    @property
    def n_frames(self) -> int:
        """Return the number of frames in the dataset."""

        return int(self.coords.shape[0])

    @property
    def n_atoms(self) -> int:
        """Return the number of atoms per frame."""

        return int(self.coords.shape[1])


def load_deepmd_dataset(dataset_path: Path | str) -> DatasetBundle:
    """Load a DeepMD training set from ``dataset_path``.

    Parameters
    ----------
    dataset_path:
        Path to the directory containing ``set.*`` folders and ``type.raw``.

    Returns
    -------
    DatasetBundle
        Object containing the raw arrays and metadata extracted from the dataset.
    """

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
    """Map dataset atom types to the ordering expected by the model."""

    model_type_map = list(model.get_type_map())
    index_lookup = {symbol: model_type_map.index(symbol) for symbol in dataset.type_symbols}
    mapped = [index_lookup[dataset.type_symbols[idx]] for idx in dataset.atom_types]
    return mapped


def batched(iterable: np.ndarray, batch_size: int) -> Iterable[tuple[int, int]]:
    """Yield (start, stop) indices that split an array into batches."""

    total = int(iterable.shape[0])
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        yield start, stop


def evaluate_descriptors(
    dataset: DatasetBundle,
    model: DeepPot,
    batch_size: int = 32,
) -> np.ndarray:
    """Evaluate descriptors for the provided dataset using ``model``.

    Parameters
    ----------
    dataset:
        Loaded DeepMD dataset.
    model:
        Frozen Deep Potential model.
    batch_size:
        Number of frames evaluated per batch. ``32`` is a reasonable default
        for most CPU-bound environments.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_frames, n_atoms, descriptor_dim)`` containing the
        descriptor of every atom in every frame.
    """

    mapped_types = build_model_type_indices(dataset, model)

    descriptor_frames: list[np.ndarray] = []
    for start, stop in batched(dataset.coords, batch_size):
        cells = dataset.cells[start:stop]
        if cells.ndim == 3 and cells.shape[-2:] == (3, 3):
            cells_batch = cells
        else:
            cells_batch = cells.reshape(-1, 3, 3)

        desc = model.eval_descriptor(
            dataset.coords[start:stop],
            cells_batch,
            mapped_types,
        )
        descriptor_frames.append(desc)

    return np.concatenate(descriptor_frames, axis=0)
