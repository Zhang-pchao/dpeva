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
import os
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@dataclass
class DatasetBundle:
    """Container for arrays and metadata required during analysis."""

    systems: List[dpdata.LabeledSystem]  # List of individual systems
    coords_list: List[np.ndarray]  # List of coordinate arrays
    cells_list: List[np.ndarray]  # List of cell arrays
    energies: np.ndarray  # Concatenated energies (1D array)
    n_atoms_list: List[int]  # Number of atoms for each frame
    atom_types_list: List[List[int]]  # Atom types for each system
    type_symbols_list: List[List[str]]  # Type symbols for each system
    system_configs: List[str]  # System configuration for each frame (e.g., "64H128O65Ti32N")
    source_paths: List[str]  # Source path for each frame
    frame_to_dataset: List[int]  # Maps global frame index to dataset index

    @property
    def n_frames(self) -> int:
        """Total number of frames contained in the dataset."""
        return len(self.energies)
    
    def get_frame_data(self, frame_idx: int) -> tuple:
        """Get coordinates, cell, atom_types, and type_symbols for a specific frame."""
        dataset_idx = self.frame_to_dataset[frame_idx]
        local_idx = frame_idx - sum(len(self.coords_list[i]) for i in range(dataset_idx))
        
        return (
            self.coords_list[dataset_idx][local_idx],
            self.cells_list[dataset_idx][local_idx],
            self.atom_types_list[dataset_idx],
            self.type_symbols_list[dataset_idx]
        )


# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------

def configure_gpu_environment(gpu_id: str = "0") -> None:
    """Configure environment variables for optimal GPU usage with DeepMD-kit v2."""
    
    # Enable GPU for TensorFlow
    if gpu_id:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        logging.info("Set CUDA_VISIBLE_DEVICES to: %s", gpu_id)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logging.info("GPU disabled, using CPU only")
    
    # Set threading for better performance
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('TF_INTRA_OP_PARALLELISM_THREADS', '1')
    os.environ.setdefault('TF_INTER_OP_PARALLELISM_THREADS', '1')
    
    # Try to check TensorFlow GPU availability
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logging.info("TensorFlow detected %d GPU(s): %s", len(gpus), gpus)
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info("Enabled memory growth for GPU: %s", gpu)
                except RuntimeError as e:
                    logging.warning("Could not set memory growth for GPU: %s", e)
        else:
            logging.warning("No GPU detected by TensorFlow. Running on CPU.")
    except ImportError:
        logging.warning("TensorFlow not available for GPU check.")
    except Exception as e:
        logging.warning("Error checking GPU availability: %s", e)


# ---------------------------------------------------------------------------
# Dataset and model helpers
# ---------------------------------------------------------------------------

def get_system_config_name(system: dpdata.LabeledSystem) -> str:
    """Generate a system configuration name like '64H128O65Ti32N' with periodic table order.
    Skip elements with count 0.
    """
    atom_names = system.get_atom_names()
    atom_numbs = system.get_atom_numbs()
    
    # Define periodic table order for common elements
    periodic_order = ['H', 'C', 'N', 'O', 'Na', 'Cl', 'Ti']
    
    # Create dict for lookup
    name_to_count = dict(zip(atom_names, atom_numbs))
    
    # Sort by periodic table order, skip count=0
    sorted_pairs = []
    for elem in periodic_order:
        if elem in name_to_count and name_to_count[elem] > 0:
            sorted_pairs.append((name_to_count[elem], elem))
    
    # Add any elements not in the periodic_order list (at the end), skip count=0
    for elem, count in zip(atom_names, atom_numbs):
        if elem not in periodic_order and count > 0:
            sorted_pairs.append((count, elem))
    
    config_str = "".join(f"{count}{elem}" for count, elem in sorted_pairs)
    return config_str


def find_deepmd_datasets(root_paths: List[Path]) -> List[Path]:
    """Find all DeepMD dataset directories (containing set.* folders) in the given root paths."""
    dataset_dirs = []
    
    for root_path in root_paths:
        root_path = Path(root_path)
        if not root_path.exists():
            logging.warning("Path does not exist: %s", root_path)
            continue
            
        # Check if root_path itself is a dataset directory
        if any(root_path.glob("set.*")):
            dataset_dirs.append(root_path)
            logging.info("Found dataset in: %s", root_path)
        
        # Search subdirectories for dataset directories
        for subdir in root_path.rglob("*"):
            if subdir.is_dir() and any(subdir.glob("set.*")):
                dataset_dirs.append(subdir)
                logging.info("Found dataset in: %s", subdir)
    
    return dataset_dirs


def load_deepmd_dataset(dataset_path: Path | str) -> DatasetBundle:
    """Load a DeepMD dataset stored in numpy format."""

    dataset = dpdata.LabeledSystem(str(dataset_path), fmt="deepmd/npy")
    coords = np.asarray(dataset.data["coords"], dtype=np.float64)
    cells = np.asarray(dataset.data["cells"], dtype=np.float64)
    energies = np.asarray(dataset.data["energies"], dtype=np.float64)

    atom_types = list(map(int, dataset.get_atom_types()))
    type_symbols = list(dataset.get_atom_names())
    
    # Generate system configuration name
    config_name = get_system_config_name(dataset)
    n_frames = coords.shape[0]
    n_atoms = coords.shape[1]
    
    system_configs = [config_name] * n_frames
    source_paths = [str(dataset_path)] * n_frames
    n_atoms_list = [n_atoms] * n_frames
    frame_to_dataset = [0] * n_frames

    return DatasetBundle(
        systems=[dataset],
        coords_list=[coords],
        cells_list=[cells],
        energies=energies,
        n_atoms_list=n_atoms_list,
        atom_types_list=[atom_types],
        type_symbols_list=[type_symbols],
        system_configs=system_configs,
        source_paths=source_paths,
        frame_to_dataset=frame_to_dataset,
    )


def load_multiple_datasets(dataset_paths: List[Path]) -> DatasetBundle:
    """Load and concatenate multiple DeepMD datasets (supports different atom numbers)."""
    
    if not dataset_paths:
        raise ValueError("No dataset paths provided")
    
    all_systems = []
    all_coords_list = []
    all_cells_list = []
    all_energies_list = []
    all_n_atoms_list = []
    all_atom_types_list = []
    all_type_symbols_list = []
    all_configs = []
    all_sources = []
    all_frame_to_dataset = []
    
    dataset_idx = 0
    for path in dataset_paths:
        try:
            dataset = dpdata.LabeledSystem(str(path), fmt="deepmd/npy")
            coords = np.asarray(dataset.data["coords"], dtype=np.float64)
            cells = np.asarray(dataset.data["cells"], dtype=np.float64)
            energies = np.asarray(dataset.data["energies"], dtype=np.float64)
            
            atom_types = list(map(int, dataset.get_atom_types()))
            type_symbols = list(dataset.get_atom_names())
            config_name = get_system_config_name(dataset)
            
            n_frames = coords.shape[0]
            n_atoms = coords.shape[1]
            
            all_systems.append(dataset)
            all_coords_list.append(coords)
            all_cells_list.append(cells)
            all_energies_list.append(energies)
            all_n_atoms_list.extend([n_atoms] * n_frames)
            all_atom_types_list.append(atom_types)
            all_type_symbols_list.append(type_symbols)
            all_configs.extend([config_name] * n_frames)
            all_sources.extend([str(path)] * n_frames)
            all_frame_to_dataset.extend([dataset_idx] * n_frames)
            
            logging.info("Loaded %d frames from %s (config: %s, %d atoms)", 
                        n_frames, path, config_name, n_atoms)
            
            dataset_idx += 1
        except Exception as e:
            logging.error("Failed to load dataset from %s: %s", path, e)
    
    if not all_systems:
        raise ValueError("No datasets were successfully loaded")
    
    # Concatenate energies (1D array)
    all_energies = np.concatenate(all_energies_list, axis=0)
    
    return DatasetBundle(
        systems=all_systems,
        coords_list=all_coords_list,
        cells_list=all_cells_list,
        energies=all_energies,
        n_atoms_list=all_n_atoms_list,
        atom_types_list=all_atom_types_list,
        type_symbols_list=all_type_symbols_list,
        system_configs=all_configs,
        source_paths=all_sources,
        frame_to_dataset=all_frame_to_dataset,
    )


def build_model_type_indices(type_symbols: List[str], atom_types: List[int], model: DeepPot) -> List[int]:
    """Map dataset atom types to the ordering expected by the model graph."""

    model_type_map = list(model.get_type_map())
    index_lookup = {symbol: model_type_map.index(symbol) for symbol in type_symbols}
    return [index_lookup[type_symbols[idx]] for idx in atom_types]


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
    """Evaluate mean descriptor for every frame in the dataset (supports multiple systems with different atom numbers)."""

    all_mean_descriptors = []
    
    # Process each dataset separately
    for dataset_idx, (coords, cells, atom_types, type_symbols) in enumerate(
        zip(dataset.coords_list, dataset.cells_list, dataset.atom_types_list, dataset.type_symbols_list)
    ):
        mapped_types = build_model_type_indices(type_symbols, atom_types, model)
        n_frames = coords.shape[0]
        
        logging.info("Processing dataset %d/%d with %d frames", 
                    dataset_idx + 1, len(dataset.systems), n_frames)
        
        descriptor_frames: list[np.ndarray] = []
        for start, stop in batched(n_frames, batch_size):
            cells_batch = cells[start:stop]
            if cells_batch.ndim == 3 and cells_batch.shape[-2:] == (3, 3):
                pass
            else:
                cells_batch = cells_batch.reshape(-1, 3, 3)

            descriptor = model.eval_descriptor(
                coords[start:stop],
                cells_batch,
                mapped_types,
            )
            # Compute mean over atoms for each frame
            # descriptor shape: (batch, n_atoms, descriptor_dim) -> (batch, descriptor_dim)
            mean_descriptor = descriptor.mean(axis=1)
            descriptor_frames.append(mean_descriptor)
        
        dataset_mean_descriptors = np.concatenate(descriptor_frames, axis=0)
        all_mean_descriptors.append(dataset_mean_descriptors)
        logging.info("Dataset %d descriptors shape: %s", dataset_idx + 1, dataset_mean_descriptors.shape)
    
    # Concatenate all mean descriptors (now they have the same shape)
    return np.concatenate(all_mean_descriptors, axis=0)


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

def compute_pca_embedding(data: np.ndarray, random_state: int) -> np.ndarray:
    """Reduce descriptor features to two principal components."""

    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(data)


def compute_tsne_embedding(data: np.ndarray, random_state: int, perplexity: int = 30) -> np.ndarray:
    """Reduce descriptor features to two dimensions using t-SNE."""
    
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, 
                   max_iter=1000, verbose=1)
    return reducer.fit_transform(data)


def downsample_by_config(
    indices: np.ndarray, 
    system_configs: List[str], 
    max_per_config: int = 200
) -> np.ndarray:
    """Downsample frames for each system configuration."""
    
    config_to_indices = {}
    for idx in indices:
        config = system_configs[idx]
        if config not in config_to_indices:
            config_to_indices[config] = []
        config_to_indices[config].append(idx)
    
    selected_indices = []
    for config, config_indices in config_to_indices.items():
        if len(config_indices) > max_per_config:
            # Uniformly sample max_per_config frames
            step = len(config_indices) / max_per_config
            sampled = [config_indices[int(i * step)] for i in range(max_per_config)]
            selected_indices.extend(sampled)
            logging.info("Downsampled %s from %d to %d frames", 
                        config, len(config_indices), len(sampled))
        else:
            selected_indices.extend(config_indices)
    
    return np.array(selected_indices, dtype=int)


def select_representative_per_config(
    embedding: np.ndarray, 
    system_configs: List[str]
) -> dict[str, int]:
    """Select one representative frame per system configuration from PCA embedding."""
    
    config_to_indices = {}
    for idx, config in enumerate(system_configs):
        if config not in config_to_indices:
            config_to_indices[config] = []
        config_to_indices[config].append(idx)
    
    representatives = {}
    for config, indices in config_to_indices.items():
        if len(indices) == 1:
            representatives[config] = indices[0]
        else:
            # Select the frame closest to the centroid in PCA space
            config_embedding = embedding[indices]
            centroid = config_embedding.mean(axis=0)
            distances = np.linalg.norm(config_embedding - centroid, axis=1)
            closest_idx = indices[np.argmin(distances)]
            representatives[config] = closest_idx
    
    return representatives


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
    dataset: DatasetBundle,
    frame_indices: Sequence[Tuple[str, int]],
    destination: Path,
) -> None:
    """Write representative structures to ``destination`` as XYZ files."""

    destination.mkdir(parents=True, exist_ok=True)

    for label, frame_index in frame_indices:
        coords, cell, atom_types, type_symbols = dataset.get_frame_data(frame_index)
        symbols = [type_symbols[idx] for idx in atom_types]
        
        atoms = Atoms(
            symbols=symbols,
            positions=coords,
            cell=cell,
            pbc=True,
        )
        file_path = destination / f"sample_{label}_{frame_index:05d}.xyz"
        write(file_path, atoms)
        logging.info("Exported %s", file_path)


def export_config_representatives(
    dataset: DatasetBundle,
    config_to_frame: dict[str, int],
    destination: Path,
) -> None:
    """Export one representative structure per system configuration as extended XYZ."""
    
    destination.mkdir(parents=True, exist_ok=True)
    
    for config_name, frame_idx in config_to_frame.items():
        coords, cell, atom_types, type_symbols = dataset.get_frame_data(frame_idx)
        symbols = [type_symbols[idx] for idx in atom_types]
        
        atoms = Atoms(
            symbols=symbols,
            positions=coords,
            cell=cell,
            pbc=True,
        )
        # Add config name as info
        atoms.info['config'] = config_name
        atoms.info['frame_index'] = frame_idx
        atoms.info['source'] = dataset.source_paths[frame_idx]
        
        file_path = destination / f"representative_{config_name}.xyz"
        write(file_path, atoms, format='extxyz')
        logging.info("Exported representative for %s: %s", config_name, file_path)


def save_metadata(output_dir: Path, metadata: dict[str, object]) -> None:
    """Persist metadata about the analysis for reproducibility."""

    metadata_path = output_dir / "descriptor_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def find_annotation_positions(
    embedding: np.ndarray,
    config_to_frame: dict[str, int],
    min_distance: float = 0.1
) -> dict[str, tuple[float, float, str]]:
    """Find non-overlapping positions for text annotations.
    
    Returns dict mapping config to (x, y, alignment)
    """
    positions = {}
    occupied = []
    
    # Calculate bounding box
    x_range = embedding[:, 0].max() - embedding[:, 0].min()
    y_range = embedding[:, 1].max() - embedding[:, 1].min()
    
    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        
        # Try different offsets to avoid overlap
        offsets = [
            (0.05 * x_range, 0.05 * y_range, 'left'),
            (-0.05 * x_range, 0.05 * y_range, 'right'),
            (0.05 * x_range, -0.05 * y_range, 'left'),
            (-0.05 * x_range, -0.05 * y_range, 'right'),
            (0.08 * x_range, 0, 'left'),
            (-0.08 * x_range, 0, 'right'),
        ]
        
        best_pos = None
        best_dist = 0
        
        for dx, dy, align in offsets:
            text_x = point_x + dx
            text_y = point_y + dy
            
            # Check distance to all occupied positions
            min_dist = float('inf')
            for occ_x, occ_y in occupied:
                dist = np.sqrt((text_x - occ_x)**2 + (text_y - occ_y)**2)
                min_dist = min(min_dist, dist)
            
            if min_dist > best_dist:
                best_dist = min_dist
                best_pos = (text_x, text_y, align)
        
        if best_pos:
            positions[config] = best_pos
            occupied.append((best_pos[0], best_pos[1]))
    
    return positions


def render_structure_ovito_style(
    atoms: Atoms,
    output_path: Path,
    size: tuple[int, int] = (300, 300),
    camera_pos: str = 'xy'
) -> np.ndarray:
    """Render atomic structure in OVITO style and save as high-quality image."""
    from io import BytesIO
    from PIL import Image
    import matplotlib.patches as mpatches
    
    # Element colors (OVITO-like)
    element_colors = {
        'H': '#FFFFFF',   # White
        'C': '#909090',   # Gray
        'N': '#3050F8',   # Blue
        'O': '#FF0D0D',   # Red
        'Na': '#AB5CF2',  # Purple
        'Cl': '#1FF01F',  # Green
        'Ti': '#BFC2C7',  # Light gray
    }
    
    # Element radii (smaller to avoid overlap)
    element_radii = {
        'H': 0.2, 'C': 0.45, 'N': 0.4, 'O': 0.38,
        'Na': 0.55, 'Cl': 0.6, 'Ti': 0.5
    }
    
    fig = plt.figure(figsize=(size[0]/100, size[1]/100), dpi=200)  # Higher DPI for better quality
    ax = fig.add_subplot(111, projection='3d')
    
    # Get positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Center the structure
    center = positions.mean(axis=0)
    positions = positions - center
    
    # Plot atoms (smaller size)
    for pos, symbol in zip(positions, symbols):
        color = element_colors.get(symbol, '#CCCCCC')
        radius = element_radii.get(symbol, 0.4)
        ax.scatter(pos[0], pos[1], pos[2], 
                  c=color, s=radius*200, alpha=0.95, edgecolors='black', linewidth=0.3)
    
    # Set view angle
    if camera_pos == 'xy':
        ax.view_init(elev=20, azim=45)
    elif camera_pos == 'xz':
        ax.view_init(elev=60, azim=45)
    else:
        ax.view_init(elev=30, azim=30)
    
    # Remove all axes, labels, and frames
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    
    # Make background completely transparent/white, no borders
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Hide the axis lines completely
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    
    # Equal aspect ratio
    max_range = np.array([positions[:,0].max()-positions[:,0].min(),
                         positions[:,1].max()-positions[:,1].min(),
                         positions[:,2].max()-positions[:,2].min()]).max() / 2.0
    mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
    mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
    mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout(pad=0)
    
    # Save to file with higher DPI
    plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.02, 
                facecolor='none', edgecolor='none')
    
    # Also return as array
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', pad_inches=0.02,
                edgecolor='none',facecolor='none')
    plt.close(fig)
    
    buf.seek(0)
    img = Image.open(buf)
    
    return np.array(img)


def group_configs_by_chemistry(configs: List[str]) -> dict[str, int]:
    """Group configs by chemical composition and assign sequential numbers."""
    
    # Extract elements from each config
    def get_elements(config: str) -> frozenset:
        elements = set()
        import re
        # Match patterns like "64H" or "128O"
        for match in re.finditer(r'(\d+)([A-Z][a-z]?)', config):
            elements.add(match.group(2))
        return frozenset(elements)
    
    # Group configs by their element sets
    element_groups = {}
    for config in configs:
        elem_set = get_elements(config)
        elem_key = tuple(sorted(elem_set))
        if elem_key not in element_groups:
            element_groups[elem_key] = []
        element_groups[elem_key].append(config)
    
    # Sort groups and assign numbers
    config_to_number = {}
    number = 1
    
    # Sort groups by their element composition
    for elem_key in sorted(element_groups.keys()):
        # Within each group, sort configs alphabetically
        for config in sorted(element_groups[elem_key]):
            config_to_number[config] = number
            number += 1
    
    return config_to_number


def create_sketch_map_with_numbers(
    embedding: np.ndarray,
    energies_per_atom: np.ndarray,
    config_to_frame: dict[str, int],
    output_path: Path,
    figsize: tuple[float, float] = (12, 8)
) -> None:
    """Create sketch map with numbered labels (chemically similar configs get close numbers)."""
    
    # Group configs by chemistry and assign numbers
    config_to_number = group_configs_by_chemistry(list(config_to_frame.keys()))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=energies_per_atom,
        cmap='viridis',
        s=30,
        alpha=0.6,
        edgecolors='none'
    )
    
    # Add numbered annotations
    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        number = config_to_number[config]
        
        # Highlight the representative point
        ax.scatter([point_x], [point_y], s=90, 
                  facecolors='none', edgecolors='red', linewidths=2.5, alpha=0.8, zorder=5)
        
        # Add number label on the point
        ax.text(point_x, point_y, str(number), 
               fontsize=9, fontweight='bold', ha='center', va='center',
               color='white', zorder=6,
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='red', alpha=0.8, edgecolor='darkred'))
    
    # Add smaller colorbar at bottom left, horizontal
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, fraction=0.03, aspect=15)
    cbar.set_label('E/atom (eV)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Position colorbar at bottom left (smaller and shorter)
    cbar.ax.set_position([0.08, 0.04, 0.27, 0.015])
    
    # Remove axes, labels, and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='none', bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved sketch map with numbers to %s", output_path)
    
    # Save number mapping to JSON
    number_to_config = {v: k for k, v in config_to_number.items()}
    mapping_path = output_path.parent / "number_to_config_mapping.json"
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(number_to_config, f, indent=2, sort_keys=True)
    logging.info("Saved number-to-config mapping to %s", mapping_path)


def create_sketch_map_with_labels(
    embedding: np.ndarray,
    energies_per_atom: np.ndarray,
    config_to_frame: dict[str, int],
    output_path: Path,
    figsize: tuple[float, float] = (12, 8)
) -> None:
    """Create sketch map with text labels and arrows pointing to representative configs."""
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=energies_per_atom,
        cmap='viridis',
        s=30,
        alpha=0.6,
        edgecolors='none'
    )
    
    # Find annotation positions
    annotation_positions = find_annotation_positions(embedding, config_to_frame)
    
    # Add annotations with arrows
    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        
        if config in annotation_positions:
            text_x, text_y, align = annotation_positions[config]
            
            # Highlight the representative point
            ax.scatter([point_x], [point_y], s=100, 
                      facecolors='none', edgecolors='red', linewidths=2, zorder=5)
            
            # Add arrow and text (larger font)
            ax.annotate(
                config,
                xy=(point_x, point_y),
                xytext=(text_x, text_y),
                fontsize=12,
                ha=align,
                va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray'),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.3',
                    color='red',
                    lw=2
                ),
                zorder=6
            )
    
    # Add smaller colorbar at bottom left, horizontal
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, fraction=0.03, aspect=15)
    cbar.set_label('E/atom (eV)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Position colorbar at bottom left (smaller and shorter)
    cbar.ax.set_position([0.08, 0.04, 0.27, 0.015])
    
    # Remove axes, labels, and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='none',bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved sketch map with labels to %s", output_path)


def create_sketch_map_with_structures(
    embedding: np.ndarray,
    energies_per_atom: np.ndarray,
    dataset: DatasetBundle,
    config_to_frame: dict[str, int],
    output_path: Path,
    figsize: tuple[float, float] = (12, 10)
) -> None:
    """Create sketch map with OVITO-style structure thumbnails instead of text labels."""
    
    # Create directory for structure images
    struct_img_dir = output_path.parent / "structure_images"
    struct_img_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=energies_per_atom,
        cmap='viridis',
        s=30,
        alpha=0.6,
        edgecolors='none'
    )
    
    # Calculate plot range for thumbnail sizing
    x_range = embedding[:, 0].max() - embedding[:, 0].min()
    y_range = embedding[:, 1].max() - embedding[:, 1].min()
    
    # Add structure thumbnails
    for config, frame_idx in config_to_frame.items():
        point_x, point_y = embedding[frame_idx]
        
        # Highlight the representative point
        ax.scatter([point_x], [point_y], s=100, 
                  facecolors='none', edgecolors='red', linewidths=2, zorder=5)
        
        # Get frame data
        coords, cell, atom_types, type_symbols = dataset.get_frame_data(frame_idx)
        symbols = [type_symbols[idx] for idx in atom_types]
        
        # Create atoms object
        atoms = Atoms(
            symbols=symbols,
            positions=coords,
            cell=cell,
            pbc=True,
        )
        
        try:
            # Render structure in OVITO style and save
            struct_img_path = struct_img_dir / f"structure_{config}.png"
            img_array = render_structure_ovito_style(atoms, struct_img_path, size=(150, 150))
            logging.info("Saved structure image for %s: %s", config, struct_img_path)
            
            # Load and insert the saved image
            from PIL import Image
            img = Image.open(struct_img_path)
            
            # Position thumbnail offset from point
            img_x = point_x + 0.05 * x_range
            img_y = point_y + 0.05 * y_range
            
            # Convert to data coordinates for extent (smaller size)
            img_width = 0.12 * x_range
            img_height = 0.12 * y_range
            
            extent = [img_x, img_x + img_width, img_y, img_y + img_height]
            ax.imshow(img, extent=extent, zorder=7, aspect='auto')
            
            # Add arrow
            ax.annotate(
                '',
                xy=(point_x, point_y),
                xytext=(img_x + img_width/2, img_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                zorder=6
            )
        except Exception as e:
            logging.warning("Could not render thumbnail for %s: %s", config, e)
    
    # Add smaller colorbar at bottom left, horizontal
    cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                       pad=0.05, fraction=0.03, aspect=15)
    cbar.set_label('E/atom (eV)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Position colorbar at bottom left (smaller and shorter)
    cbar.ax.set_position([0.08, 0.04, 0.27, 0.015])
    
    # Remove axes, labels, and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='none',bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved sketch map with structures to %s", output_path)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def build_argument_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser for the script."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze DeepMD training sets with a frozen Deep Potential model "
            "and create a PCA sketch map colored by energy per atom."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs='+',
        help="One or more paths to search for DeepMD datasets (will search subdirectories for set.* folders).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to the frozen_model.pb file produced by deepmd-kit.",
    )
    parser.add_argument(
        "--output",
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
        default=100,
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
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use (default: 0). Set to empty string to use CPU.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tsne",
        choices=["pca", "tsne"],
        help="Dimensionality reduction method: 'pca' or 'tsne' (default: tsne).",
    )
    parser.add_argument(
        "--perplexity",
        type=int,
        default=30,
        help="Perplexity for t-SNE (default: 30). Typical values are between 5 and 50.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Configure GPU environment before loading the model
    configure_gpu_environment(args.gpu)

    args.output.mkdir(parents=True, exist_ok=True)
    descriptor_cache = args.output / "descriptor_frames.npy"
    
    # Find all dataset directories from the provided paths
    logging.info("Searching for datasets in: %s", args.dataset)
    dataset_paths = find_deepmd_datasets(args.dataset)
    
    if not dataset_paths:
        raise ValueError("No DeepMD datasets found in the provided paths")
    
    logging.info("Found %d dataset(s)", len(dataset_paths))
    
    # Load and merge all datasets
    dataset = load_multiple_datasets(dataset_paths)
    logging.info("Total frames loaded: %d", dataset.n_frames)
    
    # Report system configurations
    unique_configs = sorted(set(dataset.system_configs))
    config_counts = {cfg: dataset.system_configs.count(cfg) for cfg in unique_configs}
    logging.info("System configurations found:")
    for cfg, count in config_counts.items():
        logging.info("  %s: %d frames", cfg, count)

    if args.head is not None:
        logging.warning("The --head parameter is not supported by the current DeepPot version and will be ignored.")

    logging.info("Loading model graph from %s", args.model)
    model = DeepPot(str(args.model))

    # Downsample if needed
    all_indices = np.arange(dataset.n_frames)
    selected_indices = downsample_by_config(all_indices, dataset.system_configs, max_per_config=200)
    logging.info("Using %d frames after downsampling", len(selected_indices))

    if descriptor_cache.exists() and not args.overwrite:
        logging.info("Reusing cached descriptors from %s", descriptor_cache)
        mean_descriptors = np.load(descriptor_cache)
    else:
        logging.info("Evaluating descriptors in batches of %d", args.batch_size)
        mean_descriptors = evaluate_descriptors(dataset, model, batch_size=args.batch_size)
        np.save(descriptor_cache, mean_descriptors)
        logging.info("Mean descriptors stored in %s (shape: %s)", descriptor_cache, mean_descriptors.shape)

    # Apply downsampling to descriptors and other data
    mean_descriptors_selected = mean_descriptors[selected_indices]
    energies_selected = dataset.energies[selected_indices]
    n_atoms_selected = np.array([dataset.n_atoms_list[i] for i in selected_indices])
    system_configs_selected = [dataset.system_configs[i] for i in selected_indices]
    
    energies_per_atom = energies_selected / n_atoms_selected

    # Choose dimensionality reduction method
    if args.method == 'tsne':
        logging.info("Computing t-SNE embedding for the sketch map (perplexity=%d)", args.perplexity)
        embedding = compute_tsne_embedding(mean_descriptors_selected, 
                                          random_state=args.random_state,
                                          perplexity=args.perplexity)
    else:
        logging.info("Computing PCA embedding for the sketch map")
        embedding = compute_pca_embedding(mean_descriptors_selected, random_state=args.random_state)

    # Select representative frame per configuration
    config_to_frame_local = select_representative_per_config(embedding, system_configs_selected)
    # Map back to original indices
    config_to_frame_global = {cfg: selected_indices[idx] for cfg, idx in config_to_frame_local.items()}
    
    logging.info("Selected representative frames:")
    for cfg, idx in config_to_frame_global.items():
        logging.info("  %s: frame %d", cfg, idx)

    # Save data
    df = pd.DataFrame(
        {
            "pc1": embedding[:, 0],
            "pc2": embedding[:, 1],
            "energy_per_atom": energies_per_atom,
            "original_frame_index": selected_indices,
            "system_config": system_configs_selected,
        }
    )

    csv_path = args.output / "sketch_map_points.csv"
    df.to_csv(csv_path, index=False)
    logging.info("Stored embedding coordinates in %s", csv_path)

    # Create sketch map with text labels
    figure_path = args.output / "sketch_map.png"
    create_sketch_map_with_labels(
        embedding, 
        energies_per_atom, 
        config_to_frame_local, 
        figure_path
    )

    # Create sketch map with numbered labels
    figure_num_path = args.output / "sketch_map_num.png"
    create_sketch_map_with_numbers(
        embedding,
        energies_per_atom,
        config_to_frame_local,
        figure_num_path
    )

    # # Create sketch map with structure thumbnails (commented out - not needed by default)
    # figure_geo_path = args.output / "sketch_map_geo.png"
    # create_sketch_map_with_structures(
    #     embedding,
    #     energies_per_atom,
    #     dataset,
    #     config_to_frame_global,
    #     figure_geo_path
    # )
    
    # Generate individual structure images (OVITO style)
    struct_img_dir = args.output / "structure_images"
    struct_img_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Generating OVITO-style structure images...")
    for config, frame_idx in config_to_frame_global.items():
        coords, cell, atom_types, type_symbols = dataset.get_frame_data(frame_idx)
        symbols = [type_symbols[idx] for idx in atom_types]
        
        atoms = Atoms(
            symbols=symbols,
            positions=coords,
            cell=cell,
            pbc=True,
        )
        
        struct_img_path = struct_img_dir / f"structure_{config}.png"
        render_structure_ovito_style(atoms, struct_img_path, size=(300, 300))
        logging.info("Saved structure image: %s", struct_img_path)

    # Export representative structures per configuration
    config_dir = args.output / "config_representatives"
    export_config_representatives(dataset, config_to_frame_global, config_dir)

    # Also export samples from extreme regions (old behavior)
    sample_dir = args.output / "structures"
    selection = select_representative_frames(embedding, args.sample_count)
    # Map back to global indices
    selection_global = [(label, selected_indices[idx]) for label, idx in selection]
    export_samples(dataset, selection_global, sample_dir)

    selection_path = args.output / "selected_frames.json"
    with selection_path.open("w", encoding="utf-8") as handle:
        json.dump({label: int(index) for label, index in selection_global}, handle, indent=2)
    logging.info("Recorded representative frames in %s", selection_path)

    metadata = {
        "dataset_paths": [str(p) for p in dataset_paths],
        "model": str(args.model.resolve()),
        "n_frames_total": dataset.n_frames,
        "n_frames_after_downsample": len(selected_indices),
        "n_atoms_range": [min(dataset.n_atoms_list), max(dataset.n_atoms_list)],
        "mean_descriptor_shape": mean_descriptors.shape,
        "system_configurations": config_counts,
        "head": args.head,
        "batch_size": args.batch_size,
        "sample_count": args.sample_count,
        "random_state": args.random_state,
    }
    save_metadata(args.output, metadata)
    logging.info("Metadata written to descriptor_metadata.json")


if __name__ == "__main__":
    main()
