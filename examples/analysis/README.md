# Analysis Scripts Overview

This directory collects helper utilities for exploring Deep Potential (DP) datasets, inspecting descriptor quality, and generating visualization artifacts. The table below summarizes how each script fits into the workflow and the most important entry points to look at before reusing or modifying them.

| File | Key responsibilities |
| --- | --- |
| `common.py` | Houses shared utilities such as the `DatasetBundle` helper for loading DeepMD datasets, atom-type bookkeeping helpers, and convenience wrappers for evaluating frozen DP models. Every other script imports this module to avoid duplicating data-loading logic. |
| `generate_descriptors.py` | CLI tool that reads a DeepMD dataset together with a `frozen_model.pb` checkpoint, evaluates atomic descriptors in batch, and stores the raw tensors plus frame-averaged descriptors, per-frame energies, and metadata in an output directory. Start from the `main()` function when wiring it into automated pipelines. |
| `plot_sketch_map.py` | Given precomputed descriptors, produces a 2D sketch map with energies encoded in the color scale. It performs dimensionality reduction (via PCA by default), renders matplotlib scatter plots, and exports representative structures (XYZ format) along with the selection log. The `SketchMapPlotter` class centralizes the plotting logic. |
| `dp_dataset_sketch.py` | Orchestrates a full end-to-end analysis: loading one or more datasets, configuring GPU acceleration, evaluating descriptors, running dimensionality reduction, plotting, and exporting sampled structures. The `SketchWorkflow` class ties together the helper functions under clearly named methods (e.g., `prepare_dataset`, `compute_descriptors`, `embed_descriptors`, `plot_results`). **For dimensionality reduction, prefer the t-SNE pathway** (`--tsne` flag and the `run_tsne` helper); the PCA implementation is known to be unstable for some datasets, so it is **not recommended** until the numerical issues are resolved. |
| `dp_dataset_sketch.sh` | Sample Slurm submission script showing how to call `dp_dataset_sketch.py` in a preconfigured deep-learning environment, including GPU selection, memory limits, and key command-line arguments. |

## Recommended dependencies

All scripts assume an environment with `deepmd-kit`, `dpdata`, `numpy`, `ase`, `matplotlib`, `seaborn`, `pandas`, and `scikit-learn`. Install any optional extras required by your workload (for example, CUDA-enabled PyTorch if you extend the GPU utilities) and customize command-line parameters or job configurations as needed.
