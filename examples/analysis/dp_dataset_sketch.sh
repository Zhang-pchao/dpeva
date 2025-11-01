#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 1000:00:00
#SBATCH --gpus 3090:1
#SBATCH --job-name=tsne
#SBATCH --mem=50GB

cd $SLURM_SUBMIT_DIR

module purge
module load conda
source activate /home/pengchao/app/deepmodeling/deepmd-kit/deepmd-kit-v2.2.11

python dp_dataset_sketch.py \
        --dataset /home/pengchao/bubble_ion/TiO/dataset/tio2_dpdataset /home/pengchao/bubble_ion/NaCl/nacl_dpdataset /home/pengchao/bubble_ion/SCAN_H2O_H3O_OH_N2_Nanobubble/train_dataset \
        --model /home/pengchao/bubble_ion/TiO/dp/train/001/frozen_model.pb \
        --output ./sketch_map_output_multi_dataset_tsne \
        --batch-size 1000 \
        --sample-count 6 \
        --random-state 13 \
        --gpu 0 \
        --method tsne \
        --perplexity 20