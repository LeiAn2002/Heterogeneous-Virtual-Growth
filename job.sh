#!/bin/sh
#SBATCH --partition=IllinoisComputes
#SBATCH --account=leian2-ic
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128G


module load anaconda3/2024.10
eval "$(conda shell.bash hook)"
conda activate virtualgrowth
python3 scripts/virtual_growth_script.py
python3 scripts/generate_mesh_script.py
python3 scripts/homogenize_script.py