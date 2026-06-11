#!/bin/bash
#SBATCH --job-name=pibic_rl
#SBATCH --array=0-59%10          
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --partition=short-simple
#SBATCH --time=02:00:00
#SBATCH --output=logs/ddpg_%A_%a.txt
#SBATCH --error=logs/ddpg_%A_%a_err.txt

module load Python/3.10
source $HOME/ProjetoPibic-1/.venv/bin/activate

mkdir -p logs
cd $HOME/ProjetoPibic-1
python -m environment.script_final --config_id $SLURM_ARRAY_TASK_ID   