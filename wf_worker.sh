#!/bin/bash
# Worker generico do walk-forward. NAO submeta direto -- use submit_walkforward.sh.
# Se WF_MODE estiver setado (selecionar|avaliar) roda esse modo; senao treina o
# config_id vindo do SLURM_ARRAY_TASK_ID. WF_STAGE/WF_SEEDS vem do --export.

module load Python/3.10
source $HOME/ProjetoPibic-1/.venv/bin/activate
cd $HOME/ProjetoPibic-1
mkdir -p logs

export WF_N_ENVS=${WF_N_ENVS:-8}
export WF_TIMESTEPS=${WF_TIMESTEPS:-300000}

if [ -n "$WF_MODE" ]; then
    python -m environment.walf_forward_validation --mode "$WF_MODE"
else
    python -m environment.walf_forward_validation --config_id "$SLURM_ARRAY_TASK_ID"
fi
