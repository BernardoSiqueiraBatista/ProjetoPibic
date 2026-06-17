#!/bin/bash
# ======================================================================
# RODE UMA VEZ NO LOGIN NODE:   bash submit_walkforward.sh [seeds]
#   ex.: bash submit_walkforward.sh 30          (fase A, 1 seed)
#        bash submit_walkforward.sh 42,123      (fase B, treina so as novas)
#
# Submete os 4 passos do walk-forward ENCADEADOS por dependencia do SLURM:
#   1) estagio 1  : treina candidatos COM contexto      (job-array)
#   2) selecionar : elege o vencedor por fold no treino  (1 no)
#   3) estagio 2  : ablacao -- vencedor SEM contexto      (job-array)
#   4) avaliar    : ablacao pareada + benchmarks no teste (1 no)
# Cada passo so comeca quando o anterior termina (afterany = roda mesmo se
# alguns treinos falharem; selecao/avaliacao ignoram modelos ausentes).
# ======================================================================
set -euo pipefail

SEEDS="${1:-30}"
THR="${WF_THROTTLE:-20}"   # max de tasks simultaneas por array (suba se a fila permitir)

# tamanhos das grades (dependem do nº de seeds desta submissao)
nseeds=$(awk -F, '{print NF}' <<< "$SEEDS")
n1=$((18 * 3 * 5 * nseeds - 1))   # estagio 1: folds x algos x rewards x seeds
n2=$((18 * nseeds - 1))           # estagio 2: folds (1 vencedor) x seeds

COMMON="--ntasks=1 --mem=8G -c 8 --partition=short-simple --time=04:00:00"
mkdir -p logs

echo "seeds=$SEEDS | estagio1=0-$n1 | estagio2=0-$n2 | throttle=%$THR"

j1=$(sbatch --parsable $COMMON --job-name=wf_s1 --array=0-${n1}%${THR} \
     --output=logs/wf_s1_%A_%a.txt --error=logs/wf_s1_%A_%a_err.txt \
     --export=ALL,WF_STAGE=1,WF_SEEDS="$SEEDS" wf_worker.sh)
echo "1) estagio1 (com contexto)  job=$j1"

j2=$(sbatch --parsable $COMMON --job-name=wf_sel --dependency=afterany:$j1 \
     --output=logs/wf_sel_%A.txt --error=logs/wf_sel_%A_err.txt \
     --export=ALL,WF_MODE=selecionar,WF_SEEDS="$SEEDS" wf_worker.sh)
echo "2) selecionar               job=$j2  (apos $j1)"

j3=$(sbatch --parsable $COMMON --job-name=wf_s2 --array=0-${n2}%${THR} --dependency=afterok:$j2 \
     --output=logs/wf_s2_%A_%a.txt --error=logs/wf_s2_%A_%a_err.txt \
     --export=ALL,WF_STAGE=2,WF_SEEDS="$SEEDS" wf_worker.sh)
echo "3) estagio2 (ablacao)       job=$j3  (apos $j2)"

j4=$(sbatch --parsable $COMMON --job-name=wf_av --dependency=afterok:$j3 \
     --output=logs/wf_av_%A.txt --error=logs/wf_av_%A_err.txt \
     --export=ALL,WF_MODE=avaliar,WF_SEEDS="$SEEDS" wf_worker.sh)
echo "4) avaliar                  job=$j4  (apos $j3)"

echo
echo "tudo submetido. acompanhe com:  squeue -u \$USER"
echo "resultados ao final em: metrics/walkforward/walkforward_metrics.csv"
