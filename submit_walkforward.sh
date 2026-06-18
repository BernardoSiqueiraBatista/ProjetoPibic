#!/bin/bash
# ======================================================================
# RODE UMA VEZ NO LOGIN NODE:   bash submit_walkforward.sh [seeds] [step]
#   ex.: bash submit_walkforward.sh 30           (1 seed, step=1 diario)
#        bash submit_walkforward.sh 30 21        (1 seed, step=21 ~mensal)
#        bash submit_walkforward.sh 30 63        (1 seed, step=63 ~trimestral)
#        bash submit_walkforward.sh 42,123 21    (fase B de seeds, mensal)
#
# Submete os 4 passos do walk-forward ENCADEADOS por dependencia do SLURM:
#   1) estagio 1  : treina candidatos COM contexto      (job-array)
#   2) selecionar : elege o vencedor por fold no treino  (1 no)
#   3) estagio 2  : ablacao -- vencedor SEM contexto      (job-array)
#   4) avaliar    : ablacao pareada + benchmarks no teste (1 no)
# Cada passo so comeca quando o anterior termina (afterany = roda mesmo se
# alguns treinos falharem; selecao/avaliacao ignoram modelos ausentes).
#
# step=1 escreve em metrics/plots/models/walkforward (caminhos antigos).
# step!=1 escreve em ..._step{N} -- NAO colide com o pipeline step=1.
# ======================================================================
set -euo pipefail

SEEDS="${1:-30}"
STEP="${2:-1}"
THR="${WF_THROTTLE:-20}"   # max de tasks simultaneas por array (suba se a fila permitir)
TAG="s${STEP}"             # sufixo p/ nomes de job e logs (separa horizontes)

# tamanhos das grades (dependem do nº de seeds desta submissao)
nseeds=$(awk -F, '{print NF}' <<< "$SEEDS")
n1=$((18 * 3 * 5 * nseeds - 1))   # estagio 1: folds x algos x rewards x seeds
n2=$((18 * nseeds - 1))           # estagio 2: folds (1 vencedor) x seeds

COMMON="--ntasks=1 --mem=8G -c 8 --partition=short-simple --time=04:00:00"
mkdir -p logs

echo "seeds=$SEEDS | step=$STEP | estagio1=0-$n1 | estagio2=0-$n2 | throttle=%$THR"

j1=$(sbatch --parsable $COMMON --job-name=wf1_${TAG} --array=0-${n1}%${THR} \
     --output=logs/wf_${TAG}_s1_%A_%a.txt --error=logs/wf_${TAG}_s1_%A_%a_err.txt \
     --export=ALL,WF_STAGE=1,WF_SEEDS="$SEEDS",WF_STEP="$STEP" wf_worker.sh)
echo "1) estagio1 (com contexto)  job=$j1"

j2=$(sbatch --parsable $COMMON --job-name=wfsel_${TAG} --dependency=afterany:$j1 \
     --output=logs/wf_${TAG}_sel_%A.txt --error=logs/wf_${TAG}_sel_%A_err.txt \
     --export=ALL,WF_MODE=selecionar,WF_SEEDS="$SEEDS",WF_STEP="$STEP" wf_worker.sh)
echo "2) selecionar               job=$j2  (apos $j1)"

j3=$(sbatch --parsable $COMMON --job-name=wf2_${TAG} --array=0-${n2}%${THR} --dependency=afterok:$j2 \
     --output=logs/wf_${TAG}_s2_%A_%a.txt --error=logs/wf_${TAG}_s2_%A_%a_err.txt \
     --export=ALL,WF_STAGE=2,WF_SEEDS="$SEEDS",WF_STEP="$STEP" wf_worker.sh)
echo "3) estagio2 (ablacao)       job=$j3  (apos $j2)"

j4=$(sbatch --parsable $COMMON --job-name=wfav_${TAG} --dependency=afterok:$j3 \
     --output=logs/wf_${TAG}_av_%A.txt --error=logs/wf_${TAG}_av_%A_err.txt \
     --export=ALL,WF_MODE=avaliar,WF_SEEDS="$SEEDS",WF_STEP="$STEP" wf_worker.sh)
echo "4) avaliar                  job=$j4  (apos $j3)"

ns="walkforward"; [ "$STEP" != "1" ] && ns="walkforward_step${STEP}"
echo
echo "tudo submetido. acompanhe com:  squeue -u \$USER"
echo "resultados ao final em: metrics/${ns}/walkforward_metrics.csv"