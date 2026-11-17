# Reward Function Design for Portfolio Optimization Using Reinforcement Learning

📄 Paper: [`paper/RL_Portfolio_CEC_2027.pdf`](paper/RL_Portfolio_CEC_2027.pdf) (manuscript in preparation for IEEE CEC 2027)

This repository contains the code from an undergraduate research project (PIBIC/CAPES, CIn/UFPE, Aug 2025 – Aug 2026) on portfolio optimization with deep reinforcement learning (DRL) on Brazilian stocks (B3).
It compares five reward functions (log-return, Huang, Sortino, Omega, Calmar) across three Stable-Baselines3 agents (DDPG, SAC, PPO).
Each agent rebalances daily, monthly or quarterly and observes technical indicators plus an optional macroeconomic context vector.
Agents are evaluated with an expanding-window walk-forward protocol against Equal-Weight and Markowitz (MVO) benchmarks.

## Method

**MDP** (`environment/environment.py`, `PortfolioEnv`, Gymnasium):

- **State:** per-asset technical features, current portfolio weights, the macro context vector (only if `use_context=True`), and each asset's 30-day return.
- **Action:** a vector in `[0, 1]^n_assets`, normalized to sum to 1 (long-only). An all-zero action falls back to equal weights.
- **Transition:** the chosen weights are held for `step_size` trading days and daily returns are compounded over that period.
- **Training episodes:** 252 decisions, starting at a random date.
- **Evaluation:** one continuous pass over the test period.
- **Normalization:** observations and rewards go through `VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10)`.

**Rewards.** Each reward is computed over a 30-day window of the portfolio's daily returns, with a risk-free rate of 8% a.a. (0.08/252 per day), and is clipped to `[-10, 10]`.

| `reward_type` | Definition |
|---|---|
| `log-retorno` | `log(1 + R)`, where `R` is the compounded return over the holding period |
| `huang-return` | `0.6·R + 0.3·Sharpe − 0.1·volatility − 0.05·turnover` |
| `sortino` | Annualized Sortino ratio |
| `omega` | Omega ratio − 1 |
| `calmar` | Annualized mean return / max drawdown |

**Algorithms** (`environment/walk_forward_validation.py`, `build_model`):

| | Settings |
|---|---|
| All three | learning rate 1e-4; actor `[256, 256, 128]`; critic/value `[256, 128]`; ReLU |
| DDPG | γ = 0.98, batch 256, Gaussian action noise σ = 0.1 |
| SAC | γ = 0.99, batch 256, τ = 0.001 |
| PPO | γ = 0.99, `n_steps` 1024, batch 64, 15 epochs, `ent_coef` 0.05 |

Training runs for 300,000 timesteps by default (`WF_TIMESTEPS`).

**Rebalancing frequency.** Set with the `WF_STEP` environment variable: `1` is daily, `21` is about monthly and `63` is about quarterly.

**Features** (`src/feature_engineering/feature_engineering.py`):

- Per-asset indicators: SMA(20) trend, RSI(14), Bollinger Bands(20, 2) and OBV % change, each z-scored over a rolling 252-day window.
- Macro context (`src/pre_processing/macro.py`): Focus Report expectations for IPCA, Selic and GDP (via `python-bcb`), plus the Ibovespa and VIX (via `yfinance`).

**Walk-forward validation.** The pipeline uses an expanding anchor with 18 folds. Each fold trains on 2000 through the year before its test year and tests on one year, from 2007 to 2024. Each test year also gets 30 warm-up trading days from the previous year. The pipeline runs in four steps:

1. **Stage 1:** train every `{DDPG, PPO, SAC} × 5 rewards × seeds` combination *with* context.
2. **Selection** (`--mode selecionar`): for each fold, pick the best `(algo, reward)` by Sharpe ratio on the training period, averaged over seeds, skipping degenerate (constant-weight) policies. The winners are saved to `winners.json`.
3. **Stage 2:** retrain each fold's winner *without* context, as a paired ablation.
4. **Evaluation** (`--mode avaliar`): compare the winner with context, the winner without context, Equal-Weight and MVO in each test year.

> **Note:** the paper reports a **single train/test split** (2000–2020 training, 2021–2026 test; see its Section IV and its stated limitations), not this walk-forward pipeline. That split is not yet included in this repository; see *Results*.

## Repository structure

```
ProjetoPibic/
├── paper/RL_Portfolio_CEC_2027.pdf   # manuscript (in preparation)
├── Pibic_Tratamento.ipynb            # data collection (yfinance) + EDA; selects the top-50 most liquid tickers
├── FinalTop50.csv                    # quarterly OHLCV + sector for those 50 tickers (notebook output)
├── requirements.txt
├── src/
│   ├── pre_processing/
│   │   ├── cleaning.py               # data/raw/csv_ohlcv → data/processed/data_cleaned (IQR winsorization, keeps assets covering 2000–2024)
│   │   ├── macro.py                  # Focus (BCB) + ^BVSP/^VIX → data/external/context.csv
│   │   └── data_loader.py            # aligns features/macro/prices, drops tickers with NaNs, writes master_*_{train,val,test}.csv
│   └── feature_engineering/
│       └── feature_engineering.py    # technical indicators → data/processado_features
├── environment/
│   ├── environment.py                # PortfolioEnv (Gymnasium) + reward functions
│   ├── train.py                      # shared helpers (pre_processamento, seeding, RewardLoggerCallback) + earlier DDPG training routines
│   ├── evaluate.py                   # metrics (Sharpe, drawdown, ...), Equal-Weight and MVO (PyPortfolioOpt) baselines
│   ├── script_final.py               # earlier single-split grid search over algorithms/steps (pre-walk-forward)
│   ├── walk_forward_validation.py    # walk-forward pipeline (train / select / ablation / evaluate)
│   └── walk_forward_compare.py       # per fold: best DDPG vs PPO vs SAC on the test year, plus benchmarks
├── submit_walkforward.sh             # submits the 4 walk-forward steps as chained SLURM jobs
├── wf_worker.sh                      # SLURM worker called by submit_walkforward.sh
└── metrics/walkforward/              # results of the daily (step=1) walk-forward run
    ├── winners.json                  # selected (algo, reward) per fold
    ├── walkforward_selecao_treino.csv
    └── walkforward_metrics.csv
```

Running the pipeline generates `data/`, `environment/models/`, `plots/`, `tensorboard/` and `logs/`. When the step is not 1, output directories get a `_step{N}` suffix.

## Installation

The cluster worker uses **Python 3.10** (`module load Python/3.10` in `wf_worker.sh`).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pandas_ta          # imported by feature_engineering.py but not listed in requirements.txt
```

`requirements.txt` does not pin versions.

## How to run

All commands are run from the repository root.

### 1. Data

`cleaning.py` expects daily OHLCV CSVs (one per ticker, with a `Date` index) in `data/raw/csv_ohlcv/`. The notebook downloads data with `yfinance` for the ticker list in `coleta.csv.csv`, which is not included.

```bash
python src/pre_processing/cleaning.py
python src/feature_engineering/feature_engineering.py
python src/pre_processing/macro.py                  # writes data/external/context.csv (create data/external/ first)
mkdir -p data/processed/data_train_val
python src/pre_processing/data_loader.py            # writes master_{features,macro,prices}_{train,val,test}.csv
```

### 2. Walk-forward, locally

The pipeline reads these environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `WF_STAGE` | `1` | `1` = with context, `2` = ablation without context |
| `WF_SEEDS` | `30,42,123` | Comma-separated seeds |
| `WF_STEP` | `1` | Rebalancing step: `1`, `21` or `63` |
| `WF_N_ENVS` | `8` | Parallel environments (use `1` on a laptop) |
| `WF_TIMESTEPS` | `300000` | Training timesteps |
| `WF_DEVICE` | `auto` | Torch device (PPO always runs on CPU) |

```bash
export WF_SEEDS=30 WF_STEP=1 WF_N_ENVS=1

WF_STAGE=1 python -m environment.walk_forward_validation --list    # list config_ids
WF_STAGE=1 python -m environment.walk_forward_validation --all     # stage 1 (or --config_id <i>)
python -m environment.walk_forward_validation --mode selecionar    # → metrics/walkforward/winners.json
WF_STAGE=2 python -m environment.walk_forward_validation --all     # stage 2: ablation without context
python -m environment.walk_forward_validation --mode avaliar       # → metrics/walkforward/walkforward_metrics.csv

python -m environment.walk_forward_compare                         # optional: best per algorithm
```

### 3. Walk-forward on SLURM

On the login node:

```bash
bash submit_walkforward.sh [seeds] [step]
# e.g.
bash submit_walkforward.sh 30          # daily
bash submit_walkforward.sh 30 21       # ~monthly
bash submit_walkforward.sh 30 63       # ~quarterly
bash submit_walkforward.sh 42,123 21   # more seeds, monthly
```

This submits stage 1 (job array) → selection → stage 2 (job array) → evaluation, chained with `--dependency`. You can raise the array throttle with `WF_THROTTLE` (default 20).

Before submitting, edit `wf_worker.sh` and `submit_walkforward.sh` for your cluster:

- `wf_worker.sh` assumes the repository and its `.venv` are at `$HOME/ProjetoPibic-1`.
- `submit_walkforward.sh` uses partition `short-simple`.

## Results

- **Walk-forward outputs:** [`metrics/walkforward/`](metrics/walkforward/) has the per-fold winners (`winners.json`), the training-period selection table and the test-period metrics, with and without context, for Equal-Weight and MVO. Each table reports Total/Annual Return, Volatility, Sharpe, Max Drawdown and Total Operations.
- **Paper results:** for the single-split results, analysis and discussion, see the [paper](paper/RL_Portfolio_CEC_2027.pdf).

## Citation

```bibtex
@unpublished{batista2026reward,
  title  = {Reward Function Design for Portfolio Optimization Using Reinforcement Learning},
  author = {Batista, Bernardo Siqueira and Sampaio, Maria Paula Perazzo and de Almeida Filho, Adiel Teixeira},
  note   = {Manuscript in preparation for the IEEE Congress on Evolutionary Computation (CEC 2027)},
  year   = {2026}
}
```

## Authors and acknowledgments

- **Bernardo Siqueira Batista**
- **Maria Paula Perazzo Sampaio**
- **Adiel Teixeira de Almeida Filho**

All three are at the Centro de Informática (CIn), Universidade Federal de Pernambuco (UFPE), Recife, Brazil.

This work was supported by the PIBIC/CAPES undergraduate research program. Training experiments ran on CIn/UFPE infrastructure, including the Apuana cluster.
