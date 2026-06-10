import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypfopt
from pypfopt import EfficientFrontier, risk_models, expected_returns

from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.environment import PortfolioEnv

base_path = os.path.dirname(os.path.abspath(__file__))
prices_train_path  = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_train.csv")
feat_val_path      = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_val.csv")
macro_val_path     = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_val.csv")
prices_val_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_val.csv")
feat_test_path     = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_test.csv")
macro_test_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_test.csv")
prices_test_path   = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_test.csv")
model_path = os.path.join(base_path, "..", "ddpg_portfolio_final.zip")
model_sac = os.path.join(base_path, "..", "sac_portfolio_final.zip")
model_ppo = os.path.join(base_path, "..", "ppo_portfolio_final.zip")

import random
import torch

def _filter_tickers(df_features, df_prices):
    import json
    tickers_path = os.path.join(base_path, "..", "tickers_grid_search.json")
    if not os.path.exists(tickers_path):
        print("AVISO: tickers_grid_search.json não encontrado — obs space pode divergir do treino.")
        return df_features, df_prices
    with open(tickers_path) as f:
        tickers = json.load(f)
    feat_cols  = [c for c in df_features.columns if c.split('_F_')[0] in tickers]
    price_cols = [t for t in tickers if t in df_prices.columns]
    return df_features[feat_cols], df_prices[price_cols]


def iniciar_semente(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #Isso aqui é para CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics(portfolio_values):
    values = pd.Series(portfolio_values)
    returns = values.pct_change().dropna()

    n_steps = len(returns)
    total_return = values.iloc[-1] - 1.0
    # fator de crescimento composto, anualizado pelo nº de períodos
    annual_return = values.iloc[-1] ** (252 / n_steps) - 1 if n_steps > 0 else 0.0

    volatility = returns.std() * np.sqrt(252)
    risk_free_daily = 0.08 / 252
    sharpe = np.sqrt(252) * (returns.mean() - risk_free_daily) / (returns.std() + 1e-8)

    rolling_max = values.cummax()
    drawdown = (values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "Total Return": float(total_return),
        "Annual Return": float(annual_return),
        "Volatility": float(volatility),
        "Sharpe Ratio": float(sharpe),
        "Max Drawdown": float(max_drawdown),
    }


def count_operations(weights_prev, weights_curr, threshold=1e-4):
    diff = weights_curr - weights_prev
    operations = np.sum(np.abs(diff) > threshold)
    return operations


def compute_operations_metrics(weights_df, threshold=1e-4,step=1):
    diffs = weights_df.diff().iloc[step:]
    buys = (diffs > threshold).sum(axis=1)
    sells = (diffs < -threshold).sum(axis=1)
    ops_per_day = buys + sells
    return {
        "Total Operations": int(ops_per_day.sum()),
        "Avg Daily Operations": float(ops_per_day.mean()),
        "Total Buys": int(buys.sum()),
        "Total Sells": int(sells.sum()),
    }


def equal_weight_baseline(prices):

    #alterar codigo para ignorar primeiros 30 dias
    df_prices = prices.iloc[30:]
    returns = df_prices.pct_change().dropna()

    if len(returns) == 0:
        return pd.Series([1.0])

    equal_returns = returns.mean(axis=1)
    equity = (1 + equal_returns).cumprod()

    equity = pd.concat([pd.Series([1.0]), equity])
    equity.reset_index(drop=True, inplace=True)

    return equity


def mvo_baseline(test_prices, train_prices):

    mu = expected_returns.mean_historical_return(train_prices)
    S = risk_models.sample_cov(train_prices)

    ef = EfficientFrontier(mu, S)

    weights = ef.max_sharpe()

    weights = pd.Series(weights)

    test_prices = test_prices.iloc[30:]

    returns = test_prices.pct_change().dropna()

    portfolio_returns = returns.dot(weights)

    equity = (1 + portfolio_returns).cumprod()

    return equity

def evaluate_model(model_path, df_features, df_prices, df_macro, SEED,reward_type,
                   vec_normalize_path):

    df_prices = df_prices.loc[df_features.index]

    env = PortfolioEnv(df_features, df_macro, df_prices, reward_type)
    env = DummyVecEnv([lambda: env])
    env.seed(SEED)

    env = VecNormalize.load(vec_normalize_path, env)

    print("Obs mean:", env.obs_rms.mean[:5])
    print("Obs var:", env.obs_rms.var[:5])

    env.training = False 

    env.norm_reward = False

    obs = env.reset()

    model = DDPG.load(model_path)

    portfolio_values = [1.0]
    weights_history = []

    done = False

    step_count = 0
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]

        weights_history.append(info["weights"])

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(weights_history, columns=env.envs[0].tickers)
    weights_df.to_csv(os.path.join(base_path, "..", "data", "ddpg_weights_val.csv"), index=False)
    print(weights_df.describe())
    print("\nConcentração média (max weight por dia):")
    print(weights_df.max(axis=1).mean())
    return pd.Series(portfolio_values), weights_df

def evaluate_model_sac(model_path, df_features, df_prices, df_macro, SEED):

    df_prices = df_prices.loc[df_features.index]

    env = PortfolioEnv(df_features, df_macro, df_prices, "log-retorno")
    env = DummyVecEnv([lambda: env])
    env.seed(SEED)
    
    env = VecNormalize.load("vec_normalize_stats.pkl", env)

    env.training = False 

    env.norm_reward = False
    obs = env.reset()

    model = SAC.load(model_path)

    portfolio_values = [1.0]
    weights_history = []

    done = False
    step_count = 0
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]

        weights_history.append(info["weights"])

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(weights_history, columns=env.envs[0].tickers)
    weights_df.to_csv(os.path.join(base_path, "..", "data", "sac_weights_val.csv"), index=False)
    return pd.Series(portfolio_values), weights_df

def evaluate_model_ppo(model_path, df_features, df_prices, df_macro, SEED=42):

    df_prices = df_prices.loc[df_features.index]

    env = PortfolioEnv(df_features, df_macro, df_prices, "log-retorno")
    env = DummyVecEnv([lambda: env])
    env.seed(SEED)
    
    env = VecNormalize.load("vec_normalize_stats.pkl", env)

    env.training = False 

    env.norm_reward = False
    obs = env.reset()

    model = PPO.load(model_path)

    portfolio_values = [1.0]
    weights_history = []

    done = False
    step_count = 0
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]

        weights_history.append(info["weights"])

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(weights_history, columns=env.envs[0].tickers)
    weights_df.to_csv(os.path.join(base_path, "..", "data", "ppo_weights_val.csv"), index=False)
    return pd.Series(portfolio_values), weights_df


def evaluate_convergence(df_features, df_prices, df_macro, seed=42):
    import glob

    models_dir = os.path.join(base_path, "..", "models")
    pkl_file   = os.path.join(base_path, "..", "vec_normalize_stats_macro.pkl")

    pattern = os.path.join(models_dir, "ddpg_portfolio_model_*_steps.zip")
    files   = sorted(glob.glob(pattern))
    steps   = sorted(
        int(os.path.basename(f).replace("ddpg_portfolio_model_", "").replace("_steps.zip", ""))
        for f in files
    )

    if not steps:
        print(f"Nenhum checkpoint ddpg_portfolio_model encontrado em {models_dir}")
        return

    print(f"Checkpoints encontrados: {steps[0]} → {steps[-1]} ({len(steps)} total)")

    sharpes = []
    for step in steps:
        model_file = os.path.join(models_dir, f"ddpg_portfolio_model_{step}_steps.zip")
        try:
            values, _ = evaluate_model(model_file, df_features, df_prices, df_macro, seed, "log-retorno", pkl_file)
            metrics = compute_metrics(values)
            sharpes.append({"step": step, "sharpe": metrics["Sharpe Ratio"]})
            print(f"Step {step:>7}: Sharpe = {metrics['Sharpe Ratio']:.4f}")
        except Exception as e:
            print(f"Erro em step {step}: {e}")

    if not sharpes:
        print("Nenhum resultado para plotar.")
        return

    plots_dir = os.path.join(base_path, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    df_conv = pd.DataFrame(sharpes).sort_values("step")
    plt.figure()
    plt.plot(df_conv["step"], df_conv["sharpe"], marker='o')
    plt.xlabel("Timesteps")
    plt.ylabel("Sharpe Ratio (validação)")
    plt.title("Convergência — Sharpe por checkpoint (DDPG)")
    plt.grid()
    plt.savefig(os.path.join(plots_dir, "convergence.png"), dpi=150)
    plt.show()

def plot_grid_search_per_reward(df_results):
    import matplotlib.pyplot as plt
    import numpy as np

    reward_types = df_results["Reward"].unique()
    n_rewards = len(reward_types)

    fig, axes = plt.subplots(1, n_rewards, figsize=(7 * n_rewards, 5))
    if n_rewards == 1:
        axes = [axes]

    for ax, reward in zip(axes, reward_types):
        subset = df_results[df_results["Reward"] == reward]

        pivot = subset.groupby(["LR", "Gamma"])["Sharpe Ratio"].mean().unstack()

        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=df_results["Sharpe Ratio"].min(),
                       vmax=df_results["Sharpe Ratio"].max())

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(g) for g in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(lr) for lr in pivot.index])
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Learning Rate")
        ax.set_title(f"Grid Search — {reward}")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontweight="bold", fontsize=12)

        plt.colorbar(im, ax=ax, label="Sharpe Ratio")

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/grid_search_per_reward.png", dpi=150)
    plt.show()
    print("Salvo em plots/grid_search_per_reward.png")

def evaluate_grid_search(df_features, df_prices, df_macro):
    learning_rates = [1e-4, 3e-4]

    gammas         = [0.95, 0.98, 0.99]

    seeds          = [30, 42, 123]
    
    reward_types = ["log-retorno", "huang-return", "sortino"]
   

    results = []

    for lr, gamma, seed, reward_type in itertools.product(learning_rates, gammas, seeds, reward_types):
        run_name   = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward_type}"
        model_file = os.path.join(base_path, "..", f"{run_name}_ddpg_portfolio_final")
        pkl_file   = os.path.join(base_path, "..", f"vec_normalize_{run_name}.pkl")

        if not os.path.exists(model_file) or not os.path.exists(pkl_file):
            print(f"Pulando {run_name}: arquivos não encontrados")
            continue

        print(f"Avaliando {run_name}...")
        try:
            portfolio_values, weights_df = evaluate_model(
                model_file, df_features, df_prices, df_macro, seed,  reward_type ,pkl_file
            )
            metrics = compute_metrics(portfolio_values)
            metrics.update(compute_operations_metrics(weights_df))
            metrics.update({"Run": run_name, "LR": lr, "Gamma": gamma, "Seed": seed, "Reward":reward_type})
            results.append(metrics)
        except Exception as e:
            print(f"Erro em {run_name}: {e}")

    if not results:
        print("Nenhum modelo encontrado. Rode train_grid_search() primeiro.")
        return None

    metric_cols = [
        "Sharpe Ratio", "Total Return", "Annual Return",
        "Volatility", "Max Drawdown",
        "Total Operations", "Avg Daily Operations",
    ]
    int_metrics = {"Total Operations", "Total Buys", "Total Sells"}

    df_results = pd.DataFrame(results)

    # ── Individual runs ──────────────────────────────────────────────────────
    cols_display = ["Run", "LR", "Gamma", "Seed", "Reward"] + metric_cols
    print("\n===== GRID SEARCH — RUNS INDIVIDUAIS (val 2016–2019) =====")
    print(
        df_results[cols_display]
        .sort_values("Sharpe Ratio", ascending=False)
        .to_string(index=False)
    )

    # Aqui perguntar a duvida, é melhor ter uma tunagem de hiperparametros geral , ou escolher um
    # set de hiperparametros para cada reward function
    grouped = df_results.groupby(["LR", "Gamma", "Reward"])[metric_cols]
    means   = grouped.mean()
    stds    = grouped.std()

    agg_rows = []
    for (lr, gamma, reward_type) in means.index:
        row = {"LR": lr, "Gamma": gamma, "Reward": reward_type}
        for col in metric_cols:
            m, s = means.loc[(lr, gamma, reward_type), col], stds.loc[(lr, gamma, reward_type), col]
            if col in int_metrics:
                row[col] = f"{m:.0f} ± {s:.0f}"
            else:
                row[col] = f"{m:.4f} ± {s:.4f}"
        row["_sharpe_mean"] = means.loc[(lr, gamma, reward_type), "Sharpe Ratio"]
        agg_rows.append(row)

    df_agg = (
        pd.DataFrame(agg_rows)
        .sort_values("_sharpe_mean", ascending=False)
        .drop(columns=["_sharpe_mean"])
        .reset_index(drop=True)
    )

    print("\n===== MELHOR CONFIGURAÇÃO POR REWARD FUNCTION =====")
    best_per_reward = {}
    for reward in df_results["Reward"].unique():
        subset = df_agg[df_agg["Reward"] == reward].iloc[0]
        best_per_reward[reward] = {
            "LR": subset["LR"],
            "Gamma": subset["Gamma"],
            "Sharpe": subset["Sharpe Ratio"]
        }
        print(f"\n{reward}:")
        print(f"  Melhor LR={subset['LR']}, Gamma={subset['Gamma']}")
        print(f"  Sharpe: {subset['Sharpe Ratio']}")

    plot_grid_search_per_reward(df_results)
    return df_results, df_agg, best_per_reward

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    SEED = 42
    iniciar_semente(SEED)

   
    print("\n===== AVALIAÇÃO EM VALIDAÇÃO (2016–2019) =====")
    df_train_prices = pd.read_csv(prices_train_path, index_col=0, parse_dates=True)
    df_features, df_macro, df_prices = _load_val_data()

    print("Período val:", df_features.index.min().date(), "→", df_features.index.max().date())

   
    df_results, df_agg, best_per_reward = evaluate_grid_search(df_features, df_prices, df_macro)

 
    print("\n===== AVALIAÇÃO EM TESTE (2019–2024) =====")
    df_feat_test   = pd.read_csv(feat_test_path,   index_col=0, parse_dates=True).dropna()
    df_macro_test  = pd.read_csv(macro_test_path,  index_col=0, parse_dates=True).loc[df_feat_test.index]
    df_prices_test = pd.read_csv(prices_test_path, index_col=0, parse_dates=True).loc[df_feat_test.index]
    df_feat_test, df_prices_test = _filter_tickers(df_feat_test, df_prices_test)
    df_macro_test = df_macro_test.loc[df_feat_test.index]

    print("Período teste:", df_feat_test.index.min().date(), "→", df_feat_test.index.max().date())

    # ── AVALIAR MELHOR MODELO POR REWARD NO TESTE ───────────────────────────
    seeds = [30, 42, 123]
    test_results = {}
    all_equity_curves = {}

    for reward_type, best in best_per_reward.items():
        lr, gamma = best["LR"], best["Gamma"]
        all_values = []

        for seed in seeds:
            run_name   = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward_type}"
            model_file = os.path.join(base_path, "..", f"{run_name}_ddpg_portfolio_final")
            pkl_file   = os.path.join(base_path, "..", f"vec_normalize_{run_name}.pkl")

            if not os.path.exists(model_file):
                print(f"Modelo não encontrado: {model_file}")
                continue

            values, _ = evaluate_model(
                model_file, df_feat_test, df_prices_test, df_macro_test,
                seed, reward_type, pkl_file
            )
            all_values.append(values)

        if not all_values:
            continue

        min_len = min(len(v) for v in all_values)
        aligned = [v.iloc[:min_len].reset_index(drop=True) for v in all_values]
        mean_values = pd.concat(aligned, axis=1).mean(axis=1)
        metrics = compute_metrics(mean_values)
        test_results[reward_type] = metrics
        all_equity_curves[reward_type] = mean_values

        print(f"\n--- Teste: {reward_type} (média {len(all_values)} seeds) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # ── BENCHMARKS ──────────────────────────────────────────────────────────
    eq_test_values  = equal_weight_baseline(df_prices_test)
    mvo_test_values = mvo_baseline(df_prices_test, df_train_prices)

    eq_metrics  = compute_metrics(eq_test_values)
    mvo_metrics = compute_metrics(mvo_test_values)

    print("\n--- Teste: Equal Weight ---")
    for k, v in eq_metrics.items():
        print(f"{k}: {v:.4f}")

    # ── PLOTS ───────────────────────────────────────────────────────────────
    os.makedirs("plots", exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"log-retorno": "tab:blue", "huang-return": "tab:orange", "sortino": "tab:green"}
    for reward_type, curve in all_equity_curves.items():
        ax.plot(curve.values, label=f"RL ({reward_type})", color=colors.get(reward_type))

    ax.plot(eq_test_values.values, label="Equal Weight", color="black", linestyle="--")
    ax.set_title("Equity Curve — Teste (2019–2024)")
    ax.set_xlabel("Dias")
    ax.set_ylabel("Valor do Portfólio")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.savefig("plots/equity_curves_test.png", dpi=150)
    plt.show()

def evaluate_convergence_rewards(df_features, df_prices, df_macro,
                                  best_lr=1e-4, best_gamma=0.98,
                                  seeds=[30, 42, 123]):
    import glob

    reward_types  = ["log-retorno", "huang-return", "sortino"]
    reward_labels = {"log-retorno": "Log-Return", "huang-return": "Huang (Combinada)", "sortino": "Sortino"}
    colors        = {"log-retorno": "tab:blue", "huang-return": "tab:orange", "sortino": "tab:green"}

    sample_steps = list(range(25000, 325000, 25000))

    plots_dir = os.path.join(base_path, "..", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for reward_type in reward_types:
        sharpes_per_step = {s: [] for s in sample_steps}

        for seed in seeds:
            run_name  = f"ddpg_lr{best_lr}_g{best_gamma}_s{seed}_r{reward_type}"
            models_dir = os.path.join(base_path, "..", "models", run_name)
            pkl_file   = os.path.join(base_path, "..", f"vec_normalize_{run_name}.pkl")

            if not os.path.exists(pkl_file):
                print(f"pkl não encontrado: {run_name}")
                continue

            for step in sample_steps:
                model_file = os.path.join(models_dir, f"{run_name}_{step}_steps.zip")
                if not os.path.exists(model_file):
                    continue
                try:
                    values, _ = evaluate_model(model_file, df_features, df_prices, df_macro, seed, reward_type, pkl_file)
                    metrics = compute_metrics(values)
                    sharpes_per_step[step].append(metrics["Sharpe Ratio"])
                    print(f"[{reward_type}] seed={seed} step={step}: Sharpe={metrics['Sharpe Ratio']:.4f}")
                except Exception as e:
                    print(f"Erro [{reward_type}] seed={seed} step={step}: {e}")

        steps_ok = sorted([s for s, vals in sharpes_per_step.items() if vals])
        if not steps_ok:
            print(f"Nenhum dado para {reward_type}")
            continue

        means = np.array([np.mean(sharpes_per_step[s]) for s in steps_ok])
        stds  = np.array([np.std(sharpes_per_step[s])  for s in steps_ok])

        plt.plot(steps_ok, means, marker='o', label=reward_labels[reward_type], color=colors[reward_type])
        plt.fill_between(steps_ok, means - stds, means + stds, alpha=0.2, color=colors[reward_type])

    plt.xlabel("Timesteps")
    plt.ylabel("Sharpe Ratio (validação)")
    plt.title(f"Convergência por Função de Recompensa  (lr={best_lr}, γ={best_gamma})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "convergence_reward_comparison.png"), dpi=150)
    plt.show()
    print("Salvo em plots/convergence_reward_comparison.png")


def _load_val_data():
    df_features = pd.read_csv(feat_val_path,   index_col=0, parse_dates=True).dropna()
    df_macro    = pd.read_csv(macro_val_path,  index_col=0, parse_dates=True).loc[df_features.index]
    df_prices   = pd.read_csv(prices_val_path, index_col=0, parse_dates=True).loc[df_features.index]
    df_features, df_prices = _filter_tickers(df_features, df_prices)
    df_macro = df_macro.loc[df_features.index]
    return df_features, df_macro, df_prices


def _load_test_data():
    df_features = pd.read_csv(feat_test_path,   index_col=0, parse_dates=True).dropna()
    df_macro    = pd.read_csv(macro_test_path,  index_col=0, parse_dates=True).loc[df_features.index]
    df_prices   = pd.read_csv(prices_test_path, index_col=0, parse_dates=True).loc[df_features.index]
    df_features, df_prices = _filter_tickers(df_features, df_prices)
    df_macro = df_macro.loc[df_features.index]
    return df_features, df_macro, df_prices


def evaluate_all_rewards(seed: int = 42, lr: float = 1e-4, gamma: float = 0.98):
    """Avalia as 5 reward functions (config única) em validação e teste."""
    iniciar_semente(seed)

    reward_types = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]
    colors = {
        "log-retorno":  "tab:blue",
        "huang-return": "tab:orange",
        "sortino":      "tab:green",
        "omega":        "tab:purple",
        "calmar":       "tab:red",
    }

    df_feat_val,  df_macro_val,  df_prices_val  = _load_val_data()
    df_feat_test, df_macro_test, df_prices_test = _load_test_data()

    df_train_prices = pd.read_csv(prices_train_path, index_col=0, parse_dates=True)
    common_tickers  = df_prices_val.columns.tolist()
    df_train_prices = df_train_prices[[t for t in common_tickers if t in df_train_prices.columns]]

    val_curves,  test_curves  = {}, {}
    val_metrics, test_metrics = {}, {}

    for reward_type in reward_types:
        run_name   = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward_type}"
        model_file = os.path.join(base_path, "..", f"{run_name}_ddpg_portfolio_final")
        pkl_file   = os.path.join(base_path, "..", f"vec_normalize_{run_name}.pkl")

        if not os.path.exists(model_file + ".zip") and not os.path.exists(model_file):
            print(f"Modelo não encontrado: {run_name} — pulando")
            continue

        print(f"\n--- Validação: {reward_type} ---")
        v_val, _ = evaluate_model(model_file, df_feat_val, df_prices_val, df_macro_val,
                                  seed, reward_type, pkl_file)
        val_curves[reward_type]  = v_val
        val_metrics[reward_type] = compute_metrics(v_val)

        print(f"--- Teste: {reward_type} ---")
        v_test, _ = evaluate_model(model_file, df_feat_test, df_prices_test, df_macro_test,
                                   seed, reward_type, pkl_file)
        test_curves[reward_type]  = v_test
        test_metrics[reward_type] = compute_metrics(v_test)

    # Baselines
    eq_val        = equal_weight_baseline(df_prices_val)
    eq_test       = equal_weight_baseline(df_prices_test)
    mvo_val       = mvo_baseline(df_prices_val,  df_train_prices)
    mvo_test      = mvo_baseline(df_prices_test, df_train_prices)
    eq_val_m      = compute_metrics(eq_val)
    eq_test_m     = compute_metrics(eq_test)
    mvo_val_m     = compute_metrics(mvo_val)
    mvo_test_m    = compute_metrics(mvo_test)

    # Tabelas de métricas
    metric_cols = ["Total Return", "Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]

    def _print_table(m_dict, extra, period):
        rows = [{"Model": f"RL ({rt})", **{k: m[k] for k in metric_cols}}
                for rt, m in m_dict.items()]
        for label, m in extra:
            rows.append({"Model": label, **{k: m[k] for k in metric_cols}})
        df_tbl = pd.DataFrame(rows).set_index("Model")
        print(f"\n===== {period} =====")
        print(df_tbl.to_string(float_format=lambda x: f"{x:.4f}"))

    _print_table(val_metrics,
                 [("Equal Weight", eq_val_m), ("MVO", mvo_val_m)],
                 "VALIDAÇÃO")
    _print_table(test_metrics,
                 [("Equal Weight", eq_test_m), ("MVO", mvo_test_m)],
                 "TESTE")

    # Plots equity curve
    os.makedirs("plots", exist_ok=True)

    for period, curves, eq_curve, mvo_curve, title in [
        ("val",  val_curves,  eq_val,  mvo_val,  "Equity Curve — Validação"),
        ("test", test_curves, eq_test, mvo_test, "Equity Curve — Teste (2019–2024)"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        for rt, curve in curves.items():
            ax.plot(curve.values, label=f"RL ({rt})", color=colors[rt], linewidth=1.5)
        ax.plot(eq_curve.values,  label="Equal Weight", color="black", linestyle="--", linewidth=1.5)
        ax.plot(mvo_curve.values, label="MVO",          color="gray",  linestyle=":",  linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Dias")
        ax.set_ylabel("Valor do Portfólio")
        ax.legend()
        ax.grid(alpha=0.4)
        plt.tight_layout()
        path = f"plots/equity_curves_{period}_all_rewards.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Salvo: {path}")


def run_grid_search_evaluation():
    df_features, df_macro, df_prices = _load_val_data()
    evaluate_grid_search(df_features, df_prices, df_macro)


def run_convergence_evaluation(best_lr=1e-4, best_gamma=0.98):
    iniciar_semente(42)
    df_features, df_macro, df_prices = _load_val_data()
    evaluate_convergence_rewards(df_features, df_prices, df_macro, best_lr, best_gamma)


if __name__ == "__main__":
    evaluate_all_rewards()

    