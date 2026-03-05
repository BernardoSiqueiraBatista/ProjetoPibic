import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG
from environment.environment import PortfolioEnv


def compute_metrics(portfolio_values):

    values = pd.Series(portfolio_values)

    if len(values) < 2:
        print("⚠️ Série muito curta para calcular métricas.")
        return {
            "Total Return": 0,
            "Annual Return": 0,
            "Volatility": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0
        }

    returns = values.pct_change().dropna()

    if len(returns) == 0:
        print("⚠️ Retornos vazios.")
        return {
            "Total Return": 0,
            "Annual Return": 0,
            "Volatility": 0,
            "Sharpe Ratio": 0,
            "Max Drawdown": 0
        }

    total_return = values.iloc[-1] - 1
    annual_return = (values.iloc[-1]) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-8)

    rolling_max = values.cummax()
    drawdown = (values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "Annual Return": annual_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }



def equal_weight_baseline(prices):

    returns = prices.pct_change().dropna()

    if len(returns) == 0:
        return pd.Series([1.0])

    equal_returns = returns.mean(axis=1)
    equity = (1 + equal_returns).cumprod()

    equity = pd.concat([pd.Series([1.0]), equity])
    equity.reset_index(drop=True, inplace=True)

    return equity


def mvo_baseline(prices, window=252, rebalance_freq=21):
    """
    Mean-Variance Optimization (Max Sharpe)
    - Long-only
    - Soma dos pesos = 1
    - Rebalanceamento periódico
    """

    returns = prices.pct_change().dropna()

    if len(returns) < window:
        print("⚠️ Dados insuficientes para MVO.")
        return pd.Series([1.0])

    portfolio_values = [1.0]
    weights = np.ones(prices.shape[1]) / prices.shape[1]  # inicial equal weight

    for t in range(window, len(returns)):

        # Rebalanceamento
        if (t - window) % rebalance_freq == 0:
            window_returns = returns.iloc[t-window:t]

            mu = window_returns.mean().values
            cov = window_returns.cov().values

            try:
                inv_cov = np.linalg.pinv(cov)

                raw_weights = inv_cov @ mu
                raw_weights = np.maximum(raw_weights, 0)  # long only

                if raw_weights.sum() > 0:
                    weights = raw_weights / raw_weights.sum()
                else:
                    weights = np.ones_like(raw_weights) / len(raw_weights)

            except Exception:
                weights = np.ones(prices.shape[1]) / prices.shape[1]

        daily_return = np.dot(weights, returns.iloc[t].values)

        portfolio_values.append(
            portfolio_values[-1] * (1 + daily_return)
        )

    return pd.Series(portfolio_values)


def evaluate_model(model_path, df_data, df_prices):

    df_prices = df_prices.loc[df_data.index]

    env = PortfolioEnv(df_data, df_prices, n_context_cols=4)

    obs, _ = env.reset()

    model = DDPG.load(model_path)

    portfolio_values = [1.0]

    done = False

    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )

    return pd.Series(portfolio_values)




def main():

    base_path = os.path.dirname(os.path.abspath(__file__))

    # ====== USAR VALIDAÇÃO ======
    print("\n===== AVALIAÇÃO EM VALIDAÇÃO (_val) =====")

    feat_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_val.csv")
    macro_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_val.csv")
    prices_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_val.csv")
    model_path = os.path.join(base_path, "..", "ddpg_portfolio_final.zip")

    # ====== CARREGAR DADOS ======
    df_features = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df_macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    # Garantir alinhamento
    common_index = df_features.index
    df_macro = df_macro.loc[common_index]
    df_prices = df_prices.loc[common_index]

    df_features = df_features.dropna()
    df_macro = df_macro.loc[df_features.index]
    df_prices = df_prices.loc[df_features.index]

    print("Período:",
          df_features.index.min().date(),
          "→",
          df_features.index.max().date())


    env = PortfolioEnv(df_features, df_macro, df_prices)
    obs, _ = env.reset()

    model = DDPG.load(model_path)

    portfolio_values = [1.0]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        portfolio_return = info.get("portfolio_return", 0)
        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )

    rl_values = pd.Series(portfolio_values)

    weights_df = pd.DataFrame(env.action_history)

    weights_df.to_csv(os.path.join(base_path, "..", "data", "ddpg_weights_val.csv"), index=False)

    eq_values = equal_weight_baseline(df_prices)
    mvo_values = mvo_baseline(df_prices)

  
    rl_metrics = compute_metrics(rl_values)
    eq_metrics = compute_metrics(eq_values)
    mvo_metrics = compute_metrics(mvo_values)

    print("\n===== RESULTADOS =====")

    for name, metrics in {
        "RL (DDPG)": rl_metrics,
        "Equal Weight": eq_metrics,
        "MVO": mvo_metrics
    }.items():
        print(f"\n--- {name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


    plt.figure(figsize=(12, 6))
    plt.plot(rl_values.values, label="RL (DDPG)")
    plt.plot(eq_values.values, label="Equal Weight")
    plt.plot(mvo_values.values, label="MVO")
    plt.legend()
    plt.grid()
    plt.title("Equity Curve Comparison (_val)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()