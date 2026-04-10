import os
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
prices_train_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_train.csv")
feat_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_val.csv")
macro_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_val.csv")
prices_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_val.csv")
model_path = os.path.join(base_path, "..", "ddpg_portfolio_final.zip")
model_sac = os.path.join(base_path, "..", "sac_portfolio_final.zip")
model_ppo = os.path.join(base_path, "..", "ppo_portfolio_final.zip")

import random
import torch

def iniciar_semente(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #Isso aqui é para CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def compute_metrics(portfolio_values):

    values = pd.Series(portfolio_values)

  
    returns = values.pct_change().dropna()

    total_return = values.iloc[-1] - 1
    annual_return = (values.iloc[-1]) ** (252 / len(returns)) - 1
    volatility = returns.std() * np.sqrt(252)
    
    risk_free_daily = 0.08 / 252
    sharpe = np.sqrt(252) * (returns.mean() - risk_free_daily) / (returns.std() + 1e-8)

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

def evaluate_model(model_path, df_features, df_prices, df_macro, SEED=42):

  

    df_prices = df_prices.loc[df_features.index]

    env = PortfolioEnv(df_features, df_macro, df_prices)
    env = DummyVecEnv([lambda: env])
    env.seed(SEED)
    
    env = VecNormalize.load("vec_normalize_stats.pkl", env)

    env.training = False 

    env.norm_reward = False

    obs = env.reset()
    env.envs[0].action_history  = []

    model = DDPG.load(model_path)

    portfolio_values = [1.0]

    done = False

    step_count = 0
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(env.envs[0].action_history)

    weights_df.to_csv(os.path.join(base_path, "..", "data", "ddpg_weights_val.csv"), index=False)
    return pd.Series(portfolio_values)

def evaluate_model_sac(model_path, df_features, df_prices, df_macro, SEED=42):

    df_prices = df_prices.loc[df_features.index]

    env = PortfolioEnv(df_features, df_macro, df_prices)
    env = DummyVecEnv([lambda: env])
    env.seed(SEED)
    
    env = VecNormalize.load("vec_normalize_stats.pkl", env)

    env.training = False 

    env.norm_reward = False
    obs = env.reset()
    env.envs[0].action_history = []

    model = SAC.load(model_path)

    portfolio_values = [1.0]

    done = False
    step_count = 0
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]


        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(env.envs[0].action_history)
    weights_df.to_csv(os.path.join(base_path, "..", "data", "sac_weights_val.csv"), index=False)
    return pd.Series(portfolio_values)

def evaluate_model_ppo(model_path, df_features, df_prices, df_macro, SEED=42):
    
    df_prices = df_prices.loc[df_features.index]

    env = PortfolioEnv(df_features, df_macro, df_prices)
    env = DummyVecEnv([lambda: env])
    env.seed(SEED)
    
    env = VecNormalize.load("vec_normalize_stats.pkl", env)

    env.training = False 

    env.norm_reward = False
    obs = env.reset()
    env.envs[0].action_history = []

    model = PPO.load(model_path)

    portfolio_values = [1.0]

    done = False
    step_count = 0
    while not done:

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, dones, infos = env.step(action)
        done = dones[0]
        info = infos[0]


        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(env.envs[0].action_history)
    weights_df.to_csv(os.path.join(base_path, "..", "data", "ppo_weights_val.csv"), index=False)
    return pd.Series(portfolio_values)




def main():

    base_path = os.path.dirname(os.path.abspath(__file__))

    SEED = 42
    iniciar_semente(SEED)


  
    print("\n===== AVALIAÇÃO EM VALIDAÇÃO (_val) =====")
    
    df_train_prices = pd.read_csv(prices_train_path, index_col=0, parse_dates=True)
    df_features = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df_macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    df_prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)

   
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


    rl_values = evaluate_model(model_path, df_features, df_prices,df_macro, SEED)

    rl_values_sac = evaluate_model_sac(model_sac, df_features, df_prices,df_macro, SEED)

    r_values_ppo = evaluate_model_ppo(model_ppo, df_features, df_prices,df_macro, SEED)

    eq_values = equal_weight_baseline(df_prices)

    mvo_values = mvo_baseline(df_prices, df_train_prices)

  
    rl_metrics = compute_metrics(rl_values)
    sac_metrics = compute_metrics(rl_values_sac)
    eq_metrics = compute_metrics(eq_values)
    mvo_metrics = compute_metrics(mvo_values)
    ppo_metrics = compute_metrics(r_values_ppo)

    print("\n===== RESULTADOS =====")
    model = PPO.load(model_ppo)
    print(model.policy)
    print( "oi")

    for name, metrics in {
        "RL (DDPG)": rl_metrics,
        "Equal Weight": eq_metrics,
        #Ignorar por enquanto
        #"MVO": mvo_metrics,
        
        #Ignorar por enquanto

        #"RL (SAC)": sac_metrics,
        #"PPO" : ppo_metrics
    }.items():
        print(f"\n--- {name} ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")


    plt.figure(figsize=(12, 6))

    plt.plot(eq_values.values, label="Equal Weight")

    #plt.plot(mvo_values.values, label="MVO")
    #plt.plot(rl_values_sac.values, label="RL (SAC)")
    plt.plot(rl_values.values, label="RL (DDPG)")
    #plt.plot(r_values_ppo.values, label="RL (PPO)")
    plt.legend()
    plt.grid()
    plt.title("Equity Curve Comparison (_val)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()