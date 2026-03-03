import pandas as pd
import numpy as np
import os
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from environment.environment import PortfolioEnv


def train():

    base_path = os.path.dirname(os.path.abspath(__file__))

    features_path = os.path.join(base_path, "..", "data", "processed", "master_features_train.csv")
    macro_path    = os.path.join(base_path, "..", "data", "processed", "master_macro_train.csv")
    prices_path   = os.path.join(base_path, "..", "data", "processed","master_prices_train.csv")

    print("Carregando dados...")

    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df_macro    = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    df_prices   = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    print(f"Ativos: {len(df_prices.columns)}")
    print(f"Features técnicas: {df_features.shape[1]}")
    print(f"Macro features: {df_macro.shape[1]}")

    env = PortfolioEnv(df_features, df_macro, df_prices)
    env = DummyVecEnv([lambda: env])

    n_actions = len(df_prices.columns)

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        learning_rate=1e-4,
        batch_size=128,
        buffer_size=50000,
        tau=0.005,
        gamma=0.99,
        device="auto"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ddpg_portfolio_model'
    )

    print("Iniciando treinamento...")
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save("ddpg_portfolio_final")
    print("Treinamento concluído!")
    

if __name__ == "__main__":
    train()