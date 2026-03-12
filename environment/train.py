import pandas as pd
import numpy as np
import os
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from environment.environment import PortfolioEnv


def train():

    base_path = os.path.dirname(os.path.abspath(__file__))

    features_path = os.path.join(base_path, "..", "data", "processed","data_train_val", "master_features_train.csv")
    macro_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_train.csv")
    prices_path   = os.path.join(base_path, "..", "data", "processed","data_train_val", "master_prices_train.csv")

    print("Carregando dados...")

    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df_macro    = pd.read_csv(macro_path, index_col=0, parse_dates=True)
    df_prices   = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    nan_per_col = df_features.isna().sum()

    tickers_with_nan = set()

    for col, n_nan in nan_per_col.items():
        if n_nan > 0:
            ticker = col.split('_F_')[0]
            tickers_with_nan.add(ticker)

    print("Ativos com NaN:")
    print(tickers_with_nan)

    print("Total NaN:", df_features.isna().sum().sum())

    tickers_to_remove = list(tickers_with_nan)

    print("Removendo ativos:", tickers_to_remove)

    cols_to_drop = [
        col for col in df_features.columns
        if col.split('_F_')[0] in tickers_to_remove
    ]

    df_features = df_features.drop(columns=cols_to_drop)

    df_prices = df_prices.drop(columns=tickers_to_remove, errors="ignore")

    env = PortfolioEnv(df_features, df_macro, df_prices)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.save("vec_normalize_stats.pkl")

    n_actions = len(df_prices.columns)

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=100000,
    tau=0.005,
    gamma=0.995,
    train_freq=(1, "step"),
    gradient_steps=1,
    verbose=1,
    device="auto"
)
    model_2 = SAC("MlpPolicy", env, learning_rate=3e-4, batch_size=256, buffer_size=100000, train_freq=(1, "step"), gradient_steps=1, verbose=1, device="auto")
    

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ddpg_portfolio_model'
    )
    checkpoint_callback_2 = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='sac_portfolio_model'
    )

    print("Iniciando treinamento...")
    model.learn(
        total_timesteps=200000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    model_2.learn(
        total_timesteps=200000,
        callback=checkpoint_callback_2,
        progress_bar=True
    )

    model.save("ddpg_portfolio_final")
    model_2.save("sac_portfolio_final")
    print("Treinamento concluído!")
   
    

if __name__ == "__main__":
    train()