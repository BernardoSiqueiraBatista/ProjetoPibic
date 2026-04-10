import pandas as pd
import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from environment.environment import PortfolioEnv
import torch
import random

import torch as th

policy_kwargs_ppo = dict(
    activation_fn=th.nn.ReLU,

    net_arch=dict(
            
        pi=[256, 256, 128], 

        vf=[256, 128]

    )
)

policy_kwargs_sac = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(
        pi=[256, 256, 128],
        qf=[256, 128]
    )
)

policy_kwargs_ddpg = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(

        pi=[256, 256, 128],

        qf=[256, 128]
    )

)

use_context = True
def iniciar_semente(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #Isso aqui é para GPU, caso eu va treinar no APUANA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def train():

    minha_seed = 42
    iniciar_semente(minha_seed)


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

    env = PortfolioEnv(df_features, df_macro, df_prices, use_context)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    n_actions = len(df_prices.columns)

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    model = DDPG(
    "MlpPolicy",
    env,
    seed=minha_seed,
    action_noise=action_noise,
    learning_rate=1e-4,
    batch_size=256,
    buffer_size=100000,
    tau=0.005,
    gamma=0.98,
    train_freq=(1, "step"),
    gradient_steps=1,
    verbose=1,
    device="auto",
    policy_kwargs=policy_kwargs_ddpg
)
    """model_2 = SAC("MlpPolicy", env, seed=minha_seed, learning_rate=1e-4, 
                  batch_size=256, buffer_size=100_000, tau = 0.001, gamma=0.99, train_freq=(1, "step"), 
                  gradient_steps=1, verbose=1, device="auto", policy_kwargs=policy_kwargs_sac)
    
    model_3 = PPO(
    "MlpPolicy",
    env,
    seed=minha_seed,
    learning_rate=1e-4,
    n_steps=1024,         
    batch_size=64,
    clip_range=0.2,
    n_epochs=15,         
    gamma=0.99,          
    ent_coef=0.05,       
    verbose=1,
    device="cpu",
    policy_kwargs=policy_kwargs_ppo
)"""


    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ddpg_portfolio_model'
    )
    """checkpoint_callback_2 = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='sac_portfolio_model'
    )
    checkpoint_callback_3 = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ppo_portfolio_model'
    )"""

    print("Iniciando treinamento...")
    
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    """model_2.learn(
        total_timesteps=100000,
        callback=checkpoint_callback_2,
        progress_bar=True
    )
    
    model_3.learn(
        total_timesteps=300000,
        callback=checkpoint_callback_3,
        progress_bar=True
    )"""

    
    model.save("ddpg_portfolio_final")
    """model_2.save("sac_portfolio_final")
    
    model_3.save("ppo_portfolio_final")
    """
    env.save("vec_normalize_stats.pkl")
    print("Treinamento concluído!")
   
    

if __name__ == "__main__":
    train()