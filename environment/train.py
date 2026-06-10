import pandas as pd
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from environment.environment import PortfolioEnv

import torch
import random

import torch as th

print("oi")
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


class RewardLoggerCallback(BaseCallback):
    """Registra a reward por step durante o treino e salva um gráfico ao final."""

    def __init__(self, reward_type: str, seed: int, plots_dir: str, rolling_window: int = 500):
        super().__init__()
        self.reward_type = reward_type
        self.seed = seed
        self.plots_dir = plots_dir
        self.rolling_window = rolling_window
        self._rewards: list = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self._rewards.extend(np.asarray(rewards).flatten().tolist())
        return True

    def _on_training_end(self) -> None:
        if not self._rewards:
            return

        os.makedirs(self.plots_dir, exist_ok=True)
        rewards = np.array(self._rewards)
        steps = np.arange(1, len(rewards) + 1)

        w = min(self.rolling_window, len(rewards))
        kernel = np.ones(w) / w
        rolling = np.convolve(rewards, kernel, mode="valid")
        steps_rolling = steps[w - 1:]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(steps, rewards, alpha=0.3, linewidth=0.5, color="steelblue", label="Reward por step")
        axes[0].plot(steps_rolling, rolling, linewidth=1.8, color="tab:blue",
                     label=f"Média móvel ({w} steps)")
        axes[0].set_ylabel("Reward")
        axes[0].set_title(f"Convergência — {self.reward_type}  (seed={self.seed})")
        axes[0].legend()
        axes[0].grid(alpha=0.4)

        # Cumulativa da reward
        cumsum = np.cumsum(rewards)
        axes[1].plot(steps, cumsum, color="tab:orange", linewidth=1.2)
        axes[1].set_xlabel("Timesteps")
        axes[1].set_ylabel("Reward Acumulada")
        axes[1].set_title("Reward Acumulada ao Longo do Treino")
        axes[1].grid(alpha=0.4)

        plt.tight_layout()
        fname = os.path.join(self.plots_dir, f"reward_convergence_{self.reward_type}_s{self.seed}.png")
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"Convergência salva em: {fname}")


def iniciar_semente(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    #Isso aqui é para GPU, caso eu va treinar no APUANA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def pre_processamento(df_features, df_prices):

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

    df_feat = df_features.drop(columns=cols_to_drop)

    df_price = df_prices.drop(columns=tickers_to_remove, errors="ignore")
    return df_feat, df_price

def make_env(features, macro, prices, reward_type, step_size, eval_mode=False):
    def _init():   # esta função interna é a factory de verdade
        return PortfolioEnv(features, macro, prices,
                            reward_type=reward_type,
                            step_size=step_size,
                            eval_mode=eval_mode)
    return _init


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

    df_macro_no_context = df_macro.iloc[:,:-1]

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

    env = PortfolioEnv(df_features, df_macro, df_prices, "log-retorno", use_context)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    n_actions = len(df_prices.columns)

    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    print("oi")
    model = DDPG(
        "MlpPolicy",
        env,
        seed=minha_seed,
        action_noise=action_noise,
        learning_rate=1e-4,
        gamma=0.98,         
        batch_size=256,
        buffer_size=100000,
        tau=0.005,
        train_freq=(1, "step"),
        gradient_steps=1,
        verbose=0,
        device="auto",
        policy_kwargs=policy_kwargs_ddpg
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ddpg_portfolio_model'
    )
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        progress_bar=True
    )

    env.save(f"vec_normalize_stats_macro.pkl")



   
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
    print("Treinamento concluído!")
   
    

def train_grid_search_ddpg():
    
    learning_rates = [1e-4, 3e-4]

    gammas         = [0.95, 0.98, 0.99]

    seeds          = [30, 42, 123]

    reward_types = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]

    base_path = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_train.csv")
    macro_path_   = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_train.csv")
    prices_path_  = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_train.csv")

    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df_macro    = pd.read_csv(macro_path_,   index_col=0, parse_dates=True)
    df_prices   = pd.read_csv(prices_path_,  index_col=0, parse_dates=True)

    nan_per_col = df_features.isna().sum()
    tickers_with_nan = {col.split('_F_')[0] for col, n in nan_per_col.items() if n > 0}
    cols_to_drop = [c for c in df_features.columns if c.split('_F_')[0] in tickers_with_nan]
    df_features = df_features.drop(columns=cols_to_drop)
    df_prices   = df_prices.drop(columns=list(tickers_with_nan), errors="ignore")

    import json
    tickers_path = os.path.join(base_path, "..", "tickers_grid_search.json")
    with open(tickers_path, "w") as f:
        json.dump(df_prices.columns.tolist(), f)
    print(f"Tickers usados ({len(df_prices.columns)}): salvos em tickers_grid_search.json")

    combos = list(itertools.product(learning_rates, gammas, seeds, reward_types))
    print(f"Grid search: {len(combos)} combinações")

    for lr, gamma, seed, reward_type in combos:

        run_name = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward_type}"

        print(f"\n{'='*55}\nTreinando: {run_name}\n{'='*55}")

        iniciar_semente(seed)

        env_inner = PortfolioEnv(df_features, df_macro, df_prices, reward_type, use_context)
        env = DummyVecEnv([lambda e=env_inner: e])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        n_actions    = len(df_prices.columns)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = DDPG(
            "MlpPolicy",
            env,
            seed=seed,
            action_noise=action_noise,
            learning_rate=lr,
            gamma=gamma,
            batch_size=256,
            buffer_size=100_000,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=0,
            device="auto",
            policy_kwargs=policy_kwargs_ddpg,
        )

        checkpoint_dir = os.path.join("models", run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=5000,
            save_path=checkpoint_dir,
            name_prefix=run_name,
        )
        model.learn(total_timesteps=300_000, callback=ckpt_cb, progress_bar=True)

        model.save(f"{run_name}_ddpg_portfolio_final")
        env.save(f"vec_normalize_{run_name}.pkl")
        print(f"Salvo: {run_name}")

    print("\nGrid search concluído!")


def train_omega_calmar(seed: int = 42, lr: float = 1e-4, gamma: float = 0.98,
                       total_timesteps: int = 300_000):
    """Treina DDPG com as rewards omega e calmar (sem grid search, seed único)."""

    reward_types = ["omega", "calmar"]

    base_path = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_train.csv")
    macro_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_train.csv")
    prices_path   = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_train.csv")
    plots_dir     = os.path.join(base_path, "..", "plots")

    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df_macro    = pd.read_csv(macro_path,    index_col=0, parse_dates=True)
    df_prices   = pd.read_csv(prices_path,   index_col=0, parse_dates=True)

    nan_per_col      = df_features.isna().sum()
    tickers_with_nan = {col.split('_F_')[0] for col, n in nan_per_col.items() if n > 0}
    cols_to_drop     = [c for c in df_features.columns if c.split('_F_')[0] in tickers_with_nan]
    df_features      = df_features.drop(columns=cols_to_drop)
    df_prices        = df_prices.drop(columns=list(tickers_with_nan), errors="ignore")

    all_rewards: dict = {}

    for reward_type in reward_types:
        run_name = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward_type}"
        print(f"\n{'='*55}\n{run_name}\n{'='*55}")

        iniciar_semente(seed)

        env_inner = PortfolioEnv(df_features, df_macro, df_prices, reward_type, use_context)
        env = DummyVecEnv([lambda e=env_inner: e])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        n_actions    = len(df_prices.columns)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        reward_logger = RewardLoggerCallback(reward_type, seed, plots_dir)

        checkpoint_dir = os.path.join("models", run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=5000,
            save_path=checkpoint_dir,
            name_prefix=run_name,
        )

        model = DDPG(
            "MlpPolicy",
            env,
            seed=seed,
            action_noise=action_noise,
            learning_rate=lr,
            gamma=gamma,
            batch_size=256,
            buffer_size=100_000,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=1,
            device="auto",
            policy_kwargs=policy_kwargs_ddpg,
        )

        model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb, reward_logger],
                    progress_bar=True)

        model.save(f"{run_name}_ddpg_portfolio_final")
        env.save(f"vec_normalize_{run_name}.pkl")
        all_rewards[reward_type] = reward_logger._rewards
        print(f"Salvo: {run_name}")

    # Gráfico combinado omega vs calmar
    os.makedirs(plots_dir, exist_ok=True)
    colors = {"omega": "tab:purple", "calmar": "tab:red"}

    _, axes = plt.subplots(len(reward_types), 1, figsize=(12, 5 * len(reward_types)))
    if len(reward_types) == 1:
        axes = [axes]

    for ax, reward_type in zip(axes, reward_types):
        rewards = np.array(all_rewards[reward_type])
        steps   = np.arange(1, len(rewards) + 1)

        w      = min(500, len(rewards))
        kernel = np.ones(w) / w
        rolled = np.convolve(rewards, kernel, mode="valid")
        steps_r = steps[w - 1:]

        ax.plot(steps, rewards, alpha=0.2, linewidth=0.5, color=colors[reward_type])
        ax.plot(steps_r, rolled, linewidth=1.8, color=colors[reward_type],
                label=f"{reward_type} (média móvel {w} steps)")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title(f"Convergência — {reward_type}  (lr={lr}, γ={gamma}, seed={seed})")
        ax.legend()
        ax.grid(alpha=0.4)

    plt.tight_layout()
    combined_path = os.path.join(plots_dir, "reward_convergence_omega_calmar.png")
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"\nGráfico combinado salvo em: {combined_path}")
    print("Treino omega + calmar concluído!")


def train_all_rewards(seed: int = 42, lr: float = 1e-4, gamma: float = 0.98,
                      total_timesteps: int = 300_000):
    """Treina DDPG com todas as reward functions numa única configuração."""

    reward_types = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]

    base_path     = os.path.dirname(os.path.abspath(__file__))
    features_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_train.csv")
    macro_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_train.csv")
    prices_path   = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_train.csv")
    plots_dir     = os.path.join(base_path, "..", "plots")

    df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
    df_macro    = pd.read_csv(macro_path,    index_col=0, parse_dates=True)
    df_prices   = pd.read_csv(prices_path,   index_col=0, parse_dates=True)

    nan_per_col      = df_features.isna().sum()
    tickers_with_nan = {col.split('_F_')[0] for col, n in nan_per_col.items() if n > 0}
    cols_to_drop     = [c for c in df_features.columns if c.split('_F_')[0] in tickers_with_nan]
    df_features      = df_features.drop(columns=cols_to_drop)
    df_prices        = df_prices.drop(columns=list(tickers_with_nan), errors="ignore")

    all_rewards: dict = {}

    for reward_type in reward_types:
        run_name = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward_type}"
        print(f"\n{'='*55}\n{run_name}\n{'='*55}")

        iniciar_semente(seed)

        env_inner = PortfolioEnv(df_features, df_macro, df_prices, reward_type, use_context)
        env = DummyVecEnv([lambda e=env_inner: e])
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        n_actions    = len(df_prices.columns)
        
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        reward_logger  = RewardLoggerCallback(reward_type, seed, plots_dir)
        checkpoint_dir = os.path.join("models", run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_cb = CheckpointCallback(
            save_freq=5000,
            save_path=checkpoint_dir,
            name_prefix=run_name,
        )

        model = DDPG(
            "MlpPolicy",
            env,
            seed=seed,
            action_noise=action_noise,
            learning_rate=lr,
            gamma=gamma,
            batch_size=256,
            buffer_size=100_000,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            verbose=0,
            device="auto",
            policy_kwargs=policy_kwargs_ddpg,
        )

        model.learn(total_timesteps=total_timesteps, callback=[ckpt_cb, reward_logger],
                    progress_bar=True)

        model.save(f"{run_name}_ddpg_portfolio_final")
        env.save(f"vec_normalize_{run_name}.pkl")
        all_rewards[reward_type] = reward_logger._rewards
        print(f"Salvo: {run_name}")

    # Gráfico comparativo de todas as rewards
    os.makedirs(plots_dir, exist_ok=True)
    colors = {
        "log-retorno":  "tab:blue",
        "huang-return": "tab:orange",
        "sortino":      "tab:green",
        "omega":        "tab:purple",
        "calmar":       "tab:red",
    }

    fig, axes = plt.subplots(len(reward_types), 1, figsize=(12, 5 * len(reward_types)))

    for ax, reward_type in zip(axes, reward_types):
        rewards = np.array(all_rewards[reward_type])
        steps   = np.arange(1, len(rewards) + 1)

        w       = min(500, len(rewards))
        kernel  = np.ones(w) / w
        rolled  = np.convolve(rewards, kernel, mode="valid")
        steps_r = steps[w - 1:]

        ax.plot(steps, rewards, alpha=0.2, linewidth=0.5, color=colors[reward_type])
        ax.plot(steps_r, rolled, linewidth=1.8, color=colors[reward_type],
                label=f"{reward_type} (média móvel {w} steps)")
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Reward")
        ax.set_title(f"Convergência — {reward_type}  (lr={lr}, γ={gamma}, seed={seed})")
        ax.legend()
        ax.grid(alpha=0.4)

    plt.tight_layout()
    combined_path = os.path.join(plots_dir, "reward_convergence_all_rewards.png")
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"\nGráfico comparativo salvo em: {combined_path}")
    print("Treino de todas as rewards concluído!")


if __name__ == "__main__":
    print("öi")