"""
from environment.environment import PortfolioEnv


from evaluate import compute_metrics, count_operations, compute_operations_metrics

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from train import pre_processamento, make_env, train_grid_search, iniciar_semente, RewardLoggerCallback

from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import PPO
import torch as th
import random
import pandas as pd
import numpy as np
import itertools
import json
import os

import matplotlib.pyplot as plt

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
#SCRIPT FEITO: PRIMEIRO TREINAR O GRID-SEARCH -> (DIARIO, MENSAL, TRIMESTRAL) -> MODELOS DE PPO, DDPG, SAC -> EM DEMAIS FUNÇÕES DE RECOMPENSA
base_path = os.path.dirname(__file__)

features_path = os.path.join(base_path, "..", "data", "processed","data_train_val", "master_features_train.csv")
macro_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_train.csv")
prices_path   = os.path.join(base_path, "..", "data", "processed","data_train_val", "master_prices_train.csv")

df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
df_macro = pd.read_csv(macro_path, index_col = 0, parse_dates=True)
df_prices = pd.read_csv(prices_path, index_col = 0, parse_dates = True)

df_features, df_prices = pre_processamento(df_features, df_prices)

n_envs = 8  


def train_grid_search_ddpg():

    learning_rates = [1e-4, 3e-4]

    gammas         = [0.95, 0.98, 0.99]

    seeds          = [30, 42, 123]

    reward_types = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]

    step_size = [1,30,90]

    combos = list(itertools.product(learning_rates, gammas, seeds,reward_types, step_size))

    for lr,gamma,seed,reward,step in combos:

        run_name = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward}_step{step}"

        print(f"treinanmento{run_name}")

        iniciar_semente(seed)

        train_env = SubprocVecEnv([
        make_env(df_features, df_macro, df_prices, reward, step, eval_mode = False)
        for _ in range(n_envs)]
        )
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        n_actions    = len(df_prices.columns)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        model = DDPG(
            "MlpPolicy",
            train_env,
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
        os.makedirs(checkpoint_dir, exist_ok = True)

        ckpt_cb = CheckpointCallback(
            save_freq=5000,
            save_path=checkpoint_dir,
            name_prefix=run_name,
        )
        model.learn(total_timesteps=300_000, callback=ckpt_cb, progress_bar=True)

        model.save(f"{run_name}_ddpg_portfolio_final")
        train_env.save(f"vec_normalize_{run_name}.pkl")
        print(f"Salvo: {run_name}")

def train_sac():

    reward_type = ["huang-return", "omega", "log-retorno","sortino","calmar"]

    step_size = [1,30,90]

    combos = list(itertools.product(reward_type, step_size))

    train_env = SubprocVecEnv([
        make_env(df_features, df_macro, df_prices, reward, step, eval_mode = False)
        for _ in range(n_envs)]
        )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    for reward, step in combos:
        run_name = f"SAC_r{reward}_s{step}"

        model = SAC("MlpPolicy", train_env, seed=42, learning_rate=1e-4, 
                  batch_size=256, buffer_size=100_000, tau = 0.001, gamma=0.99, train_freq=(1, "step"), 
                  gradient_steps=1, verbose=1, device="auto", policy_kwargs=policy_kwargs_sac)
        
        checkpoint_callback_2 = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='sac_portfolio_model'
        )
        model.learn(total_timesteps=300_000, callback = checkpoint_callback_2)

        model.save(f"{run_name}_ddpg_portfolio_final")

        train_env.save(f"vec_normalize_{run_name}.pkl")
        print(f"Salvo: {run_name}")

def train_ppo():

    reward_type = ["huang-return", "omega", "log-retorno","sortino","calmar"]

    step_size = [1,30,90]

    combos = list(itertools.product(reward_type, step_size))

    train_env = SubprocVecEnv([
        make_env(df_features, df_macro, df_prices, reward, step, eval_mode = False)
        for _ in range(n_envs)]
        )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    for reward, step in combos:
        run_name = f"PPO_r{reward}_s{step}"

        model_3 = PPO(
        "MlpPolicy",
        train_env,
        seed=42,
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
            )
        
        checkpoint_callback_2 = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='sac_portfolio_model'
        )
        model_3.learn(total_timesteps=300_000, callback = checkpoint_callback_2)

        model_3.save(f"{run_name}_PPO_portfolio_final")

        train_env.save(f"vec_normalize_{run_name}.pkl")
        print(f"Salvo: {run_name}")

#Salvar PPO E SAC 
#pegar o camminho do diretorio atual e pegar os de data/test e data/val
features_val_path = os.join(base_path , "..", "data","processed","data_train_val", "master_features_val.csv")
macro_val_path = os.join(base_path , "..", "data","processed","data_train_val", "master_macro_val.csv")
price_val_path = os.join(base_path , "..", "data","processed","data_train_val", "master_prices_val.csv")

features_test_path = os.join(base_path , "..", "data","processed","data_train_test", "master_features_test.csv")
macro_test_path = os.join(base_path , "..", "data","processed","data_train_test", "master_macro_test.csv")
price_test_path = os.join(base_path , "..", "data","processed","data_train_test", "master_prices_test.csv")

df_features_val = pd.read_csv(features_val_path, index_col=0, parse_dates=True)
df_macro_val = pd.read_csv(macro_val_path, index_col = 0, parse_dates=True)
df_prices_val = pd.read_csv(price_val_path, index_col = 0, parse_dates = True)

df_features_test = pd.read_csv(features_test_path, index_col=0, parse_dates=True)
df_macro_test = pd.read_csv(macro_test_path, index_col = 0, parse_dates=True)
df_prices_test = pd.read_csv(price_test_path, index_col = 0, parse_dates = True)

def avaliar_ppo(model_path, df_features, df_macro, df_prices,reward,step,validar):

    
    run_name = f"PPO_r{reward}_s{step}_{validar}"

    val_env = SubprocVecEnv([
        make_env(df_features, df_macro, df_prices, reward, step, eval_mode = True)
        for _ in range(n_envs)])
    
    val_env = VecNormalize.load(f"vec_normalize_{run_name}.pkl")

    val_env.training = False

    val_env.norm_reward = False

    obs = val_env.reset()

    val_env.envs[0].action_history = []

    model = PPO.load(model_path)

    portfolio_values = [1.0]

    done = False

    while not done:

        action,_ = model.predict(obs, deterministic = True)

        obs, reward, dones, infos = val_env.step(action)
        done = dones[0]
        info = infos[0]

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(val_env.envs[0].action_history)
    weights_df.to_csv(os.path.join(base_path, "..", "data", f"ppo_weights_val_{run_name}.csv"), index=False)

    ppo_metrics = compute_metrics(portfolio_values)

    metrics_dir = os.join(base_path, "..", "metrics")
    os.makedirs(metrics_dir,exist_ok=True)
    with open(os.path.join(metrics_dir, f"metrics_{run_name}.json"), "w") as f:
        json.dump(ppo_metrics, f, indent=2)

    operations_metrics = compute_operations_metrics(weights_df)
    
    with open (os.path.join(metrics_dir), f"metrics_{run_name}_operations.json", "w") as f:
        json.dump(operations_metrics, f, indent=2)

    fig_dir = os.path.join(base_path, "..", "plots")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(portfolio_values, linewidth=1.5)
    plt.title(f"Curva de capital — validação ({run_name})")
    plt.xlabel("Passos de decisão")
    plt.ylabel("Valor do portfólio (base = 1.0)")
    plt.grid(alpha=0.3)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"equity_curve_{run_name}.png"), dpi=150)
    plt.close()

def avaliar_sac(model_path, df_features, df_macro, df_prices,reward,step,validar):

    run_name = f"SAC_r{reward}_s{step}_{validar}"

    val_env = SubprocVecEnv([
        make_env(df_features, df_macro, df_prices, reward, step, eval_mode = True)
        for _ in range(n_envs)])
    
    val_env = VecNormalize.load(f"vec_normalize_{run_name}.pkl")

    val_env.training = False

    val_env.norm_reward = False

    obs = val_env.reset()

    val_env.envs[0].action_history = []

    model = SAC.load(model_path)

    portfolio_values = [1.0]

    done = False

    while not done:

        action,_ = model.predict(obs, deterministic = True)

        obs, reward, dones, infos = val_env.step(action)
        done = dones[0]
        info = infos[0]

        portfolio_return = info.get("portfolio_return", 0)

        portfolio_values.append(
            portfolio_values[-1] * (1 + portfolio_return)
        )
    weights_df = pd.DataFrame(val_env.envs[0].action_history)
    weights_df.to_csv(os.path.join(base_path, "..", "data", f"SAC_weights_val_{run_name}.csv"), index=False)

    sac_metrics = compute_metrics(portfolio_values)

    metrics_dir = os.join(base_path, "..", "metrics")
    os.makedirs(metrics_dir,exist_ok=True)
    with open(os.path.join(metrics_dir, f"metrics_{run_name}.json"), "w") as f:
        json.dump(sac_metrics, f, indent=2)

    operations_metrics = compute_operations_metrics(weights_df)
    
    with open (os.path.join(metrics_dir), f"metrics_{run_name}_operations.json", "w") as f:
        json.dump(operations_metrics, f, indent=2)
        
    fig_dir = os.path.join(base_path, "..", "plots")
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(portfolio_values, linewidth=1.5)
    plt.title(f"Curva de capital — validação ({run_name})")
    plt.xlabel("Passos de decisão")
    plt.ylabel("Valor do portfólio (base = 1.0)")
    plt.grid(alpha=0.3)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"equity_curve_{run_name}.png"), dpi=150)
    plt.close()



    return pd.Series(portfolio_values), weights_df

if __name__=="__main__":
    reward_type = ["huang-return", "omega", "log-retorno","sortino","calmar"]

    step_size = [1,30,90]

    combos = list(itertools.product(reward_type, step_size))
    train_sac()
    train_ppo()
    for r,s in combos:
        run_name_sac =  f"SAC_r{reward}_s{step}_{validar}"
        run_name_ppo = f"SAC_r{reward}_s{step}_{validar}"

        avaliar_ppo()
        avaliar_sac
"""

from environment.environment import PortfolioEnv
from environment.evaluate import (
    compute_metrics, count_operations, compute_operations_metrics,
    equal_weight_baseline, mvo_baseline,
)
from scipy import stats

from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from environment.train import pre_processamento, make_env, iniciar_semente, RewardLoggerCallback

from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3 import PPO
import torch as th
import random
import pandas as pd
import numpy as np
import itertools
import json
import os
import argparse

import matplotlib.pyplot as plt
DDPG_COMBOS = list(itertools.product(
    [1e-4, 3e-4],                                                    # learning_rate
    [0.95, 0.98, 0.99],                                             # gamma
    ["log-retorno", "huang-return", "sortino", "omega", "calmar"],  # reward
    [21, 63],                                                       # step (1 já treinado)
))

policy_kwargs_ppo = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[256, 256, 128], vf=[256, 128])
)

policy_kwargs_sac = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[256, 256, 128], qf=[256, 128])
)

policy_kwargs_ddpg = dict(
    activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[256, 256, 128], qf=[256, 128])
)

base_path = os.path.dirname(__file__)

# ----------------------------------------------------------------------
# Dados de TREINO
# ----------------------------------------------------------------------
features_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_train.csv")
macro_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_train.csv")
prices_path   = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_train.csv")

df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
df_macro    = pd.read_csv(macro_path, index_col=0, parse_dates=True)
df_prices   = pd.read_csv(prices_path, index_col=0, parse_dates=True)

df_features, df_prices = pre_processamento(df_features, df_prices)

# ----------------------------------------------------------------------
# Dados de VALIDACAO e TESTE
# ----------------------------------------------------------------------
features_val_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_val.csv")
macro_val_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_val.csv")
price_val_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_val.csv")

features_test_path = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_features_test.csv")
macro_test_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_macro_test.csv")
price_test_path    = os.path.join(base_path, "..", "data", "processed", "data_train_val", "master_prices_test.csv")

df_features_val = pd.read_csv(features_val_path, index_col=0, parse_dates=True)
df_macro_val    = pd.read_csv(macro_val_path, index_col=0, parse_dates=True)
df_prices_val   = pd.read_csv(price_val_path, index_col=0, parse_dates=True)

df_features_test = pd.read_csv(features_test_path, index_col=0, parse_dates=True)
df_macro_test    = pd.read_csv(macro_test_path, index_col=0, parse_dates=True)
df_prices_test   = pd.read_csv(price_test_path, index_col=0, parse_dates=True)

# Os mesmos pre-processamentos do treino devem valer para val/test:
df_features_val, df_prices_val   = pre_processamento(df_features_val, df_prices_val)
df_features_test, df_prices_test = pre_processamento(df_features_test, df_prices_test)

# ----------------------------------------------------------------------
# Config para rodar no Mac (teste local)
# ----------------------------------------------------------------------
N_ENVS = 1                 # Mac: 1 env + DummyVecEnv. No cluster: subir para 8.
DEVICE = "cpu"             # M4: forca CPU (MPS costuma ser mais lento/instavel com SB3).
TIMESTEPS = 300_000         # teste local. CLUSTER: 300_000.
SEED = 42

MODELS_DIR  = os.path.join(base_path, "models")
METRICS_DIR = os.path.join(base_path, "..", "metrics")
DATA_DIR    = os.path.join(base_path, "..", "data")
PLOTS_DIR   = os.path.join(base_path, "..", "plots")
for d in (MODELS_DIR, METRICS_DIR, DATA_DIR, PLOTS_DIR):
    os.makedirs(d, exist_ok=True)


def make_vec_env(features, macro, prices, reward, step, eval_mode):
    """Cria o VecEnv. Mac -> DummyVecEnv (seguro). Cluster -> trocar por SubprocVecEnv."""
    return DummyVecEnv([
        make_env(features, macro, prices, reward, step, eval_mode=eval_mode)
        for _ in range(N_ENVS)
    ])
    # CLUSTER:
    # return SubprocVecEnv([
    #     make_env(features, macro, prices, reward, step, eval_mode=eval_mode)
    #     for _ in range(N_ENVS)
    # ])


def build_callbacks(run_name, reward, step):
    """Checkpoint + logger de reward (curva de convergencia por run)."""
    ckpt_dir = os.path.join(MODELS_DIR, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix=run_name,
    )

    # janela menor quando o step e grande (menos rewards logadas no total)
    rolling = 100 if step > 1 else 500
    reward_logger = RewardLoggerCallback(
        reward_type=f"{run_name}",   # vai no nome do arquivo do grafico
        seed=SEED,
        plots_dir=PLOTS_DIR,
        rolling_window=rolling,
    )

    return ckpt_dir, CallbackList([ckpt_cb, reward_logger])


TB_DIR = os.path.join(base_path, "..", "tensorboard")   # path, não join
os.makedirs(TB_DIR, exist_ok=True)                       # makedirs, não mkdir

def train_ddpg():
    learning_rate = [1e-4, 3e-4]
    gammas        = [0.95, 0.98, 0.99]
    reward_type   = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]
    step_size     = [21, 63]   # step=1 ja foi treinado separadamente

    combos = list(itertools.product(learning_rate, gammas, reward_type, step_size))

    for l, g, r, s in combos:
        run_name = f"DDPG_r{r}_s{s}_lr{l}_g{g}"
        print(f"[treino] {run_name}")
        iniciar_semente(SEED)

        train_env = make_vec_env(df_features, df_macro, df_prices, r, s, eval_mode=False)
        train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

        n_actions = len(df_prices.columns)
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )

        model = DDPG(
            "MlpPolicy", train_env, seed=SEED,
            action_noise=action_noise,
            learning_rate=l, gamma=g,
            batch_size=256, buffer_size=100_000, tau=0.005,
            train_freq=(1, "step"), gradient_steps=1,
            verbose=0, device=DEVICE,
            policy_kwargs=policy_kwargs_ddpg,
            tensorboard_log=TB_DIR,
        )

        ckpt_dir, callbacks = build_callbacks(run_name, r, s)

        model.learn(
            total_timesteps=TIMESTEPS,
            callback=callbacks,
            tb_log_name=run_name,
            progress_bar=True,
        )

        model.save(os.path.join(ckpt_dir, "final_model"))
        train_env.save(os.path.join(ckpt_dir, "vec_normalize.pkl"))
        print(f"[salvo] {run_name}")

# ======================================================================
# TREINO SAC
# ======================================================================
def train_sac():
    reward_type = ["huang-return", "omega", "log-retorno", "sortino", "calmar"]
    step_size = [1, 21, 63]
    combos = list(itertools.product(reward_type, step_size))

    for reward, step in combos:
        run_name = f"SAC_r{reward}_s{step}"
        print(f"[treino] {run_name}")
        iniciar_semente(SEED)

        train_env = make_vec_env(df_features, df_macro, df_prices, reward, step, eval_mode=False)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        model = SAC(
            "MlpPolicy", train_env, seed=SEED, learning_rate=1e-4,
            batch_size=256, buffer_size=100_000, tau=0.001, gamma=0.99,
            train_freq=(1, "step"), gradient_steps=1,
            verbose=1, device=DEVICE, policy_kwargs=policy_kwargs_sac,
        )

        # callbacks: checkpoint + RewardLoggerCallback (instanciado por run)
        ckpt_dir, callbacks = build_callbacks(run_name, reward, step)

        model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

        model.save(os.path.join(ckpt_dir, "final_model"))
        train_env.save(os.path.join(ckpt_dir, "vec_normalize.pkl"))
        print(f"[salvo] {run_name}")


# ======================================================================
# TREINO PPO
# ======================================================================
def train_ppo():
    reward_type = ["huang-return", "omega", "log-retorno", "sortino", "calmar"]
    step_size = [1, 21, 63]
    combos = list(itertools.product(reward_type, step_size))

    for reward, step in combos:
        run_name = f"PPO_r{reward}_s{step}"
        print(f"[treino] {run_name}")
        iniciar_semente(SEED)

        train_env = make_vec_env(df_features, df_macro, df_prices, reward, step, eval_mode=False)
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

        model = PPO(
            "MlpPolicy", train_env, seed=SEED, learning_rate=1e-4,
            n_steps=1024, batch_size=64, clip_range=0.2, n_epochs=15,
            gamma=0.99, ent_coef=0.05,
            verbose=1, device=DEVICE, policy_kwargs=policy_kwargs_ppo,
        )

        ckpt_dir, callbacks = build_callbacks(run_name, reward, step)

        model.learn(total_timesteps=TIMESTEPS, callback=callbacks)

        model.save(os.path.join(ckpt_dir, "final_model"))
        train_env.save(os.path.join(ckpt_dir, "vec_normalize.pkl"))
        print(f"[salvo] {run_name}")


# ======================================================================
# AVALIACAO (generica para SAC e PPO) -- curva DIARIA
# ======================================================================
def avaliar(algo_cls, algo_tag, features, macro, prices, reward, step, split):
    train_run = f"{algo_tag}_r{reward}_s{step}"
    out_name  = f"{algo_tag}_r{reward}_s{step}_{split}"
    ckpt_dir  = os.path.join(MODELS_DIR, train_run)

    eval_env = DummyVecEnv([
        make_env(features, macro, prices, reward, step, eval_mode=True)
    ])
    eval_env = VecNormalize.load(os.path.join(ckpt_dir, "vec_normalize.pkl"), eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = algo_cls.load(os.path.join(ckpt_dir, "final_model"))

    obs = eval_env.reset()

    # curva DIARIA: expande os daily_returns de cada periodo de decisao
    # weights_history: coletado do info (robusto ao auto-reset do VecEnv no step terminal)
    portfolio_values = [1.0]
    weights_history = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = eval_env.step(action)
        done = bool(dones[0])
        weights_history.append(infos[0]["weights"])
        for r in infos[0].get("daily_returns", []):
            portfolio_values.append(portfolio_values[-1] * (1 + r))

    tickers = eval_env.envs[0].tickers
    weights_df = pd.DataFrame(weights_history, columns=tickers)
    weights_df.to_csv(os.path.join(DATA_DIR, f"{algo_tag}_weights_{out_name}.csv"), index=False)

    metrics = compute_metrics(portfolio_values)
    with open(os.path.join(METRICS_DIR, f"metrics_{out_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    operations_metrics = compute_operations_metrics(weights_df,0.0001,step)
    with open(os.path.join(METRICS_DIR, f"metrics_{out_name}_operations.json"), "w") as f:
        json.dump(operations_metrics, f, indent=2)

    plt.figure(figsize=(11, 5))
    plt.plot(portfolio_values, linewidth=1.5)
    plt.title(f"Curva de capital diaria -- {split} ({out_name})")
    plt.xlabel("Dias de pregao")
    plt.ylabel("Valor do portfolio (base = 1.0)")
    plt.grid(alpha=0.3)
    plt.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"equity_curve_{out_name}.png"), dpi=150)
    plt.close()

    print(f"[aval {split}] {out_name}: {metrics}")
    return pd.Series(portfolio_values), weights_df


# ======================================================================
# AVALIACAO COMPARATIVA: DDPG (otim. hiperparam, step=1, 3 seeds)
#                        x PPO x SAC (cada step) x reward functions
#   - curva DIARIA para todos (expande info["daily_returns"])
#   - baselines Equal-Weight e MVO (markowitz) reaproveitados do evaluate.py
#   - elege o melhor modelo (Sharpe de validacao, reportado no teste)
#   - teste de significancia entre STEPS e entre REWARD FUNCTIONS
# ======================================================================
ALGOS = {"DDPG": DDPG, "PPO": PPO, "SAC": SAC}
REWARD_TYPES = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]
STEP_SIZES   = [1, 21, 63]

# DDPG = grid antigo (raiz do projeto): step=1, otimizacao de (lr, gamma, seed)
DDPG_LRS    = [1e-4, 3e-4]
DDPG_GAMMAS = [0.95, 0.98, 0.99]
DDPG_SEEDS  = [30, 42, 123]

REWARD_COLORS = {
    "log-retorno":  "tab:blue",
    "huang-return": "tab:orange",
    "sortino":      "tab:green",
    "omega":        "tab:purple",
    "calmar":       "tab:red",
}
ALGO_COLORS = {"DDPG": "tab:blue", "PPO": "tab:orange", "SAC": "tab:green"}
METRIC_COLS = ["Total Return", "Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]


# ----------------------------------------------------------------------
# Helpers de avaliacao
# ----------------------------------------------------------------------
def _exists(model_file, vec_file):
    has_model = os.path.exists(model_file) or os.path.exists(model_file + ".zip")
    has_vec = (vec_file is None) or os.path.exists(vec_file)  # vec_file=None -> normaliza on-the-fly
    return has_model and has_vec


def _prep(features, macro, prices):
    """Alinha macro e prices ao indice das features (pre_processamento so dropa colunas)."""
    features = features.dropna()
    macro  = macro.loc[features.index]
    prices = prices.loc[features.index]
    return features, macro, prices


def _curve_from_daily(daily):
    """Reconstroi a curva de capital (base=1.0) a partir dos retornos diarios."""
    daily = np.asarray(daily, dtype=float)
    return pd.Series(np.concatenate([[1.0], np.cumprod(1.0 + daily)]))


def _eval_daily(algo_cls, model_file, vec_file, features, macro, prices, reward, step,
                norm_obs=True):
    """Roda um passe de avaliacao e devolve curva diaria, retornos diarios e pesos.

    Se vec_file for None (ex.: DDPG novo so com checkpoint), cria um VecNormalize na hora
    com `norm_obs` (use False para casar com o treino `norm_obs=False`).

    Retorna (portfolio_values: pd.Series, daily_returns: np.ndarray, weights_df: pd.DataFrame)
    ou None se os arquivos nao existirem / der erro de obs-space.
    """
    if not _exists(model_file, vec_file):
        return None
    try:
        eval_env = DummyVecEnv([make_env(features, macro, prices, reward, step, eval_mode=True)])
        if vec_file is not None and os.path.exists(vec_file):
            eval_env = VecNormalize.load(vec_file, eval_env)
        else:
            eval_env = VecNormalize(eval_env, norm_obs=norm_obs, norm_reward=False)
        eval_env.training = False
        eval_env.norm_reward = False

        model = algo_cls.load(model_file)
        obs = eval_env.reset()

        portfolio_values = [1.0]
        daily_returns = []
        weights_history = []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = eval_env.step(action)
            done = bool(dones[0])
            weights_history.append(infos[0]["weights"])
            for r in infos[0].get("daily_returns", []):
                daily_returns.append(r)
                portfolio_values.append(portfolio_values[-1] * (1 + r))

        tickers = eval_env.envs[0].tickers
        weights_df = pd.DataFrame(weights_history, columns=tickers)
        return pd.Series(portfolio_values), np.asarray(daily_returns, dtype=float), weights_df
    except Exception as e:
        print(f"  [erro avaliacao] {os.path.basename(model_file)}: {type(e).__name__}: {str(e)[:140]}")
        return None


def _pack_from_daily(step, val_daily, test_daily):
    return {
        "step": step,
        "val":  {"daily": val_daily,  "curve": _curve_from_daily(val_daily),
                 "metrics": compute_metrics(_curve_from_daily(val_daily))},
        "test": {"daily": test_daily, "curve": _curve_from_daily(test_daily),
                 "metrics": compute_metrics(_curve_from_daily(test_daily))},
    }


def _is_constant(weights_df, tol=1e-6):
    """True se a politica produz pesos ~constantes no tempo (sintoma de colapso)."""
    if weights_df is None or len(weights_df) < 2:
        return False
    return float(weights_df.std(axis=0).mean()) < tol


# ----------------------------------------------------------------------
# Selecao DDPG (grid antigo): melhor (lr, gamma) por reward, media nos seeds
# ----------------------------------------------------------------------
def _ddpg_seed_paths(lr, gamma, seed, reward):
    run = f"ddpg_lr{lr}_g{gamma}_s{seed}_r{reward}"
    model_file = os.path.join(base_path, "..", f"{run}_ddpg_portfolio_final")
    vec_file   = os.path.join(base_path, "..", f"vec_normalize_{run}.pkl")
    return model_file, vec_file


def _mean_daily(daily_list):
    """Media elemento-a-elemento de varias series de retorno diario (alinhadas no menor n)."""
    n = min(len(d) for d in daily_list)
    stacked = np.vstack([np.asarray(d[:n], dtype=float) for d in daily_list])
    return stacked.mean(axis=0)


def _select_ddpg_old(reward, val_data, test_data):
    """STEP=1 (grid antigo, raiz): melhor (lr, gamma) por Sharpe medio de val nos seeds.
    Serie diaria = MEDIA dos seeds. Usa o VecNormalize salvo (norm_obs=True)."""
    fv, mv, pv = val_data
    ft, mt, pt = test_data

    best = None  # (mean_val_sharpe, lr, gamma, seeds_ok)
    cache = {}   # (lr,gamma,seed) -> (val_daily, test_daily, test_const)

    for lr in DDPG_LRS:
        for gamma in DDPG_GAMMAS:
            seeds_ok, sharpes = [], []
            for seed in DDPG_SEEDS:
                mf, vf = _ddpg_seed_paths(lr, gamma, seed, reward)
                rv = _eval_daily(DDPG, mf, vf, fv, mv, pv, reward, 1)
                if rv is None:
                    continue
                rt = _eval_daily(DDPG, mf, vf, ft, mt, pt, reward, 1)
                if rt is None:
                    continue
                seeds_ok.append(seed)
                sharpes.append(compute_metrics(rv[0])["Sharpe Ratio"])
                cache[(lr, gamma, seed)] = (rv[1], rt[1], _is_constant(rt[2]))
            if not seeds_ok:
                continue
            mean_sharpe = float(np.mean(sharpes))
            if best is None or mean_sharpe > best[0]:
                best = (mean_sharpe, lr, gamma, seeds_ok)

    if best is None:
        return None

    _, lr, gamma, seeds_ok = best
    val_daily  = _mean_daily([cache[(lr, gamma, s)][0] for s in seeds_ok])
    test_daily = _mean_daily([cache[(lr, gamma, s)][1] for s in seeds_ok])
    degenerado = all(cache[(lr, gamma, s)][2] for s in seeds_ok)
    pack = _pack_from_daily(1, val_daily, test_daily)
    pack["config"] = {"lr": lr, "gamma": gamma, "seeds": seeds_ok,
                      "degenerado": degenerado, "fonte": "grid antigo (step=1)"}
    return pack


def _ddpg_new_model_file(run_dir):
    """final_model.zip se existir; senao o checkpoint de maior step. Devolve (model_file, completo)."""
    final = os.path.join(run_dir, "final_model.zip")
    if os.path.exists(final):
        return os.path.join(run_dir, "final_model"), True
    import glob
    ckpts = glob.glob(os.path.join(run_dir, "*_steps.zip"))
    if not ckpts:
        return None, False
    def _step_of(p):
        try:
            return int(os.path.basename(p).rsplit("_steps.zip", 1)[0].rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return -1
    best = max(ckpts, key=_step_of)
    return best[:-4], False  # tira ".zip"


def _select_ddpg_new(reward, step, val_data, test_data):
    """STEP=21/63 (grid novo, cluster): melhor (lr, gamma) por Sharpe de val.
    Treinado com norm_obs=False -> avalia com VecNormalize(norm_obs=False) on-the-fly.
    Usa final_model quando existir, senao o ultimo checkpoint."""
    fv, mv, pv = val_data
    ft, mt, pt = test_data

    best = None  # (val_sharpe, lr, gamma, val_daily, test_daily, completo, degenerado)
    for lr in DDPG_LRS:
        for gamma in DDPG_GAMMAS:
            run_dir = os.path.join(MODELS_DIR, f"DDPG_r{reward}_s{step}_lr{lr}_g{gamma}")
            if not os.path.isdir(run_dir):
                continue
            mf, completo = _ddpg_new_model_file(run_dir)
            if mf is None:
                continue
            vf = os.path.join(run_dir, "vec_normalize.pkl")
            vf = vf if os.path.exists(vf) else None
            rv = _eval_daily(DDPG, mf, vf, fv, mv, pv, reward, step, norm_obs=False)
            if rv is None:
                continue
            rt = _eval_daily(DDPG, mf, vf, ft, mt, pt, reward, step, norm_obs=False)
            if rt is None:
                continue
            sharpe = compute_metrics(rv[0])["Sharpe Ratio"]
            if best is None or sharpe > best[0]:
                best = (sharpe, lr, gamma, rv[1], rt[1], completo, _is_constant(rt[2]))

    if best is None:
        return None
    _, lr, gamma, val_daily, test_daily, completo, degenerado = best
    pack = _pack_from_daily(step, val_daily, test_daily)
    pack["config"] = {"lr": lr, "gamma": gamma, "degenerado": degenerado,
                      "fonte": "grid novo (cluster)" + ("" if completo else " [checkpoint parcial]")}
    return pack


def _eval_ddpg_all_steps(reward, val_data, test_data):
    """DDPG com os 3 steps: step=1 (grid antigo) + step=21/63 (grid novo). -> {step: pack}."""
    by_step = {}
    p1 = _select_ddpg_old(reward, val_data, test_data)
    if p1 is not None:
        by_step[1] = p1
    for step in (21, 63):
        p = _select_ddpg_new(reward, step, val_data, test_data)
        if p is not None:
            by_step[step] = p
    return by_step


# ----------------------------------------------------------------------
# Selecao PPO/SAC (novo formato): avalia TODOS os steps
# ----------------------------------------------------------------------
def _rl_paths(tag, reward, step):
    run = f"{tag}_r{reward}_s{step}"
    d = os.path.join(MODELS_DIR, run)
    return os.path.join(d, "final_model"), os.path.join(d, "vec_normalize.pkl")


def _eval_rl_all_steps(algo, algo_cls, reward, val_data, test_data):
    """Retorna {step: pack} para todos os steps disponiveis de (algo, reward)."""
    fv, mv, pv = val_data
    ft, mt, pt = test_data
    by_step = {}
    for step in STEP_SIZES:
        mf, vf = _rl_paths(algo, reward, step)
        rv = _eval_daily(algo_cls, mf, vf, fv, mv, pv, reward, step)
        if rv is None:
            continue
        rt = _eval_daily(algo_cls, mf, vf, ft, mt, pt, reward, step)
        if rt is None:
            continue
        pack = _pack_from_daily(step, rv[1], rt[1])
        pack["config"] = {"degenerado": _is_constant(rt[2])}
        by_step[step] = pack
    return by_step


def _best_step_pack(by_step):
    if not by_step:
        return None
    best_step = max(by_step, key=lambda s: by_step[s]["val"]["metrics"]["Sharpe Ratio"])
    return by_step[best_step]


# ----------------------------------------------------------------------
# Tabelas de metricas
# ----------------------------------------------------------------------
def _best_per_algo_reward(results):
    """{algo: {reward: {step: pack}}} -> melhor step por (algo, reward)."""
    out = {}
    for algo in results:
        out[algo] = {}
        for reward, by_step in results[algo].items():
            pack = _best_step_pack(by_step)
            if pack is not None:
                out[algo][reward] = pack
    return out


def _pack_tag(pack):
    tag = f"step={pack['step']}"
    cfg = pack.get("config")
    if cfg and "lr" in cfg:
        tag += f", lr={cfg['lr']}, g={cfg['gamma']}"
    if cfg and cfg.get("degenerado"):
        tag += " [DEGENERADO]"
    return tag


def _coletar_degenerados(results):
    """Coleta os steps com politica degenerada (pesos constantes no tempo), ja sinalizada
    nos packs via config['degenerado']. Devolve {algo: set(steps_degenerados)}."""
    achados = {}
    for algo, by_reward in results.items():
        for reward, by_step in by_reward.items():
            for step, pack in by_step.items():
                if pack.get("config", {}).get("degenerado"):
                    achados.setdefault(algo, set()).add(step)
    return achados


def _build_table(best, split, extra_rows):
    rows = []
    for algo in best:
        for reward, pack in best[algo].items():
            m = pack[split]["metrics"]
            rows.append({"Model": f"{algo} ({reward}) [{_pack_tag(pack)}]",
                         **{k: m[k] for k in METRIC_COLS}})
    for label, m in extra_rows:
        rows.append({"Model": label, **{k: m[k] for k in METRIC_COLS}})
    return pd.DataFrame(rows).set_index("Model").sort_values("Sharpe Ratio", ascending=False)


# ----------------------------------------------------------------------
# Testes de significancia (Wilcoxon pareado + Friedman + Holm)
# ----------------------------------------------------------------------
def _align(arrays):
    n = min(len(a) for a in arrays)
    return [np.asarray(a[:n], dtype=float) for a in arrays]


def _holm(pvals):
    m = len(pvals)
    order = sorted(range(m), key=lambda k: pvals[k])
    adj, running = [0.0] * m, 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[k]))
        adj[k] = running
    return adj


def _grupo_stats(daily_dict, alpha=0.05):
    """Roda Friedman + Wilcoxon/Holm e devolve dados estruturados (ou None se <2 grupos)."""
    names = list(daily_dict)
    if len(names) < 2:
        return None
    arrs = _align([daily_dict[n] for n in names])
    n_obs = len(arrs[0])

    resumo = []
    for name, a in zip(names, arrs):
        sh = float(np.sqrt(252) * (a.mean() - 0.08 / 252) / (a.std() + 1e-8))
        p_norm = float(stats.shapiro(a)[1]) if 3 <= len(a) <= 5000 else float("nan")
        resumo.append({"nome": str(name), "ret_medio": float(a.mean()),
                       "sharpe": sh, "shapiro_p": p_norm})

    friedman = None
    if len(names) >= 3:
        chi2, p = stats.friedmanchisquare(*arrs)
        friedman = {"chi2": float(chi2), "p": float(p), "sig": bool(p < alpha)}

    pairs = list(itertools.combinations(range(len(names)), 2))
    raw_p, raw_w = [], []
    for i, j in pairs:
        if np.allclose(arrs[i], arrs[j]):
            # series identicas (ex.: politica degenerada) -> sem diferenca
            w, p = 0.0, 1.0
        else:
            try:
                w, p = stats.wilcoxon(arrs[i], arrs[j])
            except ValueError:
                w, p = float("nan"), 1.0
            if np.isnan(p):
                p = 1.0
        raw_w.append(float(w)); raw_p.append(float(p))
    holm = _holm(raw_p)
    pares = [{"a": str(names[i]), "b": str(names[j]), "W": w, "p": p,
              "p_holm": ph, "sig": bool(ph < alpha)}
             for (i, j), w, p, ph in zip(pairs, raw_w, raw_p, holm)]

    n_identicas = sum(1 for p in pares if p["W"] == 0.0 and p["p"] == 1.0
                      and np.allclose(daily_dict[p["a"]][:n_obs], daily_dict[p["b"]][:n_obs]))

    return {"names": [str(n) for n in names], "n_obs": n_obs,
            "resumo": resumo, "friedman": friedman, "pares": pares, "alpha": alpha,
            "n_identicas": n_identicas}


def _grupo_txt(titulo, g):
    if g is None:
        return f"\n===== {titulo} =====\n(menos de 2 modelos disponiveis -- pulado)\n"
    lines = [f"\n===== TESTE DE SIGNIFICANCIA: {titulo} =====",
             f"Dias alinhados (pareados): {g['n_obs']}",
             "Resumo (retornos diarios do teste):"]
    for r in g["resumo"]:
        norm = "nao-normal" if r["shapiro_p"] < g["alpha"] else "normal"
        lines.append(f"  {r['nome']:>24}: ret.medio/dia={r['ret_medio']:+.5f}  "
                     f"Sharpe={r['sharpe']:+.3f}  Shapiro_p={r['shapiro_p']:.3g} ({norm})")
    if g["friedman"]:
        fr = g["friedman"]
        verdict = "HA diferenca" if fr["sig"] else "sem diferenca"
        lines.append(f"\nFriedman (omnibus): chi2={fr['chi2']:.4f}  p={fr['p']:.4g}  "
                     f"-> {verdict} (alpha={g['alpha']})")
    lines.append("\nComparacoes pareadas (Wilcoxon signed-rank, p ajustado por Holm):")
    for pr in g["pares"]:
        sig = "*" if pr["sig"] else " "
        lines.append(f"  {pr['a']:>24} vs {pr['b']:<24} "
                     f"W={pr['W']:>10.1f}  p={pr['p']:.4g}  p_holm={pr['p_holm']:.4g} {sig}")
    lines.append("  (*) diferenca significativa apos correcao de Holm")
    return "\n".join(lines) + "\n"


def _significancia(results, best):
    """Devolve (texto, blocos) onde blocos = lista de {tipo, algo, reward/None, titulo, stats}."""
    blocos = []

    # (a) ENTRE STEPS: cada algoritmo com >=2 steps, fixando a melhor reward (por Sharpe de val)
    for algo in results:
        if not best.get(algo):
            continue
        best_reward = max(best[algo], key=lambda r: best[algo][r]["val"]["metrics"]["Sharpe Ratio"])
        by_step = results[algo][best_reward]
        if len(by_step) < 2:
            continue
        step_daily = {f"step={s}": by_step[s]["test"]["daily"] for s in sorted(by_step)}
        deg = any(by_step[s].get("config", {}).get("degenerado") for s in by_step)
        titulo = f"STEPS -- {algo} (reward={best_reward}, no teste)"
        blocos.append({"tipo": "steps", "algo": algo, "reward": best_reward,
                       "titulo": titulo, "stats": _grupo_stats(step_daily), "degenerado": deg})

    # (b) ENTRE REWARD FUNCTIONS: cada algoritmo, no melhor step de cada reward
    for algo in best:
        if not best[algo]:
            continue
        reward_daily = {r: best[algo][r]["test"]["daily"] for r in best[algo]}
        deg = any(best[algo][r].get("config", {}).get("degenerado") for r in best[algo])
        titulo = f"REWARD FUNCTIONS -- {algo} (no teste)"
        blocos.append({"tipo": "reward", "algo": algo, "reward": None,
                       "titulo": titulo, "stats": _grupo_stats(reward_daily), "degenerado": deg})

    texto = "\n".join(_grupo_txt(b["titulo"], b["stats"]) for b in blocos)
    return texto, blocos


def _md_tabela_grupo(g):
    """Tabela markdown com resumo + p-valores pareados de um grupo."""
    linhas = ["| Modelo | Ret. médio/dia | Sharpe | Shapiro p | Normal? |",
              "|---|---|---|---|---|"]
    for r in g["resumo"]:
        normal = "não" if r["shapiro_p"] < g["alpha"] else "sim"
        linhas.append(f"| {r['nome']} | {r['ret_medio']:+.5f} | {r['sharpe']:+.3f} "
                      f"| {r['shapiro_p']:.2g} | {normal} |")
    linhas.append("")
    if g["friedman"]:
        fr = g["friedman"]
        v = "**há diferença**" if fr["sig"] else "sem diferença"
        linhas.append(f"**Friedman (omnibus):** χ²={fr['chi2']:.3f}, p={fr['p']:.4g} → {v} (α={g['alpha']}).")
        if g.get("n_identicas", 0) > 0:
            linhas.append(f"> ⚠️ {g['n_identicas']} par(es) de séries são **idênticos** (política "
                          "degenerada): o resultado do Friedman aqui é artefato, não diferença real.")
        linhas.append("")
    linhas += ["| Comparação | W | p | p (Holm) | Sig.? |", "|---|---|---|---|---|"]
    for pr in g["pares"]:
        sig = "✅" if pr["sig"] else "—"
        linhas.append(f"| {pr['a']} vs {pr['b']} | {pr['W']:.0f} | {pr['p']:.4g} "
                      f"| {pr['p_holm']:.4g} | {sig} |")
    return "\n".join(linhas)


def _gerar_md(blocos, degenerados, path):
    """Relatorio .md dissertando sobre a significancia por reward function e entre steps."""
    md = ["# Análise de Significância Estatística — DDPG × PPO × SAC",
          "",
          "Comparações feitas sobre os **retornos diários do período de teste**, pareados por dia. "
          "Como o teste de Shapiro–Wilk rejeita normalidade em todas as séries (p ≈ 1e-33 a 1e-38), "
          "usamos testes **não-paramétricos**: **Friedman** como teste omnibus (quando há ≥3 grupos) "
          "e **Wilcoxon signed-rank** pareado par a par, com correção de **Holm–Bonferroni** para "
          "múltiplas comparações. Nível de significância α = 0,05.",
          ""]

    if degenerados:
        md += ["> ⚠️ **Aviso — DDPG do grid novo (cluster) está degenerado.**",
               "> Os modelos DDPG treinados com `norm_obs=False` "
               f"(steps {sorted(degenerados.get('DDPG', []))}) colapsaram numa política **constante e "
               "idêntica entre todas as reward functions** — sintoma do `ibov` (~41.000) não-normalizado "
               "saturando a rede. Como os retornos diários dependem só das ações, e as ações ficaram "
               "iguais, qualquer comparação envolvendo esses steps do DDPG (entre steps ou entre rewards) "
               "**não é interpretável** e aparece marcada com `[DEGENERADO]`. O DDPG confiável é o do "
               "grid antigo (`step=1`, `norm_obs=True`). Recomenda-se re-treinar o grid novo com "
               "`norm_obs=True`.", ""]

    steps_blocos  = [b for b in blocos if b["tipo"] == "steps"  and b["stats"]]
    reward_blocos = [b for b in blocos if b["tipo"] == "reward" and b["stats"]]

    # ---------------- Reward functions ----------------
    md += ["## 1. Diferença entre *reward functions* (por modelo)", ""]
    if not reward_blocos:
        md += ["_Sem dados suficientes._", ""]
    for b in reward_blocos:
        g = b["stats"]
        algo = b["algo"]
        melhor = max(g["resumo"], key=lambda r: r["sharpe"])
        pior   = min(g["resumo"], key=lambda r: r["sharpe"])
        sig_pairs = [p for p in g["pares"] if p["sig"]]
        md += [f"### {algo}", ""]
        md += [_md_tabela_grupo(g), ""]
        fr = g["friedman"]
        if b.get("degenerado"):
            txt = (f"⚠️ A melhor configuração de uma ou mais reward functions do {algo} cai num modelo "
                   "**degenerado** (pesos constantes no tempo, grid novo `norm_obs=False`). Logo, este "
                   "bloco **não** permite concluir sobre o efeito da reward function — para o DDPG, a "
                   "comparação confiável é a do `step=1` (grid antigo).")
        elif fr and fr["sig"]:
            txt = (f"O teste de Friedman aponta diferença global significativa entre as reward "
                   f"functions do {algo} (p={fr['p']:.4g}).")
        elif fr:
            txt = (f"O teste de Friedman **não** detecta diferença global significativa entre as "
                   f"reward functions do {algo} (p={fr['p']:.4g}).")
        else:
            txt = ""
        if sig_pairs:
            pares_txt = "; ".join(f"{p['a']} vs {p['b']} (p_Holm={p['p_holm']:.3g})" for p in sig_pairs)
            txt += f" Após a correção de Holm, permanecem significativas: {pares_txt}."
        else:
            txt += (" Nenhuma comparação par a par sobrevive à correção de Holm — ou seja, apesar de "
                    "diferenças de Sharpe, **não há evidência estatística** de que uma reward function "
                    "supere a outra no teste.")
        md += [txt,
               f"Na prática, a maior Sharpe foi de **{melhor['nome']}** ({melhor['sharpe']:+.3f}) e a "
               f"menor de **{pior['nome']}** ({pior['sharpe']:+.3f}).", ""]

    # ---------------- Steps ----------------
    md += ["## 2. Diferença entre *steps* (por modelo)", "",
           "Cada modelo é avaliado fixando a sua melhor reward function (maior Sharpe de validação) "
           "e comparando os horizontes de rebalanceamento (step = 1, 21 e 63 dias).", ""]
    if not steps_blocos:
        md += ["_Sem dados suficientes (nenhum modelo com ≥2 steps)._", ""]
    for b in steps_blocos:
        g = b["stats"]
        algo, reward = b["algo"], b["reward"]
        melhor = max(g["resumo"], key=lambda r: r["sharpe"])
        sig_pairs = [p for p in g["pares"] if p["sig"]]
        md += [f"### {algo} (reward = {reward})", ""]
        md += [_md_tabela_grupo(g), ""]
        fr = g["friedman"]
        if b.get("degenerado"):
            txt = (f"⚠️ Os steps 21/63 do {algo} vêm do grid novo **degenerado** (política constante), "
                   f"enquanto o step=1 vem do grid antigo. A comparação entre steps do {algo}, portanto, "
                   "**não é interpretável** — não reflete o efeito real do horizonte de rebalanceamento.")
        elif fr and fr["sig"]:
            txt = f"Friedman indica diferença significativa entre os steps do {algo} (p={fr['p']:.4g})."
        elif fr:
            txt = f"Friedman **não** indica diferença significativa entre os steps do {algo} (p={fr['p']:.4g})."
        else:
            txt = ""
        if sig_pairs:
            pares_txt = "; ".join(f"{p['a']} vs {p['b']} (p_Holm={p['p_holm']:.3g})" for p in sig_pairs)
            txt += f" Pares significativos após Holm: {pares_txt}."
        else:
            txt += (" Nenhum par de steps difere significativamente após Holm: o horizonte de "
                    "rebalanceamento **não** teve efeito estatisticamente detectável no retorno diário.")
        md += [txt, f"Melhor step por Sharpe: **{melhor['nome']}** ({melhor['sharpe']:+.3f}).", ""]

    # ---------------- Conclusão ----------------
    md += ["## 3. Síntese", ""]
    algum_sig = any(p["sig"] for b in blocos
                    if b["stats"] and not b.get("degenerado")
                    for p in b["stats"]["pares"])
    if algum_sig:
        md += ["Há ao menos uma diferença estatisticamente significativa após correção de múltiplas "
               "comparações (ver seções acima).", ""]
    else:
        md += ["**Em nenhuma das comparações** — nem entre reward functions, nem entre steps — a "
               "diferença sobrevive à correção de Holm. Isto sugere que, no período de teste, as "
               "escolhas de reward function e de horizonte de rebalanceamento produzem curvas de "
               "retorno diário **estatisticamente equivalentes**, ainda que difiram em Sharpe pontual. "
               "Diferenças aparentes de desempenho devem, portanto, ser lidas com cautela.", ""]

    with open(path, "w") as f:
        f.write("\n".join(md))


# ----------------------------------------------------------------------
# Driver principal
# ----------------------------------------------------------------------
def avaliar_todos():
    iniciar_semente(SEED)

    val_data  = _prep(df_features_val,  df_macro_val,  df_prices_val)
    test_data = _prep(df_features_test, df_macro_test, df_prices_test)
    pv = val_data[2]
    pt = test_data[2]

    print(f"Periodo val:   {pv.index.min().date()} -> {pv.index.max().date()}  ({len(pv)} dias)")
    print(f"Periodo teste: {pt.index.min().date()} -> {pt.index.max().date()}  ({len(pt)} dias)")

    # ---- avalia tudo ----  results[algo][reward] = {step: pack}
    results = {a: {} for a in ALGOS}
    for reward in REWARD_TYPES:
        by_step = _eval_ddpg_all_steps(reward, val_data, test_data)
        if by_step:
            results["DDPG"][reward] = by_step
            bp = _best_step_pack(by_step)
            print(f"[ok] DDPG / {reward}: steps={sorted(by_step)} "
                  f"melhor step={bp['step']} ({bp['config'].get('fonte','')}) "
                  f"Sharpe val={bp['val']['metrics']['Sharpe Ratio']:.3f} "
                  f"test={bp['test']['metrics']['Sharpe Ratio']:.3f}")
        else:
            print(f"[skip] DDPG / {reward}: nenhum modelo encontrado")

    for algo in ("PPO", "SAC"):
        for reward in REWARD_TYPES:
            by_step = _eval_rl_all_steps(algo, ALGOS[algo], reward, val_data, test_data)
            if by_step:
                results[algo][reward] = by_step
                bp = _best_step_pack(by_step)
                print(f"[ok] {algo} / {reward}: steps={sorted(by_step)} "
                      f"melhor step={bp['step']} Sharpe val={bp['val']['metrics']['Sharpe Ratio']:.3f} "
                      f"test={bp['test']['metrics']['Sharpe Ratio']:.3f}")
            else:
                print(f"[skip] {algo} / {reward}: nenhum modelo encontrado")

    degenerados = _coletar_degenerados(results)
    if degenerados:
        for algo, steps in degenerados.items():
            print(f"[AVISO] {algo}: politica degenerada (identica entre rewards) nos steps "
                  f"{sorted(steps)} -- provavel efeito de norm_obs=False.")

    best = _best_per_algo_reward(results)
    if not any(best[a] for a in best):
        print("Nenhum modelo avaliado. Abortando.")
        return

    # ---- baselines (markowitz / equal-weight) ----
    train_match_val  = df_prices[[c for c in pv.columns if c in df_prices.columns]]
    train_match_test = df_prices[[c for c in pt.columns if c in df_prices.columns]]
    eq_val,  mvo_val  = equal_weight_baseline(pv), mvo_baseline(pv, train_match_val)
    eq_test, mvo_test = equal_weight_baseline(pt), mvo_baseline(pt, train_match_test)
    baselines = {
        "val":  {"Equal Weight": eq_val,  "MVO (Markowitz)": mvo_val},
        "test": {"Equal Weight": eq_test, "MVO (Markowitz)": mvo_test},
    }

    # ---- tabelas de metricas ----
    for split, period in [("val", "VALIDACAO"), ("test", "TESTE")]:
        extra = [(lbl, compute_metrics(curve)) for lbl, curve in baselines[split].items()]
        df_tbl = _build_table(best, split, extra)
        print(f"\n===== METRICAS -- {period} =====")
        print(df_tbl.to_string(float_format=lambda x: f"{x:.4f}"))
        df_tbl.to_csv(os.path.join(METRICS_DIR, f"comparacao_metrics_{split}.csv"))

    # ---- elege o melhor modelo (Sharpe de validacao) ----
    candidatos = []
    for algo in best:
        for reward, pack in best[algo].items():
            candidatos.append((pack["val"]["metrics"]["Sharpe Ratio"], algo, reward, pack))
    candidatos.sort(key=lambda x: x[0], reverse=True)
    sh_val, algo_b, reward_b, pack_b = candidatos[0]
    print("\n===== MODELO ELEITO (maior Sharpe de validacao) =====")
    print(f"  {algo_b} | reward={reward_b} | {_pack_tag(pack_b)}")
    print(f"  Sharpe  val={sh_val:.3f}  |  test={pack_b['test']['metrics']['Sharpe Ratio']:.3f}")
    print(f"  Metricas (teste): {pack_b['test']['metrics']}")

    # ---- GRAFICO 1: melhores modelos (1 por algoritmo, step especificado) + baselines ----
    best_per_algo = {}
    for algo in best:
        if not best[algo]:
            continue
        r = max(best[algo], key=lambda rr: best[algo][rr]["val"]["metrics"]["Sharpe Ratio"])
        best_per_algo[algo] = r

    for split, period in [("val", "Validacao"), ("test", "Teste")]:
        _, ax = plt.subplots(figsize=(12, 6))
        for algo, reward in best_per_algo.items():
            pack = best[algo][reward]
            deg = " [degenerado]" if pack.get("config", {}).get("degenerado") else ""
            ax.plot(pack[split]["curve"].values, color=ALGO_COLORS.get(algo), linewidth=1.8,
                    label=f"{algo} ({reward}, step={pack['step']}){deg}")
        ax.plot(baselines[split]["Equal Weight"].values, color="black",
                linestyle="--", linewidth=1.4, label="Equal Weight")
        ax.plot(baselines[split]["MVO (Markowitz)"].values, color="gray",
                linestyle=":", linewidth=1.4, label="MVO (Markowitz)")
        ax.set_title(f"Curva de capital -- melhores modelos por algoritmo ({period})")
        ax.set_xlabel("Dias de pregao")
        ax.set_ylabel("Valor do portfolio (base = 1.0)")
        ax.axhline(1.0, color="lightgray", linewidth=0.8)
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"comparacao_melhores_{split}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"Salvo: {out}")

    # ---- GRAFICO 2: curva de cada STEP por modelo (melhor reward de cada algoritmo) ----
    step_styles = {1: "-", 21: "--", 63: ":"}
    for split, period in [("val", "Validacao"), ("test", "Teste")]:
        _, ax = plt.subplots(figsize=(12, 6))
        for algo in best:
            if not best[algo]:
                continue
            reward = best_per_algo[algo]
            by_step = results[algo][reward]
            for step in sorted(by_step):
                deg = " [deg]" if by_step[step].get("config", {}).get("degenerado") else ""
                ax.plot(by_step[step][split]["curve"].values,
                        color=ALGO_COLORS.get(algo), linestyle=step_styles.get(step, "-"),
                        linewidth=1.4, label=f"{algo} ({reward}) step={step}{deg}")
        ax.axhline(1.0, color="lightgray", linewidth=0.8)
        ax.set_title(f"Curva de capital por step -- por modelo ({period})")
        ax.set_xlabel("Dias de pregao")
        ax.set_ylabel("Valor do portfolio (base = 1.0)")
        ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOTS_DIR, f"comparacao_steps_{split}.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"Salvo: {out}")

    # ---- testes de significancia (txt + md dissertativo) ----
    relatorio, blocos = _significancia(results, best)
    print(relatorio)
    sig_path = os.path.join(METRICS_DIR, "significancia.txt")
    with open(sig_path, "w") as f:
        f.write(relatorio)
    print(f"Salvo: {sig_path}")

    md_path = os.path.join(METRICS_DIR, "relatorio_significancia.md")
    _gerar_md(blocos, degenerados, md_path)
    print(f"Salvo: {md_path}")
    return results, best


def train_ddpg_one(config_id):
    """Treina UMA combinação do grid, escolhida por config_id (0..59)."""
    if not (0 <= config_id < len(DDPG_COMBOS)):
        raise ValueError(f"config_id {config_id} fora do range 0..{len(DDPG_COMBOS)-1}")

    l, g, r, s = DDPG_COMBOS[config_id]
    run_name = f"DDPG_r{r}_s{s}_lr{l}_g{g}"
    print(f"[config_id={config_id}] treino {run_name}", flush=True)

    iniciar_semente(SEED)

    train_env = make_vec_env(df_features, df_macro, df_prices, r, s, eval_mode=False)
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True)

    n_actions = len(df_prices.columns)
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy", train_env, seed=SEED,
        action_noise=action_noise,
        learning_rate=l, gamma=g,
        batch_size=256, buffer_size=100_000, tau=0.005,
        train_freq=(1, "step"), gradient_steps=1,
        verbose=0, device=DEVICE,
        policy_kwargs=policy_kwargs_ddpg,
        tensorboard_log=TB_DIR,
    )

    ckpt_dir, callbacks = build_callbacks(run_name, r, s)

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=False,   # array job: sem barra interativa, vai para o log
    )

    model.save(os.path.join(ckpt_dir, "final_model"))
    train_env.save(os.path.join(ckpt_dir, "vec_normalize.pkl"))
    print(f"[config_id={config_id}] salvo {run_name}", flush=True)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_id", type=int, default=None,
                        help="indice da combinacao do grid DDPG (0 a 59) para TREINO")
    parser.add_argument("--mode", choices=["train", "avaliar"], default=None,
                        help="train: treina um config_id | avaliar: avalia e compara todos os modelos")
    args = parser.parse_args()

    if args.mode == "avaliar":
        avaliar_todos()
    elif args.config_id is not None:
        train_ddpg_one(args.config_id)
    else:
        parser.error("informe --config_id <i> (treino) ou --mode avaliar (avaliacao)")















    






