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
from environment.evaluate import compute_metrics, count_operations, compute_operations_metrics

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
    parser.add_argument("--config_id", type=int, required=True,
                        help="índice da combinação do grid (0 a 59)")
    args = parser.parse_args()
    train_ddpg_one(args.config_id)















    






