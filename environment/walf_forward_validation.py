"""
Walk-forward analysis (estilo do paper) -- pipeline em 2 ESTAGIOS.

Esquema ANCORA EXPANSIVEL, janela de teste de 1 ano. Linha do tempo completa
(2000-2024) reconstruida concatenando master_*_{train,val,test}.csv:

    Fold 1:  train 2000-2006 | test 2007
    ...
    Fold 18: train 2000-2023 | test 2024     (18 folds, teste 2007..2024)

Para NAO confundir SELECAO de modelo com ABLACAO de contexto, o contexto NAO
entra no pool de selecao. O fluxo (igual em espirito ao paper):

  ESTAGIO 1 (WF_STAGE=1): treina so os candidatos COM contexto
      {DDPG, PPO, SAC} x 5 rewards x N seeds, em 2000..(ano-1).
      -> 18 x 3 x 5 x N treinos.

  SELECAO (--mode selecionar): avalia os candidatos com contexto no TREINO
      (media das seeds) e elege o melhor (algo, reward) por fold
      (SELECTION_METRIC). Grava metrics/walkforward/winners.json.

  ESTAGIO 2 (WF_STAGE=2): para o vencedor de cada fold, retreina a MESMA
      config (algo, reward) SEM contexto (ablacao pareada).
      -> 18 x 1 x N treinos. Le winners.json.

  AVALIACAO (--mode avaliar): no ano de teste compara, por fold, o vencedor
      COM contexto vs a versao SEM contexto vs benchmarks Equal-Weight e
      Markowitz (MVO). Salva as metricas de todos.

Seeds incrementais via WF_SEEDS (ver abaixo). Selecao do estagio 1 -> testa
poucos modelos sem contexto -> ablacao limpa e barata.

IMPORTANTE: todos os envs usam VecNormalize(norm_obs=True). norm_obs=False
degenera as politicas (macro 'ibov' satura a rede). Ver ddpg-norm-obs-degenerado.
"""

import os
import json
import argparse
import itertools

import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import DDPG, PPO, SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from environment.environment import PortfolioEnv
from environment.train import pre_processamento, iniciar_semente, RewardLoggerCallback
from environment.evaluate import (
    compute_metrics,
    compute_operations_metrics,
    equal_weight_baseline,
    mvo_baseline,
)

# ----------------------------------------------------------------------
# Hiperparametros de rede (iguais aos do script_final.py / train.py)
# ----------------------------------------------------------------------
policy_kwargs_ppo = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256, 128], vf=[256, 128]))
policy_kwargs_sac = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256, 128], qf=[256, 128]))
policy_kwargs_ddpg = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256, 128], qf=[256, 128]))

ALGO_CLS = {"DDPG": DDPG, "PPO": PPO, "SAC": SAC}

# ----------------------------------------------------------------------
# Constantes de execucao (override por variavel de ambiente no cluster)
# ----------------------------------------------------------------------
TIMESTEPS = int(os.environ.get("WF_TIMESTEPS", 300_000))
N_ENVS    = int(os.environ.get("WF_N_ENVS", 8))   # Mac local: 1 | Apuana: 8
DEVICE    = os.environ.get("WF_DEVICE", "auto")   # DDPG/SAC: auto (GPU) | PPO sempre cpu
STAGE     = os.environ.get("WF_STAGE", "1")       # "1" = treina com contexto | "2" = ablacao sem contexto

SELECTION_METRIC = "Sharpe Ratio"   # criterio do "melhor no treino"
STEP_SIZE = 1
WARMUP = 30   # pregoes do ano anterior anexados ao teste (contexto/retornos recentes)

base_path   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(base_path, "models", "walkforward")
PLOTS_DIR   = os.path.join(base_path, "..", "plots", "walkforward")
METRICS_DIR = os.path.join(base_path, "..", "metrics", "walkforward")
TB_DIR      = os.path.join(base_path, "..", "tensorboard", "walkforward")
for d in (MODELS_DIR, PLOTS_DIR, METRICS_DIR, TB_DIR):
    os.makedirs(d, exist_ok=True)
WINNERS_PATH = os.path.join(METRICS_DIR, "winners.json")

# ----------------------------------------------------------------------
# Grade walk-forward
# ----------------------------------------------------------------------
TEST_YEARS = list(range(2007, 2025))                        # 2007..2024 (18 folds)
ALGOS_LIST = ["DDPG", "PPO", "SAC"]
REWARD_TYPES = ["log-retorno", "huang-return", "sortino", "omega", "calmar"]
# Seeds via env var (incremental): fase 1 = WF_SEEDS=30; fase 2 = WF_SEEDS=42,123.
# Na avaliacao/selecao, exporte TODAS as seeds ja treinadas (ex.: WF_SEEDS=30,42,123)
# -- a media usa apenas as seeds presentes no disco.
SEEDS = [int(s) for s in os.environ.get("WF_SEEDS", "30,42,123").split(",")]


def run_name(algo, use_context, reward, seed, year):
    ctx = "ctx" if use_context else "noctx"
    return f"{algo}_{ctx}_r{reward}_seed{seed}_fold{year}"


def model_label(algo, use_context):
    return algo if use_context else f"{algo} (no context)"


def fold_dir(year, name):
    return os.path.join(MODELS_DIR, f"fold{year}", name)


# ----------------------------------------------------------------------
# Configs por estagio (definem o config_id do job-array)
# ----------------------------------------------------------------------
def stage1_configs():
    """COM contexto: (year, algo, reward, seed) para todos os algos/rewards."""
    return list(itertools.product(TEST_YEARS, ALGOS_LIST, REWARD_TYPES, SEEDS))


def load_winners():
    if not os.path.exists(WINNERS_PATH):
        raise FileNotFoundError(
            f"{WINNERS_PATH} nao existe. Rode '--mode selecionar' apos o estagio 1.")
    with open(WINNERS_PATH) as f:
        return json.load(f)   # {"2007": {"algo": "PPO", "reward": "calmar"}, ...}


def stage2_configs():
    """SEM contexto: so o vencedor (algo, reward) de cada fold x seeds."""
    winners = load_winners()
    cfgs = []
    for year in TEST_YEARS:
        w = winners.get(str(year))
        if w is None:
            continue
        for seed in SEEDS:
            cfgs.append((year, w["algo"], w["reward"], seed))
    return cfgs


def get_configs():
    return stage1_configs() if STAGE == "1" else stage2_configs()


# ----------------------------------------------------------------------
# Dados: reconstroi a linha do tempo completa 2000-2024
# ----------------------------------------------------------------------
def _load_full_timeline():
    root = os.path.join(base_path, "..", "data", "processed", "data_train_val")

    def _concat(kind):
        parts = []
        for split in ("train", "val", "test"):
            f = os.path.join(root, f"master_{kind}_{split}.csv")
            parts.append(pd.read_csv(f, index_col=0, parse_dates=True))
        df = pd.concat(parts).sort_index()
        return df[~df.index.duplicated(keep="first")]

    feats, macro, prices = _concat("features"), _concat("macro"), _concat("prices")
    feats, prices = pre_processamento(feats, prices)   # remove tickers com NaN
    macro = macro.loc[feats.index]
    return feats, macro, prices


def slice_train(feats, macro, prices, test_year):
    mask = feats.index.year < test_year
    return feats.loc[mask], macro.loc[mask], prices.loc[mask]


def slice_test(feats, macro, prices, test_year, warmup=WARMUP):
    """Ano de teste + `warmup` pregoes do ano anterior (contexto inicial)."""
    pos = np.flatnonzero(feats.index.year == test_year)
    if len(pos) == 0:
        return None
    start = max(0, pos[0] - warmup)
    sl = slice(start, pos[-1] + 1)
    return feats.iloc[sl], macro.iloc[sl], prices.iloc[sl]


# ----------------------------------------------------------------------
# Ambiente
# ----------------------------------------------------------------------
def _env_factory(feats, macro, prices, reward, eval_mode, use_context):
    def _init():
        return PortfolioEnv(feats, macro, prices, reward_type=reward,
                            step_size=STEP_SIZE, eval_mode=eval_mode,
                            use_context=use_context)
    return _init


def make_train_env(feats, macro, prices, reward, use_context):
    factories = [_env_factory(feats, macro, prices, reward, False, use_context) for _ in range(N_ENVS)]
    vec = SubprocVecEnv(factories) if N_ENVS > 1 else DummyVecEnv(factories)
    return VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)


def build_model(algo, env, seed):
    n_actions = env.action_space.shape[0]
    if algo == "DDPG":
        noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        return DDPG("MlpPolicy", env, seed=seed, action_noise=noise,
                    learning_rate=1e-4, gamma=0.98, batch_size=256, buffer_size=100_000,
                    tau=0.005, train_freq=(1, "step"), gradient_steps=1, verbose=0,
                    device=DEVICE, policy_kwargs=policy_kwargs_ddpg, tensorboard_log=TB_DIR)
    if algo == "SAC":
        return SAC("MlpPolicy", env, seed=seed, learning_rate=1e-4, gamma=0.99,
                   batch_size=256, buffer_size=100_000, tau=0.001, train_freq=(1, "step"),
                   gradient_steps=1, verbose=0, device=DEVICE, policy_kwargs=policy_kwargs_sac,
                   tensorboard_log=TB_DIR)
    if algo == "PPO":
        return PPO("MlpPolicy", env, seed=seed, learning_rate=1e-4, gamma=0.99,
                   n_steps=1024, batch_size=64, clip_range=0.2, n_epochs=15, ent_coef=0.05,
                   verbose=0, device="cpu", policy_kwargs=policy_kwargs_ppo, tensorboard_log=TB_DIR)
    raise ValueError(f"algo desconhecido: {algo}")


# ----------------------------------------------------------------------
# TREINO de um config_id (job-array). use_context vem do estagio.
# ----------------------------------------------------------------------
def train_one(config_id):
    configs = get_configs()
    if config_id < 0:
        raise ValueError(f"config_id {config_id} invalido")
    if config_id >= len(configs):
        # array super-dimensionado (ex.: menos vencedores que o tamanho do array): no-op limpo
        print(f"[stage{STAGE} config_id={config_id}] >= {len(configs)} configs -- nada a treinar (skip)",
              flush=True)
        return

    year, algo, reward, seed = configs[config_id]
    use_context = (STAGE == "1")
    name = run_name(algo, use_context, reward, seed, year)
    print(f"[stage{STAGE} config_id={config_id}] {name}", flush=True)

    feats, macro, prices = _load_full_timeline()
    ft, mt, pt = slice_train(feats, macro, prices, year)
    print(f"[config_id={config_id}] train {ft.index.min().date()} -> {ft.index.max().date()} "
          f"({len(ft)} pregoes) | test={year}", flush=True)

    iniciar_semente(seed)
    env = make_train_env(ft, mt, pt, reward, use_context)
    model = build_model(algo, env, seed)

    out = fold_dir(year, name)
    os.makedirs(out, exist_ok=True)
    ckpt = CheckpointCallback(save_freq=max(50_000 // N_ENVS, 1), save_path=out, name_prefix=name)
    logger = RewardLoggerCallback(reward_type=name, seed=seed, plots_dir=PLOTS_DIR, rolling_window=500)

    model.learn(total_timesteps=TIMESTEPS, callback=CallbackList([ckpt, logger]),
                tb_log_name=name, progress_bar=False)

    model.save(os.path.join(out, "final_model"))
    env.save(os.path.join(out, "vec_normalize.pkl"))
    print(f"[stage{STAGE} config_id={config_id}] salvo em {out}", flush=True)


# ----------------------------------------------------------------------
# Helpers de avaliacao (rollout deterministico, curva DIARIA)
# ----------------------------------------------------------------------
def _curve_from_daily(daily):
    daily = np.asarray(daily, dtype=float)
    return pd.Series(np.concatenate([[1.0], np.cumprod(1.0 + daily)]))


def _mean_daily(daily_list):
    n = min(len(d) for d in daily_list)
    return np.vstack([np.asarray(d[:n], dtype=float) for d in daily_list]).mean(axis=0)


def _mean_weights(wdf_list):
    n = min(len(w) for w in wdf_list)
    aligned = [w.iloc[:n].reset_index(drop=True) for w in wdf_list]
    return sum(aligned) / len(aligned)


def _is_constant(weights_df, tol=1e-6):
    if weights_df is None or len(weights_df) < 2:
        return False
    return float(weights_df.std(axis=0).mean()) < tol


def _rollout(algo, use_context, reward, name, year, feats, macro, prices):
    """Passe deterministico. Retorna (daily_returns, weights_df) ou None."""
    out = fold_dir(year, name)
    model_file = os.path.join(out, "final_model")
    vec_file = os.path.join(out, "vec_normalize.pkl")
    if not (os.path.exists(model_file + ".zip") or os.path.exists(model_file)):
        return None
    try:
        env = DummyVecEnv([_env_factory(feats, macro, prices, reward, True, use_context)])
        env = VecNormalize.load(vec_file, env)
        env.training = False
        env.norm_reward = False
        model = ALGO_CLS[algo].load(model_file)
        obs = env.reset()
        daily, weights = [], []
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = env.step(action)
            done = bool(dones[0])
            weights.append(infos[0]["weights"])
            daily.extend(infos[0].get("daily_returns", []))
        wdf = pd.DataFrame(weights, columns=env.envs[0].tickers)
        return np.asarray(daily, dtype=float), wdf
    except Exception as e:
        print(f"  [erro rollout] {name}: {type(e).__name__}: {str(e)[:120]}")
        return None


def _eval_over_seeds(algo, use_context, reward, year, feats, macro, prices):
    """Avalia as seeds presentes de um candidato e devolve metricas da MEDIA."""
    dailies, wdfs = [], []
    for seed in SEEDS:
        res = _rollout(algo, use_context, reward, run_name(algo, use_context, reward, seed, year),
                       year, feats, macro, prices)
        if res is not None:
            dailies.append(res[0])
            wdfs.append(res[1])
    if not dailies:
        return None
    curve = _curve_from_daily(_mean_daily(dailies))
    metrics = compute_metrics(curve)
    mean_w = _mean_weights(wdfs)
    metrics.update(compute_operations_metrics(mean_w, 0.0001, STEP_SIZE))
    metrics["degenerado"] = any(_is_constant(w) for w in wdfs)
    metrics["n_seeds"] = len(dailies)
    return {"metrics": metrics, "curve": curve}


METRIC_COLS = ["Total Return", "Annual Return", "Volatility", "Sharpe Ratio", "Max Drawdown"]


def _row(year, split, label, reward, m, selected):
    return {
        "fold": year, "split": split, "model": label, "reward": reward, "selected": selected,
        **{k: m[k] for k in METRIC_COLS},
        "Total Operations": m.get("Total Operations", np.nan),
        "degenerado": m.get("degenerado", False),
    }


def _best_key(train_eval):
    def _score(item):
        _, r = item
        return (not r["metrics"]["degenerado"], r["metrics"][SELECTION_METRIC])
    return max(train_eval.items(), key=_score)[0]


# ----------------------------------------------------------------------
# SELECAO (apos estagio 1): elege vencedor por fold no TREINO -> winners.json
# ----------------------------------------------------------------------
def selecionar():
    feats, macro, prices = _load_full_timeline()
    winners, rows = {}, []

    for year in TEST_YEARS:
        ft, mt, pt = slice_train(feats, macro, prices, year)
        train_eval = {}
        for algo, reward in itertools.product(ALGOS_LIST, REWARD_TYPES):
            res = _eval_over_seeds(algo, True, reward, year, ft, mt, pt)
            if res is None:
                continue
            train_eval[(algo, reward)] = res
            m = res["metrics"]
            rows.append(_row(year, "train", model_label(algo, True), reward, m, False))

        if not train_eval:
            print(f"FOLD {year}: nenhum modelo com contexto encontrado -- pulando")
            continue

        b_algo, b_reward = _best_key(train_eval)
        winners[str(year)] = {"algo": b_algo, "reward": b_reward}
        # marca o vencedor nas linhas de treino
        for r in rows:
            if r["fold"] == year and r["model"] == model_label(b_algo, True) and r["reward"] == b_reward:
                r["selected"] = True
        bm = train_eval[(b_algo, b_reward)]["metrics"]
        print(f"FOLD {year}: vencedor (treino) = {b_algo} / {b_reward}  "
              f"Sharpe={bm['Sharpe Ratio']:.3f}")

    with open(WINNERS_PATH, "w") as f:
        json.dump(winners, f, indent=2)
    print(f"\nSalvo: {WINNERS_PATH}  ({len(winners)} folds)")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(METRICS_DIR, "walkforward_selecao_treino.csv")
    df.to_csv(out_csv, index=False)
    print(f"Salvo: {out_csv}  ({len(df)} linhas)")
    return winners


# ----------------------------------------------------------------------
# AVALIACAO FINAL (apos estagio 2): ablacao pareada + benchmarks no TESTE
# ----------------------------------------------------------------------
def avaliar_todos():
    winners = load_winners()
    feats, macro, prices = _load_full_timeline()
    rows = []

    for year in TEST_YEARS:
        w = winners.get(str(year))
        if w is None:
            continue
        algo, reward = w["algo"], w["reward"]
        test_slice = slice_test(feats, macro, prices, year)
        if test_slice is None:
            continue
        f_te, m_te, p_te = test_slice
        _, _, pt = slice_train(feats, macro, prices, year)   # precos de treino p/ MVO

        print(f"\n{'='*60}\nFOLD test={year}  | vencedor {algo} / {reward}\n{'='*60}")

        # DRL COM contexto (vencedor) vs SEM contexto (ablacao, mesma config)
        curves = {}
        for use_ctx in (True, False):
            res = _eval_over_seeds(algo, use_ctx, reward, year, f_te, m_te, p_te)
            label = model_label(algo, use_ctx)
            if res is None:
                print(f"  [teste ] {label:18s}: modelos nao encontrados -- pulando")
                continue
            m = res["metrics"]
            curves[label] = res["curve"]
            rows.append(_row(year, "test", label, reward, m, use_ctx))   # selected=True no vencedor c/ contexto
            print(f"  [teste ] {label:18s} Sharpe={m['Sharpe Ratio']:.3f} Ret={m['Total Return']:.3f}"
                  f"{' [DEGENERADO]' if m['degenerado'] else ''}")

        # benchmarks
        eq_curve = equal_weight_baseline(p_te)
        mvo_curve = mvo_baseline(p_te, pt)
        for label, curve in [("Equal Weight", eq_curve), ("Markowitz (MVO)", mvo_curve)]:
            bm = compute_metrics(curve)
            rows.append(_row(year, "test", label, "-", bm, False))
            print(f"  [bench ] {label:18s} Sharpe={bm['Sharpe Ratio']:.3f} Ret={bm['Total Return']:.3f}")

        _plot_fold(year, algo, reward, curves, eq_curve, mvo_curve)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(METRICS_DIR, "walkforward_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSalvo: {out_csv}  ({len(df)} linhas)")
    return df


def _plot_fold(year, algo, reward, curves, eq_curve, mvo_curve):
    plt.figure(figsize=(11, 5))
    style = {model_label(algo, True): ("tab:orange", "-"),
             model_label(algo, False): ("tab:red", "-.")}
    for label, curve in curves.items():
        c, ls = style.get(label, ("tab:blue", "-"))
        plt.plot(curve.values, label=f"DRL: {label}", linewidth=1.8, color=c, linestyle=ls)
    plt.plot(eq_curve.values, label="Equal Weight", linewidth=1.2, color="black", linestyle="--")
    plt.plot(mvo_curve.values, label="Markowitz (MVO)", linewidth=1.2, color="gray", linestyle=":")
    plt.axhline(1.0, color="lightgray", linewidth=0.8)
    plt.title(f"Walk-forward -- teste {year}  (vencedor: {algo} / {reward})")
    plt.xlabel("Dias de pregao"); plt.ylabel("Valor do portfolio (base = 1.0)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"fold{year}_ablacao_contexto.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  grafico: {out}")


def perform_walk_validation():
    """Roda TODOS os treinos do estagio atual em sequencia (uso local)."""
    for cid in range(len(get_configs())):
        train_one(cid)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward analysis 2 estagios (Apuana).")
    parser.add_argument("--config_id", type=int, default=None,
                        help="indice do treino para job-array (depende de WF_STAGE)")
    parser.add_argument("--mode", choices=["selecionar", "avaliar"], default=None,
                        help="selecionar: elege vencedor/fold (apos estagio 1) | "
                             "avaliar: ablacao + benchmarks no teste (apos estagio 2)")
    parser.add_argument("--list", action="store_true", help="lista os config_id do estagio atual")
    parser.add_argument("--all", action="store_true", help="roda todos os treinos do estagio em sequencia (local)")
    args = parser.parse_args()

    if args.list:
        configs = get_configs()
        ctx = "com contexto" if STAGE == "1" else "SEM contexto (ablacao)"
        for cid, (yr, algo, reward, sd) in enumerate(configs):
            print(f"{cid:4d}  fold{yr}  {algo:4s}  r={reward:<12s}  seed={sd}")
        print(f"\nEstagio {STAGE} ({ctx}) | Total: {len(configs)} configs | step={STEP_SIZE}")
    elif args.mode == "selecionar":
        selecionar()
    elif args.mode == "avaliar":
        avaliar_todos()
    elif args.all:
        perform_walk_validation()
    elif args.config_id is not None:
        train_one(args.config_id)
    else:
        parser.error("informe --config_id <i>, --mode selecionar|avaliar, --all ou --list")
