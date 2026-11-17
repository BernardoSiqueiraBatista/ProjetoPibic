"""
Comparacao por ALGORITMO no walk-forward (arquivo auxiliar, NAO sobrescreve nada).

Para cada fold (treina 2000..200x, testa 200x+1) avaliamos TODOS os modelos com
contexto que existem (3 algos x 5 rewards) no ANO DE TESTE e, para cada
algoritmo, pegamos o seu melhor resultado NO TESTE:

    melhor DDPG  x  melhor PPO  x  melhor SAC  x  Markowitz (MVO)  x  Equal-Weight

"Melhor de cada algoritmo" = entre as 5 reward functions, a de maior Sharpe NO
TESTE (media das seeds presentes). Reusa os modelos COM contexto ja treinados no
estagio 1 -- rode depois que o estagio 1 terminar.

    WF_SEEDS=30 python -m environment.walk_forward_compare

Saidas (nomes proprios, nao colidem com --mode avaliar):
    metrics/walkforward/walkforward_comparacao_algos.csv   (TODOS os testados + benchmarks)
    plots/walkforward/foldYYYY_melhor_por_algo_teste.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from environment.walk_forward_validation import (
    _load_full_timeline, slice_train, slice_test, _eval_over_seeds,
    ALGOS_LIST, REWARD_TYPES, TEST_YEARS, _row, model_label,
    PLOTS_DIR, METRICS_DIR,
)
from environment.evaluate import compute_metrics, equal_weight_baseline, mvo_baseline

ALGO_COLORS = {"DDPG": "tab:blue", "PPO": "tab:orange", "SAC": "tab:green"}


def comparar_algos():
    feats, macro, prices = _load_full_timeline()
    rows = []

    for year in TEST_YEARS:
        _, _, pt = slice_train(feats, macro, prices, year)   # precos de treino p/ MVO
        test_slice = slice_test(feats, macro, prices, year)
        if test_slice is None:
            continue
        f_te, m_te, p_te = test_slice

        print(f"\n{'='*60}\nFOLD test={year}\n{'='*60}")
        best_per_algo = {}   # algo -> (reward, curve)

        # avalia TODOS os modelos (algo x reward) no TESTE; guarda o melhor de cada algo
        for algo in ALGOS_LIST:
            cands = {}
            for reward in REWARD_TYPES:
                res = _eval_over_seeds(algo, True, reward, year, f_te, m_te, p_te)
                if res is None:
                    continue
                m = res["metrics"]
                rows.append(_row(year, "test", model_label(algo, True), reward, m, False))
                cands[reward] = res
                print(f"  {algo:4s} r={reward:<12s} Sharpe={m['Sharpe Ratio']:.3f} "
                      f"Ret={m['Total Return']:.3f}{' [DEGENERADO]' if m['degenerado'] else ''}")
            if not cands:
                print(f"  {algo}: sem modelos -- pulando")
                continue
            best_r = max(cands, key=lambda r: (not cands[r]["metrics"]["degenerado"],
                                               cands[r]["metrics"]["Sharpe Ratio"]))
            best_per_algo[algo] = (best_r, cands[best_r]["curve"])
            # marca o melhor do algo (selected=True) nas linhas ja gravadas
            for rr in rows:
                if rr["fold"] == year and rr["model"] == model_label(algo, True) and rr["reward"] == best_r:
                    rr["selected"] = True
            print(f"  -> melhor {algo} no teste: r={best_r}")

        # benchmarks
        eq_curve = equal_weight_baseline(p_te)
        mvo_curve = mvo_baseline(p_te, pt)
        for label, curve in [("Equal Weight", eq_curve), ("Markowitz (MVO)", mvo_curve)]:
            bm = compute_metrics(curve)
            rows.append(_row(year, "test", label, "-", bm, False))
            print(f"  {label:18s} Sharpe={bm['Sharpe Ratio']:.3f} Ret={bm['Total Return']:.3f}")

        _plot(year, best_per_algo, eq_curve, mvo_curve)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(METRICS_DIR, "walkforward_comparacao_algos.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSalvo: {out_csv}  ({len(df)} linhas)")
    print("(coluna 'selected' = melhor reward daquele algoritmo no teste)")
    return df


def _plot(year, best_per_algo, eq_curve, mvo_curve):
    plt.figure(figsize=(11, 5))
    for algo, (reward, curve) in best_per_algo.items():
        plt.plot(curve.values, label=f"melhor {algo} ({reward})", linewidth=1.6,
                 color=ALGO_COLORS.get(algo))
    plt.plot(eq_curve.values, label="Equal Weight", linewidth=1.2, color="black", linestyle="--")
    plt.plot(mvo_curve.values, label="Markowitz (MVO)", linewidth=1.2, color="gray", linestyle=":")
    plt.axhline(1.0, color="lightgray", linewidth=0.8)
    plt.title(f"Walk-forward -- teste {year}: melhor DDPG x PPO x SAC x benchmarks")
    plt.xlabel("Dias de pregao"); plt.ylabel("Valor do portfolio (base = 1.0)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f"fold{year}_melhor_por_algo_teste.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  grafico: {out}")


if __name__ == "__main__":
    comparar_algos()
