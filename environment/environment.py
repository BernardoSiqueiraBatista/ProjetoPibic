import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd


class PortfolioEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, df_features, df_macro, df_prices, reward_type, step_size,
                 eval_mode=False, use_context=True, max_episode_steps=252):
        super(PortfolioEnv, self).__init__()

        self.eval_mode = eval_mode
        self.df_features = df_features
        self.df_macro = df_macro
        self.prices = df_prices
        self.use_context = use_context
        self.tickers = df_prices.columns.tolist()
        self.action_history = []
        self.reward_type = reward_type
        self.max_episode_steps = max_episode_steps

        self.n_assets = len(df_prices.columns)
        self.n_feature_cols = len(df_features.columns)
        self.n_macro_cols = len(df_macro.columns)

        if use_context:
            total_size = (
                self.n_feature_cols +
                self.n_assets +
                self.n_macro_cols +
                self.n_assets
            )
        else:
            total_size = (
                self.n_feature_cols +
                self.n_assets +
                self.n_assets
            )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        self.step_size = step_size
        self.episode_start = 30
        self.current_step = 30

        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns_history = []

    # ---- métodos auxiliares para avaliação com VecEnv ----
    def reset_action_history(self):
        self.action_history = []

    def get_action_history(self):
        return self.action_history

    # ---- construção do estado ----
    def _get_observation(self):
        t = self.current_step

        ft = self.df_features.iloc[t].values
        wt = self.weights
        rt = self.calculate_recent_returns(t)

        if self.use_context:
            ct = self.df_macro.iloc[t].values
            state = np.concatenate([ft, wt, ct, rt])
        else:
            state = np.concatenate([ft, wt, rt])

        if np.isnan(ft).any():
            print("NaN nas FEATURES")
        if self.use_context and np.isnan(ct).any():
            print("NaN nas MACRO")
        if np.isnan(rt).any():
            print("NaN nos RETURNS")

        return state.astype(np.float32)

    def calculate_recent_returns(self, t):
        # guard: evita índice negativo (que o pandas leria do fim do DataFrame)
        idx = max(t - 30, 0)
        current_prices = self.prices.iloc[t].values
        old_prices = self.prices.iloc[idx].values
        rt = (current_prices / (old_prices + 1e-8)) - 1
        return rt

    # ---- recompensa ----
    def _compute_reward(self, ret, new_weights):
        """Calcula a reward a partir do histórico de retornos DIÁRIOS."""
        returns_array = np.array(self.portfolio_returns_history)
        window = 30
        recent_returns = returns_array[-window:]
        risk_free_daily = 0.08 / 252

        if len(recent_returns) > 1:
            mean_ret = recent_returns.mean()
            std_ret = recent_returns.std() + 1e-8
            sharpe = np.sqrt(252) * (mean_ret - risk_free_daily) / (std_ret + 1e-8)
            volatility = std_ret

            downside_returns = recent_returns[recent_returns < risk_free_daily] - risk_free_daily
            downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 1e-8
            sortino = np.sqrt(252) * (mean_ret - risk_free_daily) / (downside_deviation + 1e-8)

            gains = np.sum(np.maximum(recent_returns - risk_free_daily, 0))
            losses = np.sum(np.maximum(risk_free_daily - recent_returns, 0))
            omega = gains / (losses + 1e-8)

            if len(recent_returns) > 5:
                cum_rets = np.cumprod(1 + recent_returns)
                roll_max = np.maximum.accumulate(cum_rets)
                drawdowns = (cum_rets - roll_max) / (roll_max + 1e-8)
                max_dd = np.abs(drawdowns.min()) + 1e-8
                calmar = (mean_ret * 252) / max_dd
            else:
                calmar = 0.0
        else:
            sharpe = 0.0
            volatility = 0.0
            sortino = 0.0
            omega = 0.0
            calmar = 0.0

        transaction_cost = np.sum(np.abs(new_weights - self.weights)) / self.n_assets

        reward = 0.0
        if self.reward_type == "log-retorno":
            reward = np.log(1 + ret)
        elif self.reward_type == "huang-return":
            reward = 0.6 * ret + 0.3 * sharpe - 0.1 * volatility - 0.05 * transaction_cost
        elif self.reward_type == "sortino":
            reward = sortino
        elif self.reward_type == "omega":
            reward = omega - 1.0
        elif self.reward_type == "calmar":
            reward = calmar

        return float(np.clip(reward, -10.0, 10.0))

    # ---- passo do ambiente ----
    def step(self, action):
        # normaliza a ação para pesos que somam 1
        if np.sum(action) > 0:
            new_weights = action / np.sum(action)
        else:
            new_weights = np.ones(self.n_assets) / self.n_assets

        # --- segura os pesos por step_size dias, compondo os retornos ---
        # O agente decide UMA vez; o mercado anda step_size dias com esses pesos.
        cumulative = 1.0
        days_advanced = 0
        daily_returns_period = []  # retornos dia a dia (para a curva de capital diária)

        for i in range(self.step_size):
            t = self.current_step + i
            if t + 1 >= len(self.prices):
                break  # fim do dataset

            price_today = self.prices.iloc[t].values
            price_tomorrow = self.prices.iloc[t + 1].values

            asset_returns = (price_tomorrow / (price_today + 1e-8)) - 1
            asset_returns = np.nan_to_num(asset_returns, nan=0.0, posinf=0.0, neginf=0.0)

            daily_return = np.dot(new_weights, asset_returns)

            self.portfolio_returns_history.append(daily_return)  # histórico DIÁRIO (usado na reward)
            daily_returns_period.append(float(daily_return))      # exposto no info (usado na avaliação)
            cumulative *= (1 + daily_return)
            days_advanced += 1

        portfolio_return = cumulative - 1.0  # retorno COMPOSTO do período de holding
        ret = portfolio_return

        reward = self._compute_reward(ret, new_weights)

        self.weights = new_weights
        self.action_history.append(pd.Series(new_weights, index=self.tickers))
        self.current_step += self.step_size

        # --- terminação ---
        if self.eval_mode:
            # avaliação: passe contínuo, termina quando os dados acabam
            terminated = (self.current_step + 1 >= len(self.prices)) or (days_advanced == 0)
        else:
            # treino: termina por nº de steps do episódio OU fim do dataset
            steps_no_episodio = self.current_step - self.episode_start
            terminated = (steps_no_episodio >= self.max_episode_steps) or \
                         (self.current_step + 1 >= len(self.prices))
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_observation()

        if np.isnan(obs).any():
            print("OBS NaN")
        if np.isnan(reward):
            print("REWARD NaN")

        info = {
            "portfolio_return": portfolio_return,   # composto do período
            "daily_returns": daily_returns_period,  # lista dia a dia -> curva diária ao longo dos anos
            "weights": new_weights.copy(),          # pesos do período (robusto ao auto-reset do VecEnv)
        }
        return obs, reward, terminated, truncated, info

    # ---- reset ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # margem para um episódio completo + um período de holding
        max_start = len(self.df_features) - self.max_episode_steps - self.step_size - 2

        if not self.eval_mode:
            # treino: início aleatório (regimes de mercado diversos)
            self.episode_start = int(np.random.randint(30, max_start))
        else:
            # avaliação: sempre do começo, passe contínuo
            self.episode_start = 30

        self.current_step = self.episode_start

        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_returns_history = []
        self.action_history = []

        obs = self._get_observation()
        return obs, {}