import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd


class PortfolioEnv(gym.Env):

    def __init__(self, df_features, df_macro, df_prices):
        super(PortfolioEnv, self).__init__()

        self.df_features = df_features
        self.df_macro = df_macro
        self.prices = df_prices
        self.tickers = df_prices.columns.tolist()
        self.action_history = []


        self.n_assets = len(df_prices.columns)
        self.n_feature_cols = len(df_features.columns)
        self.n_macro_cols = len(df_macro.columns)

        total_size = (
            self.n_feature_cols +
            self.n_assets +
            self.n_macro_cols +
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

        self.current_step = 30
        self.weights = np.ones(self.n_assets) / self.n_assets

        self.portfolio_returns_history = []


    def _get_observation(self):
        t = self.current_step

        ft = self.df_features.iloc[t].values
        wt = self.weights
        ct = self.df_macro.iloc[t].values
        rt = self.calculate_recent_returns(t)

        state = np.concatenate([ft, wt, ct, rt])

        if np.isnan(ft).any():
            print("NaN nas FEATURES")

        if np.isnan(ct).any():
            print("NaN nas MACRO")

        if np.isnan(rt).any():
            print("NaN nos RETURNS")
        return state.astype(np.float32)


 
    def calculate_recent_returns(self, t):
        current_prices = self.prices.iloc[t].values
        old_prices = self.prices.iloc[t - 30].values

        rt = (current_prices / (old_prices + 1e-8)) - 1
        return rt

    def step(self, action):

        if np.sum(action) > 0:
            new_weights = action / np.sum(action)
        else:
            new_weights = np.ones(self.n_assets) / self.n_assets

        price_today = self.prices.iloc[self.current_step].values
        price_tomorrow = self.prices.iloc[self.current_step + 1].values

        asset_returns = (price_tomorrow / (price_today + 1e-8)) - 1
        asset_returns = np.nan_to_num(asset_returns, nan=0.0, posinf=0.0, neginf=0.0)

        portfolio_return = np.dot(new_weights, asset_returns)

        ret = portfolio_return

        self.portfolio_returns_history.append(ret)

        returns_array = np.array(self.portfolio_returns_history)

      
        window = 30
        recent_returns = returns_array[-window:]

        if len(recent_returns) > 1:
            mean_ret = recent_returns.mean()
            std_ret = recent_returns.std() + 1e-8
            sharpe = mean_ret / std_ret
            volatility = std_ret
        else:
            sharpe = 0.0
            volatility = 0.0

        #quero uma função reward que seja o np.logs de retorno, depois alterar isso aqui para combinar com a metolodiga
        ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        reward = (np.log(1 + ret)*100) if ret > -1 else -1

        self.weights = new_weights

        self.action_history.append(
     pd.Series(new_weights, index=self.tickers)
    )
    
        self.current_step += 1

        terminated = self.current_step >= (len(self.df_features) - 2)
        truncated = False

        obs = self._get_observation()

        if np.isnan(obs).any():
            print("OBS NaN")

        if np.isnan(reward):
            print("REWARD NaN")

        #if np.isnan(self.portfolio_value):
        #   print("PORTFOLIO NaN")

        return obs, reward, terminated, truncated, {
            "portfolio_return": portfolio_return
        }

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 30
        self.weights = np.ones(self.n_assets) / self.n_assets

        self.portfolio_returns_history = []

        self.action_history = []

        obs = self._get_observation()
        return obs, {}