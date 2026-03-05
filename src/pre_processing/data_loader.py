
import pandas as pd
import numpy as np
import os
import glob
def load_rl_data(features_path, macro_path):

    feature_files = glob.glob(os.path.join(features_path, "*.csv"))

    all_features = []
    all_prices = []
    for file in feature_files:
        # 1. Ajuste do Nome do Ticker:
        # Se o arquivo é "ABEV3.SA_cleaned.csv_features.csv"
        filename = os.path.basename(file)
        ticker = filename.split('_cleaned')[0] # Extrai apenas "ABEV3.SA"

        df_feat = pd.read_csv(file, index_col=0, parse_dates=True)

        project_root = os.path.abspath(
            os.path.join(features_path, "..", "..")
        )

       
        raw_path = os.path.join(
            project_root,
            "data",
            "processed",
            "data_cleaned",  
            f"{ticker}_cleaned.csv" 
        )

        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path, index_col='Date', parse_dates=True)

           
            all_prices.append(df_raw['Close'].rename(ticker))

            df_feat.columns = [f"{ticker}_{col}" for col in df_feat.columns]
            all_features.append(df_feat)
        else:
            print(f"Aviso: Preços não encontrados para {ticker} em {raw_path}")

    if not all_features or not all_prices:
        raise ValueError("Falha crítica: Nenhuma combinação de Feature + Preço foi encontrada.")

    
    df_features = pd.concat(all_features, axis=1, sort=False)
    df_prices   = pd.concat(all_prices, axis=1, sort=False)
    df_macro    = pd.read_csv(macro_path, index_col=0, parse_dates=True)


    df_features = df_features.sort_index()
    df_prices   = df_prices.sort_index()
    df_macro    = df_macro.sort_index()

 
    # alinhar preços e features
    common_index = df_prices.index.intersection(df_features.index)

    df_prices   = df_prices.loc[common_index]
    df_features = df_features.loc[common_index]

    # alinhar macro usando reindex + forward fill
    df_macro = df_macro.reindex(common_index).ffill()

    df_features = df_features.ffill()
    df_macro    = df_macro.ffill()
    df_prices   = df_prices.ffill()

  
    df_prices = df_prices.dropna(how="all")

    valid_index = df_prices.index

    df_features = df_features.loc[valid_index]
    df_macro    = df_macro.loc[valid_index]
    
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

    return df_features, df_macro, df_prices

def temporal_split(df_features, df_macro, df_prices, split_date):

    split_date = pd.to_datetime(split_date)

    df_feat_train = df_features.loc[df_features.index < split_date]
    df_feat_val   = df_features.loc[df_features.index >= split_date]

    df_macro_train = df_macro.loc[df_macro.index < split_date]
    df_macro_val   = df_macro.loc[df_macro.index >= split_date]

    df_prices_train = df_prices.loc[df_prices.index < split_date]
    df_prices_val   = df_prices.loc[df_prices.index >= split_date]

    return (
        df_feat_train, df_macro_train, df_prices_train,
        df_feat_val, df_macro_val, df_prices_val
    )

if __name__ == "__main__":
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.abspath(os.path.join(base_path, "..", ".."))
    feat_p = os.path.join(project_root, "data", "processado_features")
    macro_p = os.path.join(project_root, "data", "external", "context.csv")
    
    try:
        
        df_features, df_macro, df_prices = load_rl_data(feat_p, macro_p)
        split_date = "2018-01-01"

        print("MIN:", df_features.index.min())
        print("MAX:", df_features.index.max())
        print("Split:", pd.to_datetime(split_date))

        print(type(df_features.index))
        print(df_features.index.dtype)
        print(type(df_features.index[0]))
       
        output_path = os.path.join(base_path, "..","..", "data", "processed", "data_train_val")
        (
            df_feat_train, df_macro_train, df_prices_train,
            df_feat_val, df_macro_val, df_prices_val
        ) = temporal_split(df_features, df_macro, df_prices, split_date)

        df_feat_train.to_csv(os.path.join(output_path,"master_features_train.csv"))
        df_macro_train.to_csv(os.path.join(output_path, "master_macro_train.csv"))
        df_prices_train.to_csv(os.path.join(output_path, "master_prices_train.csv"))

        df_feat_val.to_csv(os.path.join(output_path, "master_features_val.csv"))
        df_macro_val.to_csv(os.path.join(output_path, "master_macro_val.csv"))
        df_prices_val.to_csv(os.path.join(output_path, "master_prices_val.csv"))
        
        print("--- SUCESSO ---")
        print(f"Arquivos salvos em: {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"--- ERRO CRÍTICO ---")
        print(e)