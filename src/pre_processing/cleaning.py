import os
import glob
import numpy as np
import pandas as pd


def clean_data():
   

    base_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_path, "..", ".."))

    input_path = os.path.join(project_root, "data", "raw", "csv_ohlcv")
    output_path = os.path.join(project_root, "data", "processed", "data_cleaned")

    os.makedirs(output_path, exist_ok=True)

    feature_files = glob.glob(os.path.join(input_path, "*.csv"))

    lista_outliers = []
    valid_assets = []

    for file in feature_files:

        filename = os.path.basename(file)
        ticker = filename.replace(".csv", "")

        print(f"\nProcessando {ticker}...")

        df = pd.read_csv(file, index_col="Date", parse_dates=True)

        df = df.sort_index()

   
        df = df[
            (df["Open"] > 0) &
            (df["High"] > 0) &
            (df["Low"] > 0) &
            (df["Close"] > 0)
        ]

    
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))

        df = df.dropna(subset=["Return"])

        Q1 = df["Return"].quantile(0.25)
        Q3 = df["Return"].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

     
        n_outliers = ((df["Return"] < lower) | (df["Return"] > upper)).sum()

        if n_outliers > 0:
            lista_outliers.append(ticker)
            #print(f"{n_outliers} retornos extremos detectados em {ticker} (winsorizados)")

        df["Return"] = df["Return"].clip(lower, upper)

      
        df = df.ffill()
  
        df = df.dropna()
    
        df = df.drop(columns=["Return"])
      
        output_file = os.path.join(output_path, f"{ticker}_cleaned.csv")

        print(f"{ticker} salvo em {output_file}")
        
        first_year = df.index[0].year
        
        last_year = df.index[-1].year
        if first_year <= 2000 and last_year >= 2024:
            valid_assets.append(ticker)
            df.to_csv(output_file)
        else:
            print(f"{ticker} excluído.")
        
        
   
    print("\nLimpeza concluída.")
    
    print(valid_assets)
    print(len(valid_assets))
    



if __name__ == "__main__":
    clean_data()