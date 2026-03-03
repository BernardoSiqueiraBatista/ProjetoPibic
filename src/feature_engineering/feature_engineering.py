import pandas as pd
import pandas_ta as ta
import os
import glob

base_path = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(base_path, "..", ".."))

input_dir = os.path.join(project_root, "data", "raw", "csv_ohlcv")
output_dir = os.path.join(project_root, "data", "processado_features")

print(f"Lendo de: {input_dir}")
print(f"Salvando em: {output_dir}")

os.makedirs(output_dir, exist_ok=True)

files = glob.glob(os.path.join(input_dir, "*.csv"))
print(f"Arquivos encontrados: {len(files)}")

if len(files) == 0:
    print("ERRO: Nenhum arquivo .csv encontrado! Verifique se a pasta csv_ohlcv está no lugar certo.")

for f in files:
    ticker = os.path.basename(f)
    print(f"Processando {ticker}...", end=" ")
    
    try:
        df = pd.read_csv(f, index_col='Date', parse_dates=True)
        if len(df) < 50:
            print("PULADO (muito curto)")
            continue
            
        # Features
        df['sma_20'] = ta.sma(df['Close'], length=20)
        df['F_trend_sma'] = (df['Close'] / df['sma_20']) - 1
        df['F_momentum_rsi'] = ta.rsi(df['Close'], length=14) / 100
        bbands = ta.bbands(df['Close'], length=20, std=2)
        df['F_volat_bb'] = bbands.iloc[:, 3] / 100 
        df['obv'] = ta.obv(df['Close'], df['Volume'])
        df['F_volume_obv'] = df['obv'].pct_change()

        features_cols = [c for c in df.columns if c.startswith('F_')]
        df_final = df[features_cols].dropna()
        
        save_path = os.path.join(output_dir, f"{ticker}_features.csv")
        df_final.to_csv(save_path)
        print("OK!")
    except Exception as e:
        print(f"ERRO: {e}")

print("Fim do processo.")