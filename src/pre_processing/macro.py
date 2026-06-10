import pandas as pd
import yfinance as yf
from bcb import Expectativas

start_date = "2000-01-01"
end_date = "2024-12-31"

exp = Expectativas()
try:
    ep = exp.get_endpoint('ExpectativasMercadoEstatisticasAnuais')
except:
    ep = exp.get_endpoint('ExpectativasMercadoAnuais')

def get_market_consensus(indicador):
    print(f"Baixando dados de: {indicador}...")
  
    return (ep.query()
            .filter(ep.Indicador == indicador)
            .filter(ep.Data >= start_date)
            .filter(ep.Data <= end_date)
            .select(ep.Data, ep.Media, ep.DataReferencia)
            .collect())

ipca_raw  = get_market_consensus('IPCA')
selic_raw = get_market_consensus('Selic')
pib_raw   = get_market_consensus('PIB Total')

def clean_df(df, name):
    if df.empty: return pd.DataFrame(columns=[name])
    df['Data'] = pd.to_datetime(df['Data'])
    
   
    df = df[pd.to_datetime(df['DataReferencia'], format='%Y').dt.year == df['Data'].dt.year]
    
    return df.groupby('Data')['Media'].mean().to_frame(name)

# 2) Processamento
focus_df = pd.concat([
    clean_df(ipca_raw, "ipca_expect"), 
    clean_df(selic_raw, "selic_expect"),
    clean_df(pib_raw, "pib_expect")
], axis=1)


print("Baixando dados do Yahoo Finance...")
market_data = yf.download(["^BVSP", "^VIX"], start=start_date, end=end_date)

if isinstance(market_data.columns, pd.MultiIndex):
    market_data = market_data['Close']
else:
    market_data = market_data[['Close']]

market_data.columns = ['ibov', 'vix']
market_data.index = market_data.index.tz_localize(None)

data = focus_df.merge(market_data, left_index=True, right_index=True, how="outer")
data = data.sort_index().ffill().dropna()

print("\n--- Shape do Dataset ---")
print(data.shape)
print(data.tail())


#Criar outro macro que usa

data.to_csv("data/external/context.csv")