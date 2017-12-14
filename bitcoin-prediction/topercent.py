import pandas as pd

data = pd.read_csv('bitcoin.csv')

data['Open_rate'] = 0.0
data['High_rate'] = 0.0
data['Low_rate'] = 0.0
data['Close_rate'] = 0.0

for i in range(len(data)-1):
    data['Open_rate'][i+1] = round(100*(data.Open[i+1]-data.Open[i])/data.Open[i],2)
    data['High_rate'][i+1] = round(100*(data.High[i+1]-data.High[i])/data.High[i],2)
    data['Low_rate'][i+1] = round(100*(data.Low[i+1]-data.Low[i])/data.Low[i],2)
    data['Close_rate'][i+1] = round(100*(data.Close[i+1]-data.Close[i])/data.Close[i],2)

data.to_csv("bitcoin_rate.csv")