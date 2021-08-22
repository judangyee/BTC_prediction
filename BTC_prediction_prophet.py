import pystan 
import pyupbit
from fbprophet import Prophet
from matplotlib import pyplot as plt

#BTC 최근 200시간의 데이터 불러옴
df = pyupbit.get_ohlcv("KRW-BTC", count=2000, interval="minute60")
print(df)
print(type(df))

#df에서 시간과 종가만 남기기(각각 ds와 y)
df = df.reset_index()
df['ds'] = df['index']
df['y'] = df['close']
data = df[['ds','y']]

#위에서 불러온 데이터를 학습시키기
model = Prophet()
#INFO:fbprophet:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
#INFO:fbprophet:Disabling weekly seasonality. Run prophet with weekly_seasonality=True to override this.
#위 두 오류를 없애주기 위해 True로 바꾸어줌
model.weekly_seasonality = True
model.yearly_seasonality = True
model.fit(data)

#24시간 미래 예측
future = model.make_future_dataframe(periods=24, freq='H')
forecast = model.predict(future)

graph1 = model.plot(forecast)
graph2 = model.plot_components(forecast)
print(graph1)

plt.show()