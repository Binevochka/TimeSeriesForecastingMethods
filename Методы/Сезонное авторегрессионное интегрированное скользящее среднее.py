import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

# Генерация синтетического временного ряда с сезонностью, трендом и шумом
np.random.seed(42)
time = np.arange(250)
trend = 0.3 * time  # Линейный тренд
seasonal = 10 * np.sin(2 * np.pi * time / 12)  # Сезонная составляющая (период 12)
noise = np.random.normal(scale=5, size=len(time))  # Случайный шум
data = trend + seasonal + noise

# Создание DataFrame для удобства работы
series = pd.Series(data, index=pd.date_range(start="2020-01-01", periods=len(time), freq="D"))

# Визуализация временного ряда
plt.figure(figsize=(10, 6))
plt.plot(series, label="Синтетический временной ряд")
plt.title("Синтетический временной ряд с трендом и сезонностью")
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.legend()
plt.grid(True)
plt.show()

# Проверка стационарности с тестом Дики-Фуллера
def check_stationarity(series):
    result = adfuller(series)
    print(f"p-значение теста Дики-Фуллера: {result[1]:.5f}")
    if result[1] < 0.05:
        print("Ряд стационарен.")
    else:
        print("Ряд нестационарен.")

print("Проверка стационарности исходного ряда:")
check_stationarity(series)

# Дифференцирование для достижения стационарности
series_diff = series.diff().dropna()
print("\nПроверка стационарности после дифференцирования:")
check_stationarity(series_diff)

# Построение графика автокорреляции
plt.figure(figsize=(10, 6))
autocorrelation_plot(series_diff)
plt.title("График автокорреляции для дифференцированного ряда")
plt.grid(True)
plt.show()

# Построение и обучение модели SARIMA
# Параметры SARIMA (p, d, q) x (P, D, Q, m)
order = (1, 1, 1)  # Несезонные параметры (p, d, q)
seasonal_order = (1, 1, 1, 12)  # Сезонные параметры (P, D, Q, m)

model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
fitted_model = model.fit(disp=False)

# Результаты модели
print("\nРезультаты модели SARIMA:")
print(fitted_model.summary())

# Прогнозирование
forecast_steps = 30  # Прогноз на 30 шагов вперёд
forecast = fitted_model.forecast(steps=forecast_steps)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(series, label="Исходные данные")
plt.plot(pd.date_range(series.index[-1], periods=forecast_steps + 1, freq="D")[1:], forecast, color="red", label="Прогноз")
plt.title("Прогноз модели SARIMA")
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.legend()
plt.grid(True)
plt.show()

# Создание таблицы с прогнозом
forecast_index = pd.date_range(series.index[-1], periods=forecast_steps + 1, freq="D")[1:]
forecast_df = pd.DataFrame({
    "Прогноз": forecast
}, index=forecast_index)

print("\nПрогноз на следующие шаги:")
print(forecast_df)