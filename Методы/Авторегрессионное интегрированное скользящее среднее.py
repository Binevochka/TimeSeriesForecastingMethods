import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot

# Генерация синтетического временного ряда с трендом, сезонностью и шумом
np.random.seed(42)
time = np.arange(250)
trend = 0.5 * time  # Линейный тренд
seasonal = 10 * np.sin(time * 2 * np.pi / 20)  # Сезонная составляющая
noise = np.random.normal(scale=5, size=len(time))  # Случайный шум
data = trend + seasonal + noise

# Создание DataFrame для удобства работы
series = pd.Series(data, index=pd.date_range(start='2020-01-01', periods=len(time), freq='D'))

# Визуализация исходного временного ряда
plt.figure(figsize=(10, 6))
plt.plot(series)
plt.title('Синтетический временной ряд')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.grid(True)
plt.show()

# Проверка стационарности ряда с помощью теста Дики-Фуллера
def check_stationarity(series):
    result = adfuller(series)
    print(f"p-значение теста Дики-Фуллера: {result[1]}")
    if result[1] < 0.05:
        print("Ряд стационарен.")
    else:
        print("Ряд нестационарен. Необходима дифференциация.")

check_stationarity(series)

# Дифференцирование ряда для достижения стационарности
series_diff = series.diff().dropna()

# Повторная проверка на стационарность после дифференцирования
check_stationarity(series_diff)

# Построение графика автокорреляции для дифференцированного ряда
plt.figure(figsize=(10, 6))
autocorrelation_plot(series_diff)
plt.title('Автокорреляция для дифференцированного ряда')
plt.grid(True)
plt.show()

# Строим модель ARIMA с параметрами p=1, d=1, q=1
model = ARIMA(series, order=(1, 1, 1))
fitted_model = model.fit()

# Выводим результаты модели
print("\nРезультаты модели ARIMA(1, 1, 1):")
print(fitted_model.summary())

# Прогнозирование на следующие 10 шагов
forecast_steps = 10
forecast = fitted_model.forecast(steps=forecast_steps)

# Визуализация прогноза
plt.figure(figsize=(10, 6))
plt.plot(series, label='Исходные данные')
plt.plot(pd.date_range(series.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast, color='red', label='Прогноз')
plt.fill_between(pd.date_range(series.index[-1], periods=forecast_steps + 1, freq='D')[1:], forecast - 1.96, forecast + 1.96, color='pink', alpha=0.3, label='Доверительный интервал')
plt.title('Прогноз с доверительным интервалом для ARIMA(1,1,1)')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid(True)
plt.show()

# Создание таблицы с результатами прогноза
forecast_index = pd.date_range(series.index[-1], periods=forecast_steps + 1, freq='D')[1:]
forecast_df = pd.DataFrame({
    'Прогноз': forecast,
    'Нижняя граница доверительного интервала': forecast - 1.96,
    'Верхняя граница доверительного интервала': forecast + 1.96
}, index=forecast_index)

print("\nПрогноз и доверительные интервалы:")
print(forecast_df)