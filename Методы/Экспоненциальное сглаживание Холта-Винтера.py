import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def holt_winters_additive(series, alpha, beta, gamma, season_length, n_forecast):
    """
    Реализация метода Холта-Винтера для аддитивной сезонности.

    :param series: Временной ряд (Pandas Series)
    :param alpha: Коэффициент сглаживания уровня
    :param beta: Коэффициент сглаживания тренда
    :param gamma: Коэффициент сглаживания сезонности
    :param season_length: Длина сезонного периода
    :param n_forecast: Количество шагов для прогноза
    :return: Сглаженные значения и прогнозы
    """
    n = len(series)
    level = np.zeros(n)
    trend = np.zeros(n)
    season = np.zeros(n)
    forecast = np.zeros(n + n_forecast)

    # Инициализация компонентов
    level[0] = series.iloc[:season_length].mean()
    trend[0] = (series.iloc[season_length:2 * season_length].mean() -
                series.iloc[:season_length].mean()) / season_length
    season[:season_length] = series.iloc[:season_length] - level[0]

    # Обновление компонентов
    for t in range(1, n):
        season_idx = (t - season_length) % season_length
        level[t] = alpha * (series.iloc[t] - season[season_idx]) + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        season[t] = gamma * (series.iloc[t] - level[t]) + (1 - gamma) * season[season_idx]

    # Генерация прогноза
    for t in range(n, n + n_forecast):
        forecast[t] = level[-1] + (t - n + 1) * trend[-1] + season[(t - season_length) % season_length]

    return level, trend, season, forecast

# Генерация синтетических данных с трендом и сезонностью
np.random.seed(42)
n = 100
season_length = 12
trend = np.linspace(10, 50, n)
season = 10 * np.sin(2 * np.pi * np.arange(n) / season_length)
noise = np.random.normal(0, 2, n)
data = trend + season + noise

# Создание временного ряда
series = pd.Series(data, index=pd.date_range(start="2020-01-01", periods=n, freq="M"))

# Применение метода Холта-Винтера
alpha = 0.3
beta = 0.1
gamma = 0.2
n_forecast = 12
level, trend, season, forecast = holt_winters_additive(series, alpha, beta, gamma, season_length, n_forecast)

# Построение графиков
plt.figure(figsize=(12, 6))
plt.plot(series, label="Исходный ряд", color="blue")
plt.plot(series.index, level + trend + season[:n], label="Сглаженный ряд", color="red")
plt.plot(pd.date_range(series.index[-1], periods=n_forecast + 1, freq="M")[1:],
         forecast[n:], label="Прогноз", color="green")
plt.title("Метод экспоненциального сглаживания Холта-Винтера")
plt.xlabel("Дата")
plt.ylabel("Значение")
plt.legend()
plt.grid(True)
plt.show()