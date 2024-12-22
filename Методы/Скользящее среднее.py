import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

# Генерация синтетического временного ряда
np.random.seed(42)  # Для воспроизводимости
n = 250  # Количество точек
noise = np.random.normal(0, 1, n)  # Случайные шумы
mu = 0.5  # Среднее значение ряда
theta = [0.7]  # Параметры модели MA(1)

# Построение ряда MA(1): y_t = μ + ε_t + θ_1 * ε_(t-1)
y = np.zeros(n)
for t in range(1, n):
    y[t] = mu + noise[t] + theta[0] * noise[t - 1]

# Создание DataFrame для удобства
data = pd.DataFrame({'Time': range(n), 'Value': y})
print("Первые 10 строк данных:")
print(data.head(10))

# Визуализация временного ряда
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['Value'], label='Временной ряд (MA)')
plt.title('Временной ряд, сгенерированный с использованием модели MA(1)')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid()
plt.show()

# Построение автокорреляционной функции (ACF)
plot_acf(y, lags=20)
plt.title('ACF временного ряда (MA)')
plt.show()

# Построение модели MA(1) с использованием библиотеки statsmodels
# Определение модели ARIMA(p=0, d=0, q=1), что соответствует MA(1)
model = ARIMA(y, order=(0, 0, 1))
fitted_model = model.fit()

# Вывод параметров модели
print("\nОцененные параметры модели MA(1):")
print(fitted_model.summary())

# Прогнозирование
n_forecast = 10  # Количество точек для прогноза
forecast = fitted_model.get_forecast(steps=n_forecast)
forecast_values = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# Визуализация прогноза
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['Value'], label='Оригинальные данные')
plt.plot(range(n, n + n_forecast), forecast_values, label='Прогноз', color='red')
plt.fill_between(
    range(n, n + n_forecast),
    forecast_conf_int[:, 0],  # Нижняя граница доверительного интервала
    forecast_conf_int[:, 1],  # Верхняя граница доверительного интервала
    color='pink',
    alpha=0.3,
    label='Доверительный интервал',
)
plt.title('Прогноз временного ряда (MA)')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid()
plt.show()

# Проверка остатков (шумов)
residuals = fitted_model.resid
plt.figure(figsize=(10, 5))
plt.plot(range(len(residuals)), residuals, label='Остатки (шумы)', color='green')
plt.axhline(y=0, color='red', linestyle='--', label='y=0')
plt.title('Остатки модели MA(1)')
plt.xlabel('Время')
plt.ylabel('Остаток')
plt.legend()
plt.grid()
plt.show()

# Таблица с прогнозами
forecast_df = pd.DataFrame({
    'Time': range(n, n + n_forecast),
    'Forecast': forecast_values,
    'Lower CI': forecast_conf_int[:, 0],  # Нижняя граница доверительного интервала
    'Upper CI': forecast_conf_int[:, 1]   # Верхняя граница доверительного интервала
})

# Вывод таблицы
print("\nПрогноз и доверительные интервалы:")
print(forecast_df)