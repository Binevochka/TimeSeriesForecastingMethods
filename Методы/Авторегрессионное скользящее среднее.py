import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

# Генерация синтетического временного ряда с моделью ARMA(1, 1)
np.random.seed(42)  # Для воспроизводимости
n = 250  # Количество точек временного ряда
mu = 0.5  # Среднее значение ряда
phi = 0.6  # Коэффициент авторегрессии (AR)
theta = 0.8  # Коэффициент скользящего среднего (MA)
errors = np.random.normal(0, 1, n)  # Генерация случайных ошибок (шум)

# Построение ряда ARMA(1, 1): y_t = μ + φ*y_(t-1) + ε_t + θ*ε_(t-1)
y = np.zeros(n)
for t in range(1, n):
    y[t] = mu + phi * y[t-1] + errors[t] + theta * errors[t-1]

# Создание DataFrame для удобства
data = pd.DataFrame({'Time': range(n), 'Value': y})
print("Первые 10 строк данных:")
print(data.head(10))

# Визуализация временного ряда
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['Value'], label='Временной ряд (ARMA(1,1))')
plt.title('Сгенерированный временной ряд с использованием модели ARMA(1,1)')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid()
plt.show()

# Построение автокорреляционной функции (ACF)
plot_acf(y, lags=20)
plt.title('ACF временного ряда (ARMA(1,1))')
plt.show()

# Построение модели ARMA(1, 1) с использованием библиотеки statsmodels
# Определение модели ARIMA(p=1, d=0, q=1), что соответствует ARMA(1,1)
model = ARIMA(y, order=(1, 0, 1))
fitted_model = model.fit()

# Вывод параметров модели
print("\nОцененные параметры модели ARMA(1,1):")
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
plt.title('Прогноз временного ряда (ARMA(1,1))')
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
plt.title('Остатки модели ARMA(1,1)')
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

# Вывод таблицы прогнозов
print("\nПрогноз и доверительные интервалы:")
print(forecast_df)