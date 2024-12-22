import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Генерация синтетического временного ряда
np.random.seed(42)  # Для воспроизводимости
n = 250  # Количество точек
noise = np.random.normal(0, 1, n)  # Случайный шум
y = np.zeros(n)
for t in range(1, n):
    y[t] = 0.7 * y[t - 1] + noise[t]

# Создание DataFrame для удобства работы
data = pd.DataFrame({'Time': range(n), 'Value': y})
print("Первые 10 строк данных:")
print(data.head(10))  # Вывод первых 10 строк таблицы

# Визуализация временного ряда
plt.figure(figsize=(10, 5))
plt.plot(data['Time'], data['Value'], label='Time Series')
plt.title('Временной ряд')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid()
plt.show()

# Проверка стационарности ряда (тест Дики-Фуллера)
result = adfuller(y)
print("\nРезультаты теста Дики-Фуллера:")
print(f"Статистика ADF: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
print(f"Критические значения:")
for key, value in result[4].items():
    print(f"  {key}: {value:.4f}")

if result[1] < 0.05:
    print("\nРяд стационарен (p-value < 0.05).")
else:
    print("\nРяд не стационарен (p-value >= 0.05).")

# Если ряд не стационарен, применим дифференцирование
if result[1] >= 0.05:
    y_diff = np.diff(y, n=1)  # Первый порядок разности
    print("\nПосле дифференцирования ряда:")

    # Проверка стационарности после дифференцирования
    diff_result = adfuller(y_diff)
    print(f"Статистика ADF: {diff_result[0]:.4f}")
    print(f"p-value: {diff_result[1]:.4f}")
    print(f"Критические значения:")
    for key, value in diff_result[4].items():
        print(f"  {key}: {value:.4f}")

    if diff_result[1] < 0.05:
        print("\nДифференцированный ряд стационарен.")
    else:
        print("\nДифференцированный ряд не стационарен.")

# Построение модели авторегрессии
p = 1  # Порядок модели AR
model = AutoReg(y, lags=p).fit()
print("\nОцененные параметры модели AR:")
print(model.params)

# Прогнозирование
n_forecast = 10  # Количество точек для прогноза
forecast = model.predict(start=len(y), end=len(y) + n_forecast - 1)

# Визуализация прогноза
plt.figure(figsize=(10, 5))
plt.plot(range(n), y, label='Оригинальные данные')
plt.plot(range(len(y), len(y) + n_forecast), forecast, label='Прогноз', color='red')
plt.title('Прогноз временного ряда')
plt.xlabel('Время')
plt.ylabel('Значение')
plt.legend()
plt.grid()
plt.show()

# Вывод последних значений данных и прогноза
forecast_df = pd.DataFrame({'Time': range(len(y), len(y) + n_forecast), 'Forecast': forecast})
print("\nПоследние значения данных:")
print(data.tail(5))
print("\nПрогноз на следующие точки:")
print(forecast_df)