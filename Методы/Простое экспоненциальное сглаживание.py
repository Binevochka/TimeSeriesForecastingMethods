import numpy as np
import matplotlib.pyplot as plt


def simple_exponential_smoothing(data, alpha):
    """
    Реализация метода простого экспоненциального сглаживания (SES).

    :param data: Массив данных (временной ряд)
    :param alpha: Параметр сглаживания (0 < alpha <= 1)
    :return: Сглаженные значения (массив)
    """
    smoothed = np.zeros_like(data)
    smoothed[0] = data[0]  # Начальное значение равно первому наблюдению

    # Применение формулы SES
    for t in range(1, len(data)):
        smoothed[t] = alpha * data[t] + (1 - alpha) * smoothed[t - 1]

    return smoothed


# Генерация примера данных
np.random.seed(42)
time = np.arange(1, 51)  # 50 временных точек
data = 50 + np.random.normal(0, 5, size=len(time))  # Данные с шумом

# Параметр сглаживания
alpha = 0.3

# Применение SES
smoothed_data = simple_exponential_smoothing(data, alpha)

# Визуализация
plt.figure(figsize=(10, 6))
plt.plot(time, data, label="Исходные данные", color="blue", linestyle="--", alpha=0.7)
plt.plot(time, smoothed_data, label=f"Сглаженные данные (α={alpha})", color="red")
plt.title("Простое экспоненциальное сглаживание")
plt.xlabel("Время")
plt.ylabel("Значение")
plt.legend()
plt.grid()
plt.show()