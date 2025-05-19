import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import random


# 1. Генерация исходной функции и данных
def f(x):
    return np.exp(x ** 2 + 3 * x + 2)


x_min, x_max = -2, 1
x = np.linspace(x_min, x_max, 100).reshape(-1, 1)
y_true = f(x).flatten()
noise = np.array([random.uniform(-1, 1) for _ in range(100)])
y = y_true + noise

# 2. Методы регрессии
models = {
    "Kernel Ridge": KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1),
    "SVR": SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
}

# 3. Обучение и предсказание
plt.figure(figsize=(15, 5))
for i, (name, model) in enumerate(models.items(), 1):
    model.fit(x, y)
    y_pred = model.predict(x)
    mse = mean_squared_error(y_true, y_pred)

    plt.subplot(1, 3, i)
    plt.scatter(x, y, color='blue', label='Исходные точки', s=10)
    plt.plot(x, y_true, color='green', label='Исходная функция')
    plt.plot(x, y_pred, color='red', label=f'Предсказание ({name})')
    plt.title(f"{name}\nMSE: {mse:.2f}")
    plt.legend()

plt.tight_layout()
plt.show()