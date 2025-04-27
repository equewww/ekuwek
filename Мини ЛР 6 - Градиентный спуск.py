import matplotlib.pyplot as plt
import math


# Определяем функцию и её производную
def f(x):
    return math.sin(x) + 0.3 * x ** 2  # Функция с несколькими минимумами


def df(x):
    return math.cos(x) + 0.6 * x  # Производная


# Градиентный спуск
def gradient_descent(start_x, lr, steps):
    x = start_x
    path = []

    for _ in range(steps):
        y = f(x)
        path.append((x, y))
        x = x - lr * df(x)  # Основная формула

    return path


# Параметры
start_x = -2.5  # Начальная точка
lr = 0.1  # Скорость обучения
steps = 30  # Шагов


path = gradient_descent(start_x, lr, steps)

x_vals = [i / 10 for i in range(-40, 40)]
y_vals = [f(x) for x in x_vals]
plt.figure(figsize=(10, 5))
plt.plot(x_vals, y_vals, 'b-', label='f(x) = sin(x) + 0.3x²')
plt.plot([p[0] for p in path], [p[1] for p in path], 'ro-', label='Градиентный спуск')
plt.scatter(path[-1][0], path[-1][1], s=100, c='g', label='Конечная точка')
plt.legend()
plt.grid(True)
plt.show()

print(f"Начальная точка: {start_x}")
print(f"Конечная точка: x={path[-1][0]:.3f}, y={path[-1][1]:.3f}")