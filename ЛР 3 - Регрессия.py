import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# 1. Показательная функция y = a*b^x + c
def exp_func(x, a, b, c):
    return a * (b ** x) + c


# 2. Генерация данных
np.random.seed(207)
x = np.linspace(0, 5, 50)
true_a, true_b, true_c = 2.0, 1.5, 1.0
y = exp_func(x, true_a, true_b, true_c) + np.random.uniform(-1, 1, len(x))


# 3. Частные производные
def get_da(x, y, a, b, c):
    return -2 * sum((b ** x) * (y - exp_func(x, a, b, c)))


def get_db(x, y, a, b, c):
    return -2 * sum(a * x * (b ** (x - 1)) * (y - exp_func(x, a, b, c)))


def get_dc(x, y, a, b, c):
    return -2 * sum(y - exp_func(x, a, b, c))


# 4. Инициализация параметров
a, b, c = 1.0, 1.1, 0.0
speed = 0.00001
epochs = 2000

# 5. Градиентный спуск
a_history = [a]
b_history = [b]
c_history = [c]

for _ in range(epochs):
    da = get_da(x, y, a, b, c)
    db = get_db(x, y, a, b, c)
    dc = get_dc(x, y, a, b, c)

    a -= speed * da
    b -= speed * db
    c -= speed * dc

    a_history.append(a)
    b_history.append(b)
    c_history.append(c)

# 6. Визуализация
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

ax.scatter(x, y, color='blue', label='Данные')
line, = ax.plot(x, exp_func(x, a_history[0], b_history[0], c_history[0]),
                'r-', linewidth=2, label='Регрессия')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Показательная регрессия: y = a*bˣ + c')
ax.grid(True)
ax.legend()

ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Итерация', 0, epochs, valinit=0, valstep=1)


def update(val):
    epoch = int(slider.val)
    line.set_ydata(exp_func(x, a_history[epoch], b_history[epoch], c_history[epoch]))
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.show()

print(f"Найденные параметры: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}")
print(f"Истинные параметры: a = {true_a}, b = {true_b}, c = {true_c}")