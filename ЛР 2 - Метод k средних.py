import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# Генерация данных с тремя кластерами
def generate_data():
    data = []
    # Кластер 1
    data += [(random.gauss(2, 0.5), random.gauss(2, 0.5)) for _ in range(30)]
    # Кластер 2
    data += [(random.gauss(7, 0.5), random.gauss(5, 0.5)) for _ in range(30)]
    # Кластер 3
    data += [(random.gauss(4, 0.5), random.gauss(8, 0.5)) for _ in range(30)]
    return list(zip(*data))  # Возвращаем отдельно x и y


# K-средних с сохранением истории
def k_means_with_history(x, y, k=3, max_iter=20, tol=0.05):
    points = list(zip(x, y))
    centers = random.sample(points, k)
    history = []

    for _ in range(max_iter):
        # Кластеризация
        clusters = [[] for _ in range(k)]
        labels = []
        for point in points:
            distances = [((point[0] - c[0]) ** 2 + (point[1] - c[1]) ** 2) for c in centers]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append(point)
            labels.append(cluster_idx)

        # Сохраняем текущее состояние
        history.append({
            'centers': [c for c in centers],
            'labels': labels,
            'clusters': [c for c in clusters]
        })

        # Обновляем центры
        new_centers = []
        for cluster in clusters:
            if cluster:
                avg_x = sum(p[0] for p in cluster) / len(cluster)
                avg_y = sum(p[1] for p in cluster) / len(cluster)
                new_centers.append((avg_x, avg_y))
            else:
                new_centers.append(centers[len(new_centers)])

        # Проверка сходимости
        max_change = max(((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5
                         for c1, c2 in zip(centers, new_centers))
        if max_change < tol:
            break

        centers = new_centers

    return history


# Создаем интерактивный график
def interactive_plot(x, y, history):
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25)

    # Начальный график
    sc = ax.scatter(x, y, c=history[0]['labels'], cmap='cool', alpha=0.7)
    centers_plot = ax.scatter(
        [c[0] for c in history[0]['centers']],
        [c[1] for c in history[0]['centers']],
        c='red', marker='X', s=100
    )
    ax.set_title(f'Итерация 0/{len(history) - 1}')

    # Создаем слайдер
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax_slider, 'Итерация',
        0, len(history) - 1,
        valinit=0,
        valstep=1
    )

    # Функция обновления
    def update(val):
        iteration = int(slider.val)
        state = history[iteration]

        sc.set_array(state['labels'])
        centers_plot.set_offsets(state['centers'])
        ax.set_title(f'Итерация {iteration}/{len(history) - 1}')
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# Запуск программы
x, y = generate_data()
history = k_means_with_history(x, y, k=3, max_iter=20)
interactive_plot(x, y, history)
