import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, Birch
from sklearn.preprocessing import StandardScaler
import hdbscan
from sklearn.datasets import make_blobs, make_moons, make_circles
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(207)
methods = np.random.choice(range(1, 12), size=3, replace=False)
print("Выбранные методы: ", methods)


# 1. 6 алгоритмов генерации точек
def generate_datasets():
    np.random.seed(42)

    # 1. Изотропные гауссианы (шары)
    blobs = make_blobs(n_samples=300, centers=3, random_state=42)

    # 2. Полумесяцы
    moons = make_moons(n_samples=300, noise=0.05, random_state=42)

    # 3. Круги
    circles = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=42)

    # 4. Анизотропные данные (вытянутые)
    X, y = make_blobs(n_samples=300, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    anisotropic = (np.dot(X, transformation), y)

    # 5. Разные дисперсии
    varied = make_blobs(n_samples=300, cluster_std=[1.0, 2.5, 0.5], random_state=42)

    # 6. Без структуры (шум)
    noise = (np.random.rand(300, 2), None)

    return [
        ("1. Изотропные шары", blobs),
        ("2. Полумесяцы", moons),
        ("3. Круги", circles),
        ("4. Анизотропные", anisotropic),
        ("5. Разные дисперсии", varied),
        ("6. Без структуры", noise)
    ]


# 2. 3 метода кластеризации
def apply_hdbscan(X):
    X = StandardScaler().fit_transform(X)
    return hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(X)


def apply_optics(X):
    X = StandardScaler().fit_transform(X)
    return OPTICS(min_samples=10, xi=0.05).fit_predict(X)


def apply_birch(X):
    X = StandardScaler().fit_transform(X)
    return Birch(n_clusters=3).fit_predict(X)


# 3. Визуализация
def visualize_results(datasets):
    plt.figure(figsize=(18, 12))

    for row, (name, (X, y)) in enumerate(datasets):
        # Оригинальные данные
        plt.subplot(6, 4, row * 4 + 1)
        plt.scatter(X[:, 0], X[:, 1], c=y if y is not None else 'gray', s=10)
        if row == 0: plt.title("Исходные данные")
        plt.ylabel(name, rotation=0, ha='right')

        # Применяем методы кластеризации
        methods = [
            ("HDBSCAN", apply_hdbscan(X)),
            ("OPTICS", apply_optics(X)),
            ("BIRCH", apply_birch(X))
        ]

        for col, (method_name, labels) in enumerate(methods, start=2):
            plt.subplot(6, 4, row * 4 + col)
            plt.scatter(X[:, 0], X[:, 1], c=labels, s=10, cmap='viridis')
            if row == 0: plt.title(method_name)

    plt.tight_layout()
    plt.show()

datasets = generate_datasets()
visualize_results(datasets)

print("\nАнализ результатов:")
print("1. HDBSCAN:")
print("   - Лучше всего работает с кластерами произвольной формы")
print("   - Автоматическое определение числа кластеров")
print("   - Выделяет шумовые точки (метка -1)")

print("\n2. OPTICS:")
print("   - Хорош для кластеров разной плотности")
print("   - Обнаруживает иерархическую структуру")
print("   - Требует аккуратной настройки параметров")

print("\n3. BIRCH:")
print("   - Эффективен для больших данных с выпуклыми кластерами")
print("   - Быстрая обработка данных")
print("   - Не подходит для сложных форм кластеров")
