import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Генерация данных для кластеризации
np.random.seed(7)
X, Y = make_blobs(n_samples=500, centers=4, cluster_std=1.8) # Набор данных из 500 точек, который состоит из 4 кластеров
                                                             # Стандартное отклонение для каждого кластера равно 1.8
                                                             # X - координаты точек, Y - метки кластеров

# Визуализация данных
plt.scatter(X[:,0],X[:,1], alpha=0.5)
plt.title("Исходные данные")
plt.show()

# Создание модели
kmeans = KMeans(n_clusters=4)

# Обучение
kmeans.fit(X)

# Получение меток кластеров и центров
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Визуализация результатов
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='cool', alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("Результаты кластеризации")
plt.show()

# Вывод центроидов
print("Центры кластеров:")
print(centers)

