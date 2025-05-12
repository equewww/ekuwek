import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Создание фигуры для отображения результатов
fig, axes = plt.subplots(5, 3, figsize=(18, 25))
fig.suptitle('Сравнение методов классификации', y=1.02, fontsize=16)

# Генерация данных
n_samples = 500
seed = 30

# 1. Две окружности
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed)
X1, y1 = noisy_circles

# 2. Две параболы
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed)
X2, y2 = noisy_moons

# 3. Хаотичное распределение
cluster_std = [1.0, 0.5]
varied = datasets.make_blobs(n_samples=n_samples, cluster_std=cluster_std, random_state=seed, centers=2)
X3, y3 = varied

# 4. Точки вокруг прямых
random_state = 170
X4, y4 = datasets.make_blobs(n_samples=n_samples, random_state=random_state, centers=2)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X4_aniso = np.dot(X4, transformation)
X4, y4 = (X4_aniso, y4)

# 5. Слабо пересекающиеся области
blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed, centers=2)
X5, y5 = blobs

datasets = [
    ('Две окружности', X1, y1),
    ('Две параболы', X2, y2),
    ('Хаотичное распределение', X3, y3),
    ('Точки вокруг прямых', X4, y4),
    ('Слабо пересекающиеся области', X5, y5)
]

classifiers = [
    ('KNN (k=3)', KNeighborsClassifier(n_neighbors=3)),
    ('SVM (RBF)', SVC(kernel='rbf', C=1.0)),
    ('Decision Tree', DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42))
]

# Обучение и визуализация
for row, (ds_name, X, y) in enumerate(datasets):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Создание сетки для границ решений
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    for col, (clf_name, model) in enumerate(classifiers):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Предсказание для сетки
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        ax = axes[row, col]
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

        # Обучающие точки (синие)
        for i in range(len(X_train)):
            marker = 'x' if y_train[i] == 0 else 'o'
            ax.scatter(X_train[i, 0], X_train[i, 1], marker=marker, c='blue', alpha=0.5)

        # Тестовые точки (зеленые - верно, красные - ошибки)
        for i in range(len(X_test)):
            marker = 'x' if y_test[i] == 0 else 'o'
            color = 'green' if y_test[i] == y_pred[i] else 'red'
            ax.scatter(X_test[i, 0], X_test[i, 1], marker=marker, c=color)

plt.tight_layout()
plt.show()