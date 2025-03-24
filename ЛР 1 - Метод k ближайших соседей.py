import random
import math
import matplotlib.pyplot as plt


# 1. Инициализация исходных данных
def generate_data(pointsCount1, pointsCount2, xMin1, xMax1, yMin1, yMax1, xMin2, xMax2, yMin2, yMax2):
    x = []
    y = []

    # Генерация точек для первого класса
    for _ in range(pointsCount1):
        x.append([random.uniform(xMin1, xMax1), random.uniform(yMin1, yMax1)])
        y.append(0)

    # Генерация точек для второго класса
    for _ in range(pointsCount2):
        x.append([random.uniform(xMin2, xMax2), random.uniform(yMin2, yMax2)])
        y.append(1)

    return x, y


# 2. Разбиение данных на обучающую и тестовую выборки
def train_test_split(x, y, p):
    combined = list(zip(x, y))  # Объединяем x и y в пары (координата + метка)
    random.shuffle(combined)
    x_shuffled = []
    y_shuffled = []
    for i in combined:
        x_shuffled.append(i[0])
        y_shuffled.append(i[1])

    split_index = int(len(x_shuffled) * p) # Индекс для разделения
    x_train = x_shuffled[:split_index]
    x_test = x_shuffled[split_index:]
    y_train = y_shuffled[:split_index]
    y_test = y_shuffled[split_index:]

    return x_train, x_test, y_train, y_test


# 3. Реализация метода k ближайших соседей
def fit(x_train, y_train, x_test, k):
    y_predict = []

    for test_point in x_test:
        distances = []
        for i in range(len(x_train)):
            train_point = x_train[i]
            dist = math.sqrt((test_point[0] - train_point[0]) ** 2 + (test_point[1] - train_point[1]) ** 2)
            distances.append((dist, y_train[i]))

        for i in range(len(distances)):
            for j in range(0, len(distances) - 1):
                if distances[j][0] > distances[j + 1][0]:
                    distances[j], distances[j + 1] = distances[j + 1], distances[j]

        # Выбор k ближайших
        k_nearest_classes = []
        for i in range(k):
            k_nearest_classes.append(distances[i][1])

        # Подсчет классов
        votes = [0] * 2 # votes[0] - голоса за 0 класс, votes[1] - голоса за 1 класс
        for cclass in k_nearest_classes:
            votes[cclass] += 1

        max_votes = -1
        predicted_class = -1
        for i in range(0,2):
            if votes[i] > max_votes:
                max_votes = votes[i]
                predicted_class = i

        y_predict.append(predicted_class)

    return y_predict


# 4. Реализация функции для расчета accuracy
def computeAccuracy(y_test, y_predict):
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            correct += 1
    accuracy = correct / len(y_test)
    return accuracy


# 5. Визуализация результата работы алгоритма
def results(x_train, y_train, x_test, y_test, y_predict):
    for i in range(len(x_train)):
        if y_train[i] == 0:
            # Класс 0: синие кружки
            plt.scatter(x_train[i][0], x_train[i][1], c='blue', marker='o')
        else:
            # Класс 1: синие крестики
            plt.scatter(x_train[i][0], x_train[i][1], c='blue', marker='x')

    for i in range(len(x_test)):
        if y_test[i] == y_predict[i]:
            if y_test[i] == 0:
                # Класс 0: зеленые кружки
                plt.scatter(x_test[i][0], x_test[i][1], c='green', marker='o')
            else:
                # Класс 1: зеленые крестики
                plt.scatter(x_test[i][0], x_test[i][1], c='green', marker='x')
        else:
            if y_test[i] == 0:
                # Класс 0: красные кружки
                plt.scatter(x_test[i][0], x_test[i][1], c='red', marker='o')
            else:
                # Класс 1: красные крестики
                plt.scatter(x_test[i][0], x_test[i][1], c='red', marker='x')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Метод k-ближайших соседей')
    plt.show()

# Основной код

# Параметры
pointsCount1 = 50
pointsCount2 = 50
xMin1, xMax1, yMin1, yMax1 = 0, 5, 0, 5
xMin2, xMax2, yMin2, yMax2 = 5, 10, 5, 10
k = 3
p = 0.8

# Генерация данных
x, y = generate_data(pointsCount1, pointsCount2, xMin1, xMax1, yMin1, yMax1, xMin2, xMax2, yMin2, yMax2)

# Разделение данных
x_train, x_test, y_train, y_test = train_test_split(x, y, p)

# Обучение и предсказание
y_predict = fit(x_train, y_train, x_test, k)

# Оценка точности
accuracy = computeAccuracy(y_test, y_predict)
print('Accuracy:', accuracy)

# Визуализация
results(x_train, y_train, x_test, y_test, y_predict)