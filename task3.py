'Напишите программу, строящую график функции. Коэффициенты a,b,c и диапазон задаются с клавиатуры.'
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import math
random.seed(207)
n = random.randint(1, 24)
print('Номер варианта:', n, '\n')
print('Первое задание:')
print('f(x)=a+b·x^2+c·ln x')
m = []
m = input('Введите a, b, c через пробел: ').split()
a, b ,c = m[0], m[1], m[2]
X =[]
Y = []
for x in range(1, 10):
    y = float(a) + float(b)*(x**2) + float(c)*(float(math.log(x)))
    X.append(x)
    Y.append(y)
plt.plot([x for x in X], [y for y in Y], color='pink')
plt.show()


'Напишите программу построения графика по имеющемуся дискретному набору известных значений.'
print('\n' + 'Второе задание:')
with open('C:/Users/equewww/Desktop/ww.txt') as file:
    s = file.read().splitlines()
    X2 = []
    Y2= []
    for i in s[0].split():
        X2.append(float(i))
    for i in s[1].split():
        Y2.append(float(i))
print(X2, '\n' + str(Y2))
plt.plot([i for i in X2], [i for i in Y2], color='hotpink')
plt.show()

