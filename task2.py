"""Минимальный элемент заданного одномерного массива увеличить в 2 раза"""
file= open('m.txt')
s = file.readlines()
file.close()
print(s)
m=[]
for i in s:
    if i != '':
        print(i.strip())
        for ii in i.split():
            m.append(int(ii))
min_value = min(m)
min_index = m.index(min_value)
m[min_index] = 2 * min_value
file= open('m.txt', 'w')
for k in m:
    file.write(str(k) + '\n')
file.close()
print('Ответ:', 2 * min_value)
