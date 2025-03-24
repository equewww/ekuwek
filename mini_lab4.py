import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("датасет-1.csv",sep=';')
print(df)
print(df.dtypes)

df['price'] = df['price'].str.replace(',', '.')
df['price'] = df['price'].astype(float)

plt.scatter(df["area"], df["price"], color="hotpink")
plt.xlabel("Площадь (кв.м.)")
plt.ylabel("Цена (млн руб.)")
plt.show()

reg = linear_model.LinearRegression()
reg.fit(df[['area']], df['price'])

print('Стоимость квартиры 38 м^2',reg.predict([[38]]))
print('Стоимость квартиры 200 м^2',reg.predict([[200]]))
print('Предсказанные цены',reg.predict(df[['area']]))

print("Коэффициент наклона (a):", reg.coef_[0])
print("Свободный член (b):", reg.intercept_)

df["predicted_price"] = reg.predict(df[['area']])

plt.scatter(df["area"], df["price"], color="pink")
plt.plot(df["area"], reg.predict(df[['area']]), color="hotpink")
plt.xlabel("Площадь (кв.м.)")
plt.ylabel("Цена (млн руб.)")
plt.show()

prediction_data = pd.read_csv("prediction_price.csv", sep=";")
prediction_data["area"] = prediction_data["area"].astype(float)

predictions = reg.predict(prediction_data[['area']])
prediction_data["predicted_price"] = predictions
print(prediction_data.head())

prediction_data.to_excel("new_predictions.xlsx", index=False)
print("Результаты сохранены в файл new_predictions.xlsx")