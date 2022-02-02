from tkinter import *

window = Tk()
window.title("House prediction")

Label(window, text="Bedrooms", bg="black", fg="white", font="none 12
bold
",width=15).grid(row=1,column=0,sticky=W)

bedrooms = Entry(window, width=20, bg="white")
bedrooms.grid(row=1, column=1, sticky=W)

Label(window, text="Bathrooms", bg="black", fg="white", font="none 12
bold
",width=15).grid(row=2,column=0,sticky=W)

bathrooms = Entry(window, width=20, bg="white")
bathrooms.grid(row=2, column=1, sticky=W)

Label(window, text="Square ft Living", bg="black", fg="white", font="none 12 bold", width=15).grid(row=3, column=0,
                                                                                                   sticky=W)
sqft_living = Entry(window, width=20, bg="white")
sqft_living.grid(row=3, column=1, sticky=W)

Label(window, text="Square Lot", bg="black", fg="white", font="none 12 bold", width=15).grid(row=4, column=0, sticky=W)
sqft_lot = Entry(window, width=20, bg="white")
sqft_lot.grid(row=4, column=1, sticky=W)
Label(window, text="Floors", bg="black", fg="white", font="none 12 bold", width=15).grid(row=5, column=0, sticky=W)
floors = Entry(window, width=20, bg="white")
floors.grid(row=5, column=1, sticky=W)
Label(window, text="waterfront", bg="black", fg="white", font="none 12 bold", width=15).grid(row=6, column=0, sticky=W)
waterfront = Entry(window, width=20, bg="white")
waterfront.grid(row=6, column=1, sticky=W)
Label(window, text="condition", bg="black", fg="white", font="none 12 bold", width=15).grid(row=7, column=0, sticky=W)
condition = Entry(window, width=20, bg="white")
condition.grid(row=7, column=1, sticky=W)
Label(window, text="grade", bg="black", fg="white", font="none 12 bold", width=15).grid(row=8, column=0, sticky=W)
grade = Entry(window, width=20, bg="white")
grade.grid(row=8, column=1, sticky=W)

Label(window, text="Square ft above", bg="black", fg="white", font="none 12 bold", width=15).grid(row=9, column=0,
                                                                                                  sticky=W)
sqft_above = Entry(window, width=20, bg="white")
sqft_above.grid(row=9, column=1, sticky=W)

Label(window, text="Square ft basement", bg="black", fg="white", font="none 12 bold", width=15).grid(row=10, column=0,
                                                                                                     sticky=W)
sqft_basement = Entry(window, width=20, bg="white")
sqft_basement.grid(row=10, column=1, sticky=W)


def click():


# write the input data and store it in a csv file d1=bedrooms.get()

d2 = bathrooms.get()
d3 = sqft_living.get()
d4 = sqft_lot.get()

d5 = floors.get()

d6 = waterfront.get()
d7 = condition.get()
d8 = grade.get()
d9 = sqft_above.get()
d10 = sqft_basement.get()
import csv

with open('predict.csv', 'w', newline='') as f: dataentry = csv.writer(f, delimiter=",")

dataentry.writerow(('bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'conditi on', 'grade',
                    'sqft_above', 'sqft_basement'))
dataentry.writerow((d1, d2, d3, d4, d5, d6, d7, d8, d9, d10))
import numpy as np

import pandas as pd
import math

import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score  # %matplotlib inline

# read the data from the specific location

data = pd.read_csv(r'C:\Users\Revanth\Desktop\pattern recognition\kc_house_data.csv')
reading_data =

['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'condition', 'grade', 'sqft_above',
 's qft_basement']
x = data[reading_data]
y = data['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

# implement Linear Regression

from sklearn.linear_model import LinearRegression

L_R = LinearRegression()
start = time()

L_R.fit(x_train, y_train)
end = time()

lin_time = end - start
lin_score = L_R.score(x_test, y_test)

lin_predict = L_R.predict(x_test)
var_lin = explained_variance_score(lin_predict, y_test)

# Implement DecisionTreee

from sklearn.tree import DecisionTreeRegressor

D_R = DecisionTreeRegressor()

start = time()
D_R.fit(x_train, y_train)

end = time()
dec_time = end - start

dec_score = D_R.score(x_test, y_test)
dec_predict = D_R.predict(x_test)

var_dec = explained_variance_score(dec_predict, y_test)

# Implement RandomForest Regression
from sklearn.ensemble import RandomForestRegressor

R_R = RandomForestRegressor(n_estimators=400, random_state=0)
start = time()

R_R.fit(x_train, y_train)
end = time()

ran_time = end - start
ran_score = R_R.score(x_test, y_test)

ran_predict = R_R.predict(x_test)
var_ran = explained_variance_score(ran_predict, y_test)

# Implent Adaboost

from sklearn.ensemble import AdaBoostRegressor

start = time()

A_R = AdaBoostRegressor(n_estimators=50, learning_rate=0.2, loss='exponential').fit(x_train, y_train)

end = time()
ada_time = end - start

ada_score = A_R.score(x_test, y_test)
ada_predict = A_R.predict(x_test)

var_ada = explained_variance_score(ada_predict, y_test)

# displays the accuracy score and variance score
final_results = pd.DataFrame({

    'Algorithm': ['linear Regression', 'AdaBoost', 'Random Forest', 'Decision Tree'],
    'Accuracy Score': [lin_score, ada_score, ran_score, dec_score],

    'Variance Score': [var_lin, var_ada, var_ran, var_dec]})

final_results.sort_values(by='Accuracy Score', ascending=False)
print(final_results)

predict = pd.read_csv('predict.csv')
# predicts the price for each model

pre = L_R.predict(predict)
pre1 = D_R.predict(predict)

pre2 = R_R.predict(predict)
pre3 = A_R.predict(predict)

print("price using linear regression")
print((pre))

print("price using decision tree ")
print(pre1)

print("price using Random Forest")
print(pre2)

print("price using Adaboost ")
print(pre3)

# displays the graph for time complexity between each models.

model = ['linear Regression', 'AdaBoost', 'Random Forest', 'Decision Tree']
total_Time = [lin_time, ada_time, ran_time, dec_time]

i = np.arange(len(model))
plt.bar(i, total_Time)

plt.xlabel('Machine Learning Models', fontsize=10)
plt.ylabel('Training Time', fontsize=10)

plt.xticks(i, model, fontsize=6)
plt.title('Comparison of Training Time of all ML models')

plt.show()

# displays the graph of testing data and predicted data comparison of Linear Regression plt.plot(x_test, y_test, "r.", label="Testing Data")

plt.plot(x_test, lin_predict, "g.", label="Predicted Data")
plt.figlegend(loc='lower center', ncol=5, labelspacing=0.)

plt.xlabel("X")
plt.ylabel("y")

plt.title('comparison between testing data and predicted data in linear Regression')
plt.show()

##displays the graph of testing data and predicted data comparison of Adaboost Regression plt.plot(x_test, y_test, "r.", label="Testing Data")
plt.plot(x_test, ada_predict, "g.", label="Predicted Data")

plt.figlegend(loc='lower center', ncol=5, labelspacing=0.)
plt.xlabel("X")

plt.ylabel("y")
plt.title('comparison between testing data and predicted data in Adaboost')

plt.show()

##displays the graph of testing data and predicted data comparison of Random Forest Regression

plt.plot(x_test, y_test, "r.", label="Testing Data")
plt.plot(x_test, ran_predict, "g.", label="Predicted Data")

plt.figlegend(loc='lower center', ncol=5, labelspacing=0.)
plt.xlabel("X")

plt.ylabel("y")
plt.title('comparison between testing data and predicted data Random Forest')

plt.show()

##displays the graph of testing data and predicted data comparison of Decision Tree Regression

plt.plot(x_test, y_test, "r.", label="Testing Data")
plt.plot(x_test, dec_predict, "g.", label="Predicted Data")

plt.figlegend(loc='lower center', ncol=5, labelspacing=0.)
plt.xlabel("X")

plt.ylabel("y")
plt.title('comparison between testing data and predicted data Decision Tree')

plt.show()

Button(window, text="Submit", width=6, command=click).grid(row=16, column=1, sticky=E)
window.mainloop()
