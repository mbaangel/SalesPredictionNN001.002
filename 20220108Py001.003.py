import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Data Import
sales_df = pd.read_csv("datos_de_ventas.csv")

#Visualization
sns.scatterplot(sales_df['Temperature'], sales_df['Revenue'])

#Creating Set Training
X_train = sales_df['Temperature']
Y_train = sales_df['Revenue']

#Model Creating
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#Summary Visualization
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')

#Training Model
epochs_hist = model.fit(X_train, Y_train, epochs =1000)

keys = epochs_hist.history.keys()

#Model Training Graphic
plt.plot(epochs_hist.history['loss'])
plt.title('Training Loss Progress')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

#Weights 
weights = model.get_weights()
print(weights)

#Predicción
Temp = 5
Revenue = model.predict([Temp])
print('Earnings due NR will be: ', Revenue)

Temp = 30
Revenue = model.predict([Temp])
print('Earnings due NR will be: ', Revenue)

Temp = 35
Revenue = model.predict([Temp])
print('Earnings due NR will be: ', Revenue)

#Prediction Graphic
plt.scatter(X_train, Y_train, color='gray')
plt.plot(X_train, model.predict(X_train), color = 'red')
plt.ylabel('Earnings [USD]')
plt.xlabel('Temperature [gCelsius]')
plt.title('Earnings Generated vs Temperature @Company')