#ANN Surrogate model for multi elemet wing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from matplotlib import pyplot

#training data
df=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\train3.csv")

#df=df.drop([10,31,34])

#Normalizing training Data
df['r']=df['r']/500
df['x']=df['x']/500
df['y']=df['y']/500

#testing data
df_test=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\test3.csv")

#df_test=df_test.drop([5,16])

#Normalizing testing Data
df_test['r']=df_test['r']/500
df_test['x']=df_test['x']/500
df_test['y']=df_test['y']/500

#Dividing data
X_train=df.drop('m',axis=1)
Y_train=df['m']
X_test=df_test.drop('m',axis=1)
Y_test=df_test['m']

model = keras.Sequential([
    keras.layers.Dense(30, input_shape=(3,), activation='relu'),
    keras.layers.Dense(90, activation='relu'),
    keras.layers.Dense(60, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=2000)

model.evaluate(X_test,Y_test)

Y_predicted=model.predict(np.array(X_test))

# Calculate R-squared score for the predictions
r2 = r2_score(Y_predicted, Y_test)
print(f"R-squared score: {r2}")

#plotting r2 score
pyplot.figure(figsize=(11,7),dpi=500)
pyplot.plot(Y_test,Y_test,color='black')
pyplot.scatter(Y_predicted,Y_test,color='red',marker='o')
pyplot.xlabel('Predicted $C_m [-]$',fontsize=20)
pyplot.ylabel('Actual $C_m [-]$',fontsize=20)
pyplot.title(f'Kriging surrogate model, $R^2$ score={r2:.2f}',fontsize=20)
pyplot.tick_params(axis='both', size=8,labelsize=17,direction='inout')
