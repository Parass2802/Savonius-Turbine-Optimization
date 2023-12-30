#IDW Surrogate model for GF Data

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot
from smt.surrogate_models import IDW

#training data
df=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\train3.csv")

#df=df.drop([31,34])
#df['y'][40]=df['y'][40]/2
#df['y'][38]=df['y'][38]/2
#df['y'][37]=df['y'][37]/2
#df['y'][23]=df['y'][23]/2
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
                   

sm=IDW(p=X_train.shape[0])
sm.set_training_values(np.array(X_train),np.array(Y_train))
sm.train()

Y_predicted=sm.predict_values(np.array(X_test))

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



