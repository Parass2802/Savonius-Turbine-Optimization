
from qpso import QDPSO
import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot
from smt.surrogate_models import KRG


###################################### Surrogate Model #################################################
#training data
df=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\train3.csv")

#Normalizing training Data
df['r']=df['r']/500
df['x']=df['x']/500
df['y']=df['y']/500

#testing data
df_test=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\test3.csv")

#Normalizing testing Data
df_test['r']=df_test['r']/500
df_test['x']=df_test['x']/500
df_test['y']=df_test['y']/500

#Dividing data
X_train=df.drop('m',axis=1)
Y_train=df['m']
X_test=df_test.drop('m',axis=1)
Y_test=df_test['m']
                   

sm=KRG(theta0=[20])
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

###################################### Optimization Algorithm #################################################

def function(args):
    X=[]
    X.append(args[0]/500.)                        # x1=turbine radius
    X.append(args[1]/500.)                        # x2=x distance of deflector
    X.append(args[2]/500.)                       # x3=y distance of deflector 
    X=np.array(X)                           
    X=X.reshape(-1,3)
    f=sm.predict_values(X) 
    return -f[0]

def log(s):
    best_value = [p.best_value for p in s.particles()]
    best_value_avg = np.mean(best_value)
    best_value_std = np.std(best_value)
    #print("{0: >5}  {1: >9}  {2: >9}  {3: >9}".format("Iters.", "Best", "Best(Mean)", "Best(STD)"))
    #print("{0: >5}  {1: >9.3E}  {2: >9.3E}  {3: >9.3E}".format(s.iters, s.gbest_value, best_value_avg, best_value_std))
    
    optim.append(s.gbest_value)
    

NParticle = 100
MaxIters = 100
NDim = 3
bounds = [(272.7, 1818) for i in range(0, NDim)]
g = 0.96
s = QDPSO(function, NParticle, NDim, bounds, MaxIters, g)


optim=[]

s.update(callback=log, interval=1)
print("Best position: {0}".format(s.gbest))
print("Best value: {0}".format(-s.gbest_value))

#Plotting results
pyplot.figure(figsize=(11,7),dpi=500)
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["axes.linewidth"] = 2
pyplot.plot(-np.array(optim),color='blue',lw=4)
pyplot.xlabel('Iteration [-]',fontsize=20)
pyplot.ylabel('Fitness value (Cp) [-]',fontsize=20)
pyplot.title('Quantum PSO Algorithm',fontsize=20)
pyplot.tick_params(axis='both',size=8,labelsize=17,direction='inout')

#Final Results
print(f'Max Cp: {-s.gbest_value[0]}')
print(f'Turbine Diameter: {(s.gbest[0]*2)/1000}')
print(f'Lx: {(s.gbest[1])/1000}')
print(f'Ly: {(s.gbest[2])/1000}')