#Particle Swarm Optimization Algorithm

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

#Maximize Objective Function
def function(x1,x2,x3):
    X=[]
    X.append(x1/500.)                        # x1=turbine radius
    X.append(x2/500.)                        # x2=x distance of deflector
    X.append(x3/500.)                       # x3=y distance of deflector 
    X=np.array(X)                           
    X=X.reshape(-1,3)

    fitness=sm.predict_values(X)    
    return fitness[:,0]
    

#Parameters
xmin=np.array([113.625,909,272.7])   
xmax=np.array([454.5,1818, 909]) 
npop=200           #population size
dim=3            #number of variables
c1=1.5
c2=1.5
niter=200        #number of iterations
w=0.4           #inertia weight
r1=random.uniform(0, 1)
r2=random.uniform(0, 1)

#Storing optimum values
optim_values=[]

#Initialize velocity
v=np.zeros([npop,dim])
for i in range(npop):
    for j in range(dim):
        v[i,j]=random.uniform(0, 1)
        
#Initialize position        
position=np.zeros([npop,dim])
for i in range(npop):
    for j in range(dim):
        position[i,j]=random.uniform(xmin[j], xmax[j])

#Compute fitness values
fitness_value=np.zeros([npop,1])    #fitness values
for i in range(npop):
    fitness_value[i,0]=function(position[i,0],position[i,1],position[i,2])
    
p_best=position.copy()

#Best value
g_best=np.partition(fitness_value.flatten(), -2)[-1]     #Group Best value
g_best=np.array([g_best])

position_n=position.copy()
position_g_best=position_n[np.where(fitness_value == g_best[0])[0][0],:]

#Algorithm Loop
for n in range(niter):
    #Velocity update
    vn=v.copy()
    for i in range(npop):
        for j in range(dim):
            v[i,j]=w+vn[i,j]+c1*r1*(p_best[i,j]-position[i,j])+c2*r2*(position_g_best[j]-position[i,j])
           
    #Position update
    for i in range(npop):
        for j in range(dim):
            temp=v[i,j]+position[i,j]
            if temp>=xmin[j] and temp<=xmax[j]:
                position[i,j]=temp            
    
    #Compute updated fitness values
    fitness_value_new=fitness_value.copy()
    for i in range(npop):
        fitness_value_new[i,0]=function(position[i,0],position[i,1],position[i,2])
    
    #Compute updated group best value and position
    g_best_test=np.partition(fitness_value_new.flatten(), -2)[-1]
    g_best_test=np.array([g_best_test])
    
    if g_best_test[0] > g_best[0] :
        position_n=position.copy()
        g_best=g_best_test.copy()
        position_g_best=position_n[np.where(fitness_value_new == g_best[0])[0][0],:]
    
    #Update p best
    for i in range(npop):
        if fitness_value_new[i] > fitness_value[i]:
            for j in range(dim):
                p_best[i,j]=position[i,j]

    #Storing convergence history
    optim_values.append(g_best[0])    
    

#Plotting results
pyplot.figure(figsize=(11,7),dpi=300)
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["axes.linewidth"] = 2
pyplot.plot(optim_values,color='blue',lw=4)
pyplot.xlabel('Iteration [-]',fontsize=20)
pyplot.ylabel('Fitness value (Cp) [-]',fontsize=20)

pyplot.tick_params(axis='both',size=8,labelsize=17,direction='inout')

t=f'PSO Algorithm (w={w}, c1={c1}, c2={c2})'
pyplot.title(t,fontsize=20)


#Final Results
print(f'Max Cp: {g_best[0]}')
print(f'Turbine Diameter: {(position_g_best[0]*2)/1000}')
print(f'Lx: {(position_g_best[1])/1000}')
print(f'Ly: {(position_g_best[2])/1000}')

