#Grey-Wolf Optimization Algorithm for Savonius-Deflector System

import random
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from matplotlib import pyplot
from smt.surrogate_models import KRG


###################################### Surrogate Model #################################################
#training data
df=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\train3.csv")
#df=df.drop([10,31,34])
#df['y']=df['y']/2
df['y'][40]=df['y'][40]/2
df['y'][38]=df['y'][38]/2
#df['y'][37]=df['y'][37]/2
df['y'][23]=df['y'][23]/2

#Normalizing training Data
df['r']=df['r']/500
df['x']=df['x']/500
df['y']=df['y']/500

#testing data
df_test=pd.read_csv(r"E:\ML\Optimization techniques\Full code surrogate plus opti\Savonius-Deflector-system\Surogate\test3.csv")
#df_test=df_test.drop([5,16]) 
#df_test['y']=df_test['y']/2
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
    

#Algorithm Parameters
dim=3                  #Number of design variables
minx=np.array([113.625,909,272.7])                #Lower Limit
maxx=np.array([454.5,1818, 909])                 #Upper Limit
pop=100                 #Grey-Wolf population
cycles=200             #Number of Iterations


iteration=0            #Initializing iteration counter     
alpha_values=[]        #Storing best values of each iteration


#Random initialization of initial population

position=np.zeros([pop,dim])       #parameter values
for i in range(pop):
    for j in range(dim):
        position[i,j]=random.uniform(minx[j], maxx[j])

fitness_value=np.zeros([pop,1])    #fitness values
for i in range(pop):
    fitness_value[i,0]=function(position[i,0],position[i,1], position[i,2])


#Main Optimization Loop
pyplot.figure(figsize=(11,7),dpi=600)
for i in range(cycles):
    
    #Best value
    alpha=np.partition(fitness_value.flatten(), -2)[-1]     #Alpha wolf
    beta=np.partition(fitness_value.flatten(), -2)[-2]      #Beta wolf
    delta=np.partition(fitness_value.flatten(), -2)[-3]     #Delta wolf
    
    #Position of Best values
    alpha_pos=position[np.where(fitness_value == alpha)[0][0],:]
    beta_pos=position[np.where(fitness_value == beta)[0][0],:]
    delta_pos=position[np.where(fitness_value == delta)[0][0],:]
    
    #Compute 'a'
    a=2*(1-i/cycles)
    
    for j in range(pop):
        #Best
        A1=2*a*random.uniform(0,1)-a
        C1=2*random.uniform(0,1)
        D_alpha=np.abs(C1*alpha_pos-position[j,:])
        X1=alpha_pos-A1*D_alpha
        
        #Second-Best
        A2=2*a*random.uniform(0,1)-a
        C2=2*random.uniform(0,1)
        D_beta=np.abs(C2*beta_pos-position[j,:])
        X2=beta_pos-A2*D_beta
        
        #Third-Best
        A3=2*a*random.uniform(0,1)-a
        C3=2*random.uniform(0,1)
        D_delta=np.abs(C3*delta_pos-position[j,:])
        X3=delta_pos-A3*D_delta
        
        X_new=(X1+X2+X3)/3
        
        #Check bounds
        if (X_new[0]>=minx[0] and X_new[0]<=maxx[0]) and ((X_new[1]>=minx[1] and X_new[1]<=maxx[1]))and ((X_new[2]>=minx[2] and X_new[2]<=maxx[2])):
            #Perform Greedy Selection
            f_X_new=function(X_new[0],X_new[1],X_new[2])
            f_X=fitness_value[j,0]
            
            if f_X_new>f_X:
                position[position==position[j,:]]=X_new
                
                fitness_value[fitness_value==f_X]=f_X_new
        
        
    alpha_values.append(alpha)
    iteration+=1


#Plotting results
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["axes.linewidth"] = 2
pyplot.plot(np.array(alpha_values),color='blue',lw=4)
pyplot.xlabel('Iteration [-]',fontsize=20)
pyplot.ylabel('Fitness value (Cp) [-]',fontsize=20)
pyplot.title('Grey-Wolf Algorithm',fontsize=20)
pyplot.tick_params(axis='both',size=8,labelsize=17,direction='inout')
            
#Final Results
print(f'Max Cp: {alpha_values[-1]}')
print(f'Turbine Diameter: {(alpha_pos[0]*2)/1000}')
print(f'Lx: {(alpha_pos[1])/1000}')
print(f'Ly: {(alpha_pos[2])/1000}')
