import numpy as np
import pandas as pd
from matplotlib import pyplot

#Inputs
TSR=0.9

training_data_size=30
testing_data_size=10

#Result data (Cp)
result_training=[]
result_testing=[]


for i in range(testing_data_size):
    #Read data
    df=pd.read_csv('moment-rfile_dp'+str(i)+'.out',skiprows=2,delimiter="\t")


    #Structure Data
    df.columns=['Info']
    new = df["Info"].str.split(" ", n = 0, expand = True)
    df["time_step"]= new[0]
    df["moment"]= new[1]
    df["flow_time"]= new[2]
    df.drop(columns =["Info"], inplace = True)
    df = df.astype(float)


    #Post-process
    y_l=df['moment'][3241:]                            #last rev
    x_l=df['time_step'][3241:]

    y_sl=df['moment'][2881:3241]                       #second last rev
    x_sl=df['time_step'][2881:3241]                         

    angle=np.arange(1,361)
    pyplot.figure(figsize=(11,7),dpi=300)
    pyplot.plot(angle,y_l,label='10th rev')
    pyplot.plot(angle,y_sl,label='9th rev')
    pyplot.legend()
    pyplot.title('DP'+str(i))

    moment=y_l.mean()

    constant=0.5*1.225*0.909*49*(0.909/2)

    cm=moment/constant
    cp=cm*TSR
    result_testing.append(cp)

#Save Results
(np.array(result_testing).T).tofile('testing_results.csv', sep = ',')