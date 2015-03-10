#!/usr/bin/python
'''
Created on Oct 7, 2014

@author: cadances
'''


import math
import os,sys
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from matplotlib.pylab import polyfit
import matplotlib.ticker as mtick

import numpy as np
from CTF_parser import *
from plot_saved_data import clean_directory

def temporal_scaling(file_name,data_dir,subfolder_pre,N_steps,R,Nt_compare,output_dir="results"):
    
    ## Initial Error Checks
    if(N_steps < 4):
        print "Must have at least three steps"
        print N
        sys.exit(-1)
    if(R<2):
        print "Ratio must be integer greater or equal to 2!"
        print R
        sys.exit()
    
    # Constant Parameters
    
    tol=1.0e-2
    path = os.getcwd()
    path="/home/dances/Workspace/CTF-inputs/shock_tube"
    data_dir=os.path.join(path,data_dir)
    output_dir=os.path.join(path,output_dir)
    # Initialize Data Structure
    V_data={}
    E_data={}
    O_data={}
    for variable in qoi:
        V_data[variable]=[]
        E_data[variable]=[]
        O_data[variable]=[]
    
    h=[]
    R=int(R)
    
    print "Reading Data"
    for i in range(N_steps):
        
        f_name= os.path.join(data_dir,subfolder_pre+str(i+1),file_name)
        print f_name
        Data = CTF_H5_interface(f_name)
#         Data.dt=1.7903510880009339E-004
        
        ## Time Data
#         Time_data= Data.get_time_steps()
#         ## Transient Information
#         Height = np.arange(0,Data.Nz)*Data.dz+Data.dz/2.0

        h.append(Data.dt)

        if(i>0):
            if( abs(h[-2]/h[-1]-R)>tol):
                print "R value ", R,"does not match", h[-2]/h[-1]
                print h
                sys.exit()
            print h[-2]/h[-1]
        else:
            
            time_compare = Data.dt*Nt_compare
        
#             time_compare = 1.7903510880009339E-004*Nt_compare
#             print time_compare, Data.dt, Nt_compare
            

        for variable in qoi:
            V_data[variable].append(Data.get_array_at_time(Nt_compare*R**(i),variable))
        
#         print "Nt =",Data.Nt,"dt =",Data.dt
#         print "Nz =",Data.Nz,"dz =",Data.dz
        
        Data.close_file()
        
    print "Performing Temporal Analysis at time {0:e} sec".format( time_compare )
        
    ## Calculate the difference between each iteration
    print "Calculating Differences between current and previous run"
    h_err=[]
    N=len(V_data[variable])-1
    for variable in E_data:
        print "\t",variable
        for i in range(N):
            err_i=np.sum(np.abs(V_data[variable][i+1]-V_data[variable][i]))
            E_data[variable].append(err_i)
        
        X=np.array(h[1:])
        Y=np.array(E_data[variable])
        
#         print np.shape(X), np.shape(Y)
#         print Y
        m,b=polyfit(np.log(X),np.log(Y),1)
        Y_fit=np.power(X,m)*math.exp(b)
        ss_res = np.dot(Y_fit-Y,Y_fit-Y)
        Y_mean=np.mean(Y)
        ss_tot = np.dot(Y-Y_mean,Y-Y_mean)
        r_sq=1.0-ss_res/ss_tot
        fig = plt.figure()
        ax= fig.add_subplot(111)
        
        ax.set_title("L1 Normalized difference between steps at {0:.3f} sec".format(time_compare))
        ax.set_ylabel(field[variable])
        ax.set_xlabel("Time Step Size [sec]")
        ax.loglog(X,Y,'ko',label=variable)
        ax.loglog(X,Y_fit,'k-',label="{0:.3f}x^{1:.4f}\nR^2={2:.5f}".format(math.exp(b),m,r_sq))
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
        plt.savefig(os.path.join(output_dir,"Difference_{0:s}.jpg".format(variable)))
        plt.close()

#     print "Printing Values"
#     for variable in E_data:
#         print "Variable",variable
#         for i,value in enumerate(E_data[variable]):
#             print "{0:.16f} {1:.16e}".format(h[i+1],value)
    
    print "Calculating Order of Accurracy" 
#     plt.figure()
    for variable in O_data:
        print "\t",variable
        for i in range(len(E_data[variable])-2):
            f1=E_data[variable][i+2]
            f2=E_data[variable][i+1]
            f3=E_data[variable][i+0]
            O_data[variable].append(math.log((f3-f2)/(f2-f1))/math.log(R))
        fig = plt.figure()
        ax= fig.add_subplot(111)
        ax.plot(h[2:-1],O_data[variable],'k--o',label=variable)
        ax.set_title("{0:s} at {1:.3f} sec".format(field[variable],time_compare))
        ax.set_ylabel("Temporal Order of Accurracy")
        ax.set_xlabel("Time Step Size [sec]")
        ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))
#         plt.legend("upper left")
        plt.savefig(os.path.join(output_dir,"Temporal_Order_Of_Accuracy_{0:s}.jpg".format(variable)))
        plt.close()
    
if __name__=="__main__":
    clean_directory("results.txt","results")
    
    data_dir = "temporal_scaling" # Data Directory containsing subfolders
    subfolder_pre="t" # The prefix to the subfolders. Suffix is simply step number starting at 1.
    file_name="shock_tube.ctf.h5" # File name in each subfolder
    N_steps=7 # Number of data sets. At least 4 are needed.
    R=2 # Ratio between time step sizes. Must be an integer > 2. Should be 2.
    Nt_compare=20 # The initial time step value to compare to. (dt_0*Nt_compare=t_compare)
#     print file_name,data_dir,subfolder_pre,N_steps,R
    temporal_scaling(file_name,data_dir,subfolder_pre,N_steps,R,Nt_compare)
