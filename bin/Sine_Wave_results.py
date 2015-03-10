#!/usr/bin/python
'''
Created on Oct 7, 2014

@author: cadances
'''


import math
import os
# import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import numpy as np


from Sine_Wave import Sine_Wave
from Sine_Wave_CTF import  V_wave,dt
from CTF_parser import *
from plot_saved_data import clean_directory

def main(file_name,plot_trans=True):
    Data = CTF_H5_interface(file_name)
    
    # State Point 1
    m_dot_1=Data.get_point(0, Data.Nt-1, "m_dot")
    rho_1=Data.get_point(0, Data.Nt-1, "rho")
    h_1=Data.get_point(0, Data.Nt-1, "h" )
    p_1=Data.get_point(0, 0, "P" )
    
    # State Point 2
    m_dot_2=Data.get_point(0, 0, "m_dot")
    rho_2=Data.get_point(0, 0, "rho")
    h_2=Data.get_point(0, 0, "h" )
    p_2=Data.get_point(0, 0, "P" )
    
    Vo = calc_velocity( m_dot_1, rho_1, Data.A)
    CFL = Vo*Data.dt/Data.dz
#     P = Data.Nz*Data.dt/2.0
    
    print "Number of Axial Levels", Data.Nz
    print "Number of Time Steps", Data.Nt
    print "Courant Number is ", CFL 
    print "Time Step Size", Data.dt,"seconds"
    print "Axial Size", Data.dz, "meters"
#     print "Period ", P, "seconds"
        
    Time_data= Data.get_time_steps()
    V_inlet={}
    V_inlet_exact={}
    for variable,wave in V_wave.iteritems():
        V_inlet[variable] = Data.get_array_at_location(0,variable)
        V_inlet_exact[variable] = wave.get_array_at_location(1,len(Time_data))
    
    ## Static Information
    output_dir = "results"
    
    dt_CTF=[]
    for i in range(np.size(Time_data)-4):
        dt_CTF.append( Time_data[i+1]-Time_data[i] )
    dt_CTF=np.array(dt_CTF)
    print "Average CTF time step size", np.average(dt_CTF)
    print "Average expected time step size", dt
    
    exp_time=np.ones(np.size(dt_CTF))*dt
    plt.figure()
    plt.title("Time Comparison")
    plt.plot(Time_data[0:np.size(dt_CTF)],dt_CTF,label="CTF")
    plt.plot(Time_data[0:np.size(dt_CTF)],exp_time,label="Exact")
    plt.ylabel("CTF Time [sec]")
    plt.xlabel("Time Step Size [sec]")
    plt.legend()
    plt.savefig(os.path.join(output_dir,"Time_Step.png"))
    plt.close()
    
    for variable in V_wave.keys():
        plt.figure()
        plt.title(variable + " NEAR the inlet")
        plt.plot(Time_data,V_inlet[variable],label="CTF")
        plt.plot(Time_data,V_inlet_exact[variable],label="Exact")
        plt.xlabel("Time [sec]")
        plt.ylabel(field[variable])
        plt.legend()
        plt.savefig(os.path.join(output_dir,"Inlet_{0:s}.png".format(variable)))
        plt.close()
    
    error={}
    for variable in V_wave.keys():
        error[variable]=get_integral_sum(variable, Data, V_wave[variable],end_skip=5)
    
    ## Transient Information
    Height = np.arange(0,Data.Nz)*Data.dz+Data.dz/2.0
    Nframes=20.0
    skip_frame=int(np.size(Time_data)/Nframes)
    output_dir = os.path.join(output_dir,"tmp")
    
    if( plot_trans ):
        print "Creating Transient Plots", Nframes
        for variable in V_wave.keys():
            V_max=np.max(V_wave[variable].get_array_at_time(Data.Nt,Data.Nz))
            V_min=np.min(V_wave[variable].get_array_at_time(Data.Nt,Data.Nz))
            
            print "Plotting ", field[variable]
            for j,t_value in enumerate(Time_data):
                if((j)%skip_frame!=0 and j!=0):
                    continue
                
                # Calcualte Data
                V_j = Data.get_array_at_time(j,variable)
                V_data = V_wave[variable].get_array_at_time(j,Data.Nz)
    
                # Create the plot
                plt.figure(figsize=(16,8))
                plt.subplot(1,2,1)
                plt.title("Variable " + variable+" at time {0:.6f}".format(t_value))
                plt.ylabel(field[variable])
                plt.xlabel(field["dz"])
                plt.fill_between(Height*100.0, V_j,V_data,facecolor='red',alpha=0.5)
                plt.plot(Height*100.0, V_j, label="CTF")
                plt.plot(Height*100.0, V_data, label="Exact")
                plt.legend()
                plt.ylim(V_min,V_max)
                plt.subplot(1,2,2)
                plt.plot(Height*100.0, (V_j-V_data),'r',label="error")
                plt.title("Error at time {0:.6f}".format(t_value))
                plt.ylabel(field[variable])
                plt.xlabel(field["dz"])
                plt.ylim((V_min-V_max),(V_max-V_min))
                plt.savefig(os.path.join(output_dir,variable+"_{0:04d}.png".format(j)))
                plt.close()
                if(j+1==np.size(Time_data)-3):
                    break
    print 'l1 normalized values'
    for var,err in error.iteritems():
        print var, err
    
    ## Close the File
    Data.close_file()

    
if __name__=="__main__":
    clean_directory("results.txt","results")
    
    file_name="sine_wave.ctf.h5"
    file_name= os.path.join(os.getcwd(),file_name)
    print file_name
    main(file_name,plot_trans=False)
