#!/usr/bin/python
'''
Created on Oct 7, 2014

@author: cadance
'''


import math
import os
import matplotlib.pyplot as plt


from Square_Wave import Square_Wave
from CTF_parser import *

def get_location(z,dz,Nz):
    tol = 1.0e-6
    for i in range(Nz):
        if( i*dz+dz/2.0 - z >=  - tol   ):
            return i
            
def get_time(t,dt,Nt):
    tol = 1.0e-6
    for j in range(Nt):
#         print j*dt, t
        if( j*dt - t >=  - tol   ):
            if( abs(j*dt - t) < abs((j-1)*dt - t) ):
                return j
            else:
                return j-1

def main(file_name):
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
    
    print "Courant Number is ", CFL 
    print "Time Step Size", Data.dt,"seconds"
    print "Axial Size", Data.dz, "meters"
    
    var_waves={}
    
    var_waves["rho"] = Square_Wave( rho_1, rho_2, CFL)
    var_waves["h"] = Square_Wave( h_1, h_2, CFL)
    var_waves["m_dot"] = Square_Wave( m_dot_1, m_dot_2, CFL)
    var_waves["P"] = Square_Wave( p_1, p_2, CFL)
    
    norm_err={}
    int_err={}
    
    print "Calculating Error", Data.Nz, Data.Nt
    
    # Single Normalized Error Value
    for variable in qoi:
              
#         norm_err[variable] = get_l2( variable, Data, var_waves[variable] )
        int_err[variable] = get_integral_sum( variable, Data, var_waves[variable] )


        
#     i_mid=get_location(1.500,Data.dz,Data.Nz)#(2*Data.Nz+1)/2
#     j_mid=get_time(29.8166019951,Data.dt,Data.Nt)
#     middle_point={}
#     
#     print "Error at middle point i,j" 
#     print " {0:d} , {1:d}".format( i_mid,j_mid )
#     
#     for variable in qoi:    
#         middle_point[variable]=abs(Data.get_point(i_mid,j_mid,variable)-var_waves[variable].get_point(i_mid,j_mid))
#         print "{0:s} , {1:8.9e}".format(variable,middle_point[variable])
    
    # Perfrom the calculations at a certain point in time
    
#     for variable in qoi:
#          
#         norm_err[variable] = get_l2_time( variable, j_mid, Data, var_waves[variable])
#         int_err[variable] = get_integral_sum_time( variable, j_mid, Data, var_waves[variable])
        

    
    ## Close the File
    Data.close_file()
    
    ## Print out information
#     print "Time Step ,",j_mid
#     print "l2 Normalized Error", Data.Nz, Data.Nt
#     for variable in qoi:
#         print "{0:s} , {1:8.9e}".format(variable,norm_err[variable])
    print "Integral Error", Data.Nz, Data.Nt    
    for variable in qoi:
        print "{0:s} , {1:8.9e}".format(variable,int_err[variable])
    
    
    
if __name__=="__main__":
    file_name="advection.ctf.h5"
    file_name= os.path.join(os.getcwd(),file_name)
    print file_name
    main(file_name)