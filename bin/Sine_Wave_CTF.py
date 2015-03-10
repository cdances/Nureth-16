#!/usr/bin/python
'''
Created on Oct 7, 2014

@author: cadances
'''

from matplotlib import pyplot as plt
import numpy as np
from Sine_Wave import Sine_Wave 
from CTF_Write import write_CTF_input

## Problem Parameters
L=3.000 # m
Po=10.000   # bar
T_1=40.000    # C
T_2=30.000    # C
h_1=168.412483829 # kJ/kg
h_2=126.641838971 # kJ/kg
m_1=0.005   # kg/sec
rho_1=992.610425499 # kg/m^3
rho_2=996.049650801 # kg/m^3
A=1.0e-4 # m^2
Vo=m_1/(rho_1*A) # m/sec


## Main Variables
Nx=100
dx = L/Nx # m
# dt=(L/3200.0)/(Vo)*0.5000
# print "dt",dt
CFL=2.0**(-2) 
# CFL=(dt/dx*Vo)
t_final=90.000 # sec
Period= 30.000 # sec

## Calculated Values

dt = CFL*dx/Vo
Nt = int(t_final/dt)+5
m_2=Vo*A*rho_2 # kg/sec


def refresh_V_wave():
    V_wave={}
    V_wave["m_dot"]=Sine_Wave( m_1, m_2, Period, Vo, dx, dt)
    V_wave["T_liq"]=Sine_Wave( T_1, T_2, Period, Vo, dx, dt)
    V_wave["h"]=Sine_Wave( h_1, h_2, Period, Vo, dx, dt)
    V_wave["rho"]=Sine_Wave( rho_1, rho_2, Period, Vo, dx, dt)
    return V_wave

V_wave=refresh_V_wave()

def get_forcing_function( t_array, v_array):
    data = "{0:.8f} {1:.8f} "
    table = ""
    N_array=len(t_array)
    for j in range(N_array):
        table += data.format(t_array[j], v_array[j])
        if( (j+1)%5 == 0 and j!=N_array-1):
            table += '\n'
    return table

if __name__=="__main__":
    
    m_data = V_wave["m_dot"].get_array_at_location(0,Nt)/V_wave["m_dot"].get_point(0,0)
    T_data = V_wave["T_liq"].get_array_at_location(0,Nt)/V_wave["T_liq"].get_point(0,0)
    h_data = V_wave["h"]    .get_array_at_location(0,Nt)/V_wave["h"].get_point(0,0)
    time_data = np.arange(0,Nt)*dt
    
    table_1 = get_forcing_function( time_data, m_data )
#     table_2 = get_forcing_function( time_data, T_data )
    table_2 = get_forcing_function( time_data, h_data )
    
    params={}
    params["m_dot"]=m_1
    params["T1"]=T_1*-1.0
    params["H1"]=h_1
    params["PREF"]=Po
    params["AN"]=A
    params["Nx"]=int(Nx)
    params["DXS"]=dx
    params["NPT"]=int(Nt)
    params["table_1"]=table_1
    params["table_2"]=table_2
    params["N_out"]=int(Nx+2)
    params["t_final"]=t_final
    params["CFL"]=CFL
    
    template="/home/dances/bin/sine_wave_template.inp"
    output_file="sine_wave.inp"
    write_CTF_input(template,output_file,params)

    
    
    


    
    
