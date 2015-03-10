#!/usr/bin/python
'''
Created on Oct 7, 2014

@author: cadances
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np
import math
from Square_Wave import Square_Wave 

field = {}
field["dz"]="channel_cell_height [cm]"
field["A"]="channel_flow_areas [cm^2]"
field["rho"]="channel_liquid_density [kg per m^3]"
field["h"]="channel_liquid_enthalpy [kJ per kg]"
field["m_dot"]="channel_liquid_flow_rate [kg per sec]"
field["T_liq"]="channel_liquid_temps [C]"
field["P"]="channel_pressure [Pa]"
field["time"]="time [sec]"

qoi = ["rho","h","m_dot"]
          
def calc_velocity( m_dot, rho, A):
    return m_dot/(rho*A)

class CTF_H5_interface():
    """
    Assumes a single channel
    """
    def __init__(self,filename):
        self.filename=filename
        self.opened=False
        self.open_file()
        
        step_o = self.get_time_step_name(0)
        
        self.get_time_steps()
        self.Nt = self.get_number_time_steps()
        self.Nz = self.get_mesh_size()
        self.L = self.get_axial_height()
        self.A =  self.get_area()
        self.dt = self.get_time_dt()
        self.dz = self.get_dz()
        
    def open_file(self):
        if( not self.opened):
            self.f = h5py.File(self.filename,'r')   
            self.opened=True
    def close_file(self):
        if( self.opened):
            self.f.close()
            self.opened=False
    
    def get_point(self,loc,t_step,var_name):
        self.open_file()
        t_name = self.get_time_step_name(t_step)
        f_name = field[var_name]
        val=self.f[t_name][f_name][0][0][loc][0]
        self.close_file()
#         print f_name, t_name, loc
        #      [file][time_step][field][channel][axial_index][get_value (first)]
        return val
    
    def get_array_at_time(self,t_step, var_name):
        self.open_file()
        t_name=self.get_time_step_name(t_step)
        f_name = field[var_name]
#         print t_name, f_name
        t_array= self.f[t_name][f_name][0][0][:][:]
        self.close_file()
        return np.reshape(t_array, (1,self.Nz) )[0]
    
    def get_array_at_location(self,loc, var_name):
        data=[]
        for t_step in range(self.Nt):
            t_name=self.get_time_step_name(t_step)
            data.append(self.get_point(loc,t_step,var_name))
        return np.array(data)
        
    def get_time_step_name(self,t_step):
#         print "TRANS_{0:06d}".format(t_step)
        return "TRANS_{0:06d}".format(t_step)
    
    def get_time_steps(self):
        self.open_file()
        time_steps=[]
        for step_name, step_data in self.f.items():
            if("TRANS" not in step_name):
                pass
            t_value = step_data[field["time"]][0]            
            time_steps.append(t_value)
        self.close_file()    
        return time_steps
    
    def get_time_dt(self):
        time_1=0.0
        self.open_file()
        step_name = self.get_time_step_name(2)
        dt=self.f[step_name][field["time"]][0]
        self.close_file()
        return dt
    
    def get_number_time_steps(self):
        time_steps=0
        self.open_file()
        for step_name, step_data in self.f.items():
            if("TRANS" in step_name):
                time_steps+=1
        self.close_file()    
        return time_steps
    
    def get_axial_height(self):
        self.open_file()
        
        axial_height = 0.0
        step_name = self.get_time_step_name(0)
        for dz in self.f[step_name][field["dz"]]:
            axial_height += dz
        
        self.close_file()
        return axial_height/ ( 100.0 ) # meters
    
    def get_mesh_size(self):
        self.open_file()
        step_name = self.get_time_step_name(0)
        Nz=len(self.f[step_name][field["dz"]])
        self.close_file()
        return Nz
    
    def get_dz(self):
        self.open_file()
        step_name = self.get_time_step_name(0)
        dz=self.f[step_name][field["dz"]][0]/ (100.0) # meters
        self.close_file()
        return dz
    
    def get_area(self):
        self.open_file()
        step_name = self.get_time_step_name(0)
        A=self.f[step_name][field["A"]][0][0][0] / ( 100.0**2) # meters^2
        self.close_file()
        return  A
            
def get_l2_space( var_name, loc, data, generated):
    l2_space = 0.0
    for j in range(data.Nt):
        v1=generated.get_point(loc, j)
        v2=data.get_point(loc,j,var_name)
        l2_space+=(v2-v1)**2
    
    return math.sqrt(l2_space)/data.Nt

def get_l2_time( var_name, step, data, generated):
    v_err = get_error_time( var_name, step, data, generated)
    return np.linalg.norm(v_err)#/(data.Nt*data.Nz)


def integrate_space( array ):
    Nx_min=50
    Nx=np.size(array)
    
#     if(Nx_min >= Nx or Nx%Nx_min!=0):
    return array
    
    new_array=[]
    nstep= int(Nx/Nx_min)
    for i in range(Nx_min):
        start_point=i*nstep
        end_point=(i+1)*nstep
        new_array.append(np.sum(array[start_point:end_point])/nstep)
        
    return np.array(new_array)

def get_l2( var_name, data, generated ):
    l2_j_array = []
    l2_value = 0.0
    for j in range(data.Nt):
        v2=data.get_array_at_time(j,var_name)
        v1=generated.get_array_at_time(j,data.Nz)
        v_err = (v2-v1)
        l2_j_array.append(np.linalg.norm(v_err))
    return np.linalg.norm(l2_j_array)#/(data.Nt*data.Nz)

def get_integral_sum_time( var_name, step, data, generated): 
    v_err = get_error_time( var_name, step, data, generated)
    return np.sum(v_err)/np.size(v_err)

def get_error_time( var_name, step, data, generated):
    
    v1=generated.get_array_at_time(step,data.Nz)
    v2=data.get_array_at_time(step, var_name)
    v_err = np.abs(v2 - v1) ## Calculate the error
    v_err = integrate_space(v_err) # Integrate the error
    
    
    plt.subplot(1,2,1)
    plt.title("Values {0:s} at time step {1:d}".format(var_name,step+1))
    plt.plot( integrate_space(v1) ,"-bs",label="Square Wave")
    plt.plot(integrate_space(v2),"-ko",label="CTF Data")
    plt.subplot(1,2,2)
    plt.title("Error {0:s} at time step {1:d}".format(var_name,step+1))
    plt.plot( v_err , '-rx',  label="error")
    plt.show()
    return v_err

def get_integral_sum( var_name, data, generated, end_skip=0): ## Correct I think (might have to square the answer???)
    l2_j_array = []
    l2_value = 0.0
    for j in range(data.Nt-end_skip):
        v2=data.get_array_at_time(j,var_name)
        v1=generated.get_array_at_time(j,data.Nz)
        v_err = np.abs(v2-v1)
        l2_j_array.append(np.sum(v_err))
    return np.sum(l2_j_array)/(data.Nt*data.Nz)

# What we REALLY want, is to integrate the numerical error over space and time, and show that the area of the error is decreasing as we increase the mesh, and that the rate of decrease is LINEAR

if __name__=="__main__":
#     test()
#     main()
    file_name="advection.ctf.h5"
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
    
    print "Number of time steps", Data.Nt
    print "Size of axial mesh", Data.Nz
    print "Length of axial mesh", Data.L , "m"
    print "Channel Area", Data.A*100.0**2 , "cm^2"
    print "CFL", CFL
        
    print field["P"],Data.get_point(0, 0, "P")
    
    var_name="rho"
    W_rho = Square_Wave( Data.get_point(0,Data.Nt-1,var_name), Data.get_point(0,0,var_name),CFL)
    
    Z=np.arange(Data.Nz)*Data.dz+Data.dz/2.0
    
#     frame=10
#     n_frames = int(Data.Nt/frame)
    n_frames = 10
    frame=int(Data.Nt/n_frames)
    
    for j in range(n_frames):         
        plt.plot(Data.get_array_at_time(j*frame,var_name),'k-o')
        plt.plot(W_rho.get_array_at_time(j*frame,Data.Nz),'r-o')
        plt.show()
        
    Data.close_file()

    
