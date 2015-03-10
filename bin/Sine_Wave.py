#!/usr/bin/python
'''
Created on Oct 7, 2014

@author: cadance
'''


import math
import os
import matplotlib.pyplot as plt
import numpy as np

import unittest

class Sine_Wave(object):
    def __init__(self,Y1,Y2,period,Vo, dx, dt):
        ## Y(x,t) = Y1 - (Y1- Y2)cos(period/(2*pi)*(t-x/Vo))
        self.Y1=Y1
        self.Y2=Y2
        self.period=period
        self.w=(2.0*math.pi)/period
        self.Vo=Vo
        self.dt=dt
        self.dx=dx
    
    def get_point(self,loc,t_step):
        tol = 1.0e-5
        x=loc*self.dx
        t=t_step*self.dt
        if( x >= t*self.Vo or t_step == 0 ):
            return self.Y1
        else:
#             return self.Y2 - (self.Y2-self.Y1)*(math.cos(self.w*(t-x/self.Vo)))
            return (self.Y1+self.Y2)*0.5 + 0.5*(self.Y1-self.Y2)*(math.cos(self.w*(t-x/self.Vo)))
        
    def get_array_at_time(self,t_step,N):
        t_array=[]
        for loc in range(N):
            t_array.append(self.get_point(loc, t_step))
        return np.array(t_array)
    
    def get_array_at_location(self,loc,N):
        x_array=[]
        for t_step in range(N):
            x_array.append(self.get_point(loc,t_step))
        return np.array(x_array)
        

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.Y_1=2.0
        self.Y_2=1.0
        self.Vo=1.0  # m/sec
        self.dt=1.00 # sec
        self.dx=1.0  # m
        self.N=10
        self.P=self.N*(self.dx)/(self.Vo)*1.0 # sec
        self.Y_wave = Sine_Wave( self.Y_1, self.Y_2, self.P, self.Vo, self.dx, self.dt)

    def test_get_initial_value(self):
        self.assertEqual(self.Y_1, self.Y_wave.get_point(0,0))
        
    def test_get_initial_profile(self):
        data1=np.ones(self.N)*self.Y_1
        data2=self.Y_wave.get_array_at_time(0,self.N)
        self.assertRaises( np.testing.assert_allclose(data1, data2) )
        
    def test_get_end_profile(self):
        X=np.arange(0.0,self.N)*self.dx
        X_var= self.Y_wave.w*(self.N-X/self.Vo)
        data1=0.5*((self.Y_2+self.Y_1) -(self.Y_2-self.Y_1)*np.cos(X_var))
        data2=self.Y_wave.get_array_at_time(self.N,self.N)
        self.assertRaises( np.testing.assert_allclose(data1, data2) )
        
    def test_get_inlet_condition(self):
        T=np.arange(0.0,self.N)*self.dt
        data1=0.5*((self.Y_2+self.Y_1) -(self.Y_2-self.Y_1)*np.cos(self.Y_wave.w*T))
        data2=self.Y_wave.get_array_at_location(0, self.N)
        self.assertRaises( np.testing.assert_allclose(data1, data2) )
 
    def test_min_value(self):

        data=self.Y_wave.get_array_at_time(self.N,self.N)

        self.assertEqual( self.Y_2, np.min(data) )
        
    def test_max_value(self):

        data=self.Y_wave.get_array_at_time(self.N,self.N)

        self.assertEqual( self.Y_1, np.max(data) )
        
def test_plot():
  
    Y_1=2.0
    Y_2=1.0
    Vo=1.0 # m/sec
    dt=1.00 # sec
    dx=1.0 # m
    N=100
    P=1.00*N*(dx)/(Vo) # sec
          
    Y_wave = Sine_Wave( Y_1, Y_2, P, Vo, dx, dt)
    
#     frames=10
#     for j in range(frames+1):
#         Y_data=Y_wave.get_array_at_time(j*int(N/frames)*int(dx/dt),N)
#         plt.plot(Y_data,label="Time Step %d"%(j*int(N/frames)+1))
#     plt.show()
    
    plt.plot(Y_wave.get_array_at_location(0,N))
    plt.show()
#             
if __name__=="__main__":
#     unittest.main()

    test_plot()