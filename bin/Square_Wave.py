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

class Square_Wave(object):
    def __init__(self,Y1,Y2,CFL):
        self.Y1=Y1
        self.Y2=Y2
        self.CFL=CFL
    
    def get_point(self,loc,t_step,var_name=None):
        tol = 1.0e-5
        d_point = (t_step)*self.CFL
        p_point = loc*1.0
#         print d_point, p_point
        if( t_step == 0 ): return self.Y2
        if( abs(d_point - p_point) < tol  ):
            return (self.Y1 +self.Y2)/2.0 # At Wave
        elif( d_point < p_point ): # Before Wave
            return self.Y2
        elif( d_point > p_point ): # After Wave
            return self.Y1
        
    def get_array_at_time(self,t_step,N):
        t_array=[]
        for loc in range(N):
            
            t_array.append(self.get_point(loc, t_step))
        return np.array(t_array)
        

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.v_1=1.0
        self.v_2=2.0
        self.CFL=1.0
        self.N=12
        self.v_wave = Square_Wave( self.v_1, self.v_2, self.CFL)

    def test_get_point_space(self):
        self.assertEqual(self.v_2, self.v_wave.get_point(0,0))
        
    def test_get_point_time(self):
        self.assertEqual(self.v_1, self.v_wave.get_point(0,1))
        
    def test_get_array_time(self):
        t_array = np.ones(self.N)*self.v_2
        my_array = self.v_wave.get_array_at_time(0,self.N)
        self.assertEqual(t_array.all(), my_array.all())

        # should raise an exception for an immutable sequence
        # self.assertRaises(TypeError, v_wave.get_point, (1,2,3))
 
# def test_plot():
#  
#     # State Point 1
#     v_1=1.0
#     v_2=2.0
#     CFL=4.0
#     N=12
#          
#     v_wave = Square_Wave( v_1, v_2, CFL)
#      
#     for j in range(int(N/CFL)):
#         v_data=[]
#         for i in range(N):
#             v_data.append(v_wave.get_point(i,j))
#         plt.plot(v_data,label="Time Step %d"%(j+1))
#     plt.legend()
#     plt.show()
#             
if __name__=="__main__":
    unittest.main()

#     test_plot()