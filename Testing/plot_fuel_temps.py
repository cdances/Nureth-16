#!/usr/bin/python
'''
Created on Aug 4, 2014

@author: cadance
'''

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from plot_saved_data import clean_directory

dx = 0.1* 3.28084

class Rod(object):
    def __init__(self):
        self.initialize()

    def initialize(self):
        self.number = 0
        self.X=0.0
        self.time=0.0
        self.R=[]
        self.T=[]
        
    def center2side(self):
        self.r_center=[]
        for i in range(len(self.R)-1):
            self.r_center.append((self.R[i]+self.R[i+1])/2.0)
        
    def map_data(self,x,y):
        n=len(y)
        m=len(x)
        Z=np.zeros((n,m))
        for i in range(n-1):
            for j in range(m-1):
                Z[i+0][j+0]=self.T[j][i]
        return Z
        
    def plot_snapshot(self,Nlevel=1,linestyle='k-x'):
        plt.title("rod {0:d} level {1:f} ft".format(self.number,self.X[Nlevel]))
        plt.grid()
        plt.xlabel("Radial Position [ft]")
        plt.ylabel("Temperature [F]")
#         for i in range(Nlevels):
        
        plt.plot(self.r_center,self.T[Nlevel],linestyle)
        plt.legend(loc="upper left")
        
    def rod_plot(self,plot_num,destination):
#         print self.X, self.R
        x = np.arange(0, len(self.X)+1)*dx
        r = self.R
        (Y,X) = np.meshgrid(x,r)
        Z=self.map_data(x,r)
        print "rod",self.number,"step",plot_num
        
        plt.figure(figsize=(8.0, 8.0))
        plt.title("rod {0:d} time {1:f}".format(self.number,self.time))
        plt.pcolormesh(X,Y,Z,edgecolors='face')
        plt.xlabel("Radial Position [ft]")
        plt.ylabel("Axial Position [ft]")
        plt.colorbar()
        destination = os.path.join(destination,'tmp')
        plt.savefig(os.path.join(destination,"rod_{0:d}_{1:04d}.jpg".format(self.number,plot_num)))
#         plt.grid()
#         plt.show()
               
def float_line(line):
    data=line.split(',')[1].split()
    
    f_data = []
    for item in data:
        f_data.append(float(item))
    return np.array(f_data)

def get_temps(line):
    data=line.split(',')
    data.pop(0)
    
    f_data = []
    for item in data:
        f_data.append(float(item))
    return np.array(f_data)

def plot_cell_transient(Rods,nlevel,rnode,linestyle):
    data=[]
    t=[]
    for r_num in Rods:
        for rod in Rods[r_num]:
            t.append(rod.time)
            data.append(rod.T[nlevel][rnode])
    plt.plot(t,data,linestyle,label="level {0:d} radial node {1:d}".format(nlevel,rnode))

def plot_rod_profile(Rods,r_num,nlevel,destination=''):
    plt.figure(figsize=(8.0,8.0))
    for i,rod in enumerate(Rods[r_num]):
        if(i==0):
            rod.plot_snapshot(nlevel,'g--o')
        elif( i+1 == len(Rods[r_num])):
            rod.plot_snapshot(nlevel,'r--s')
        else:
            rod.plot_snapshot(nlevel)
    plt.savefig(os.path.join(destination,"profile_rod{0:d}_level_{1:d}.jpg".format(r_num,nlevel)))
        
def plot_rod_2D(Rods,destination=''):
    # 2D Plot 
    for r_num in Rods:
        count=0
        for rod in Rods[r_num]:
            count+=1
            rod.rod_plot(count,destination)
            
def plot_rod_transient(Rods,r_num,nlevel,destination=''):
    # Transient Plot at a location
    plt.figure(figsize=(8.0, 6.0))
    plt.title("Transient of rod {0:d} axial level {1:d}".format(r_num,nlevel))
    plt.xlabel("Time [sec]")
    plt.ylabel("Temperature [F]")
    for i in range(len(Rods[r_num][0].r_center)):
        plot_cell_transient(Rods,nlevel,i,'-')
    plt.grid()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(destination,"trans_rod{0:d}_axial{1:d}.jpg".format(r_num,nlevel)))
             
def get_rods(file_name):
    f = open(file_name,'r')
    lines = f.readlines()
    f.close()
    
    Rods = {}
    
    for line in lines:
        if( "rod_temps" in line):
            time_t = float(line.split()[1])
            continue
        if( "Rod " in line):
            r_num = int(line.split(",")[1])
            if(r_num not in Rods.keys()):
                Rods[r_num]=[]
                
            Rods[r_num].append( Rod() )
            Rods[r_num][-1].time = time_t
            Rods[r_num][-1].number = r_num
            continue
        if("radial" in line):
            Rods[r_num][-1].R=float_line(line)
            Rods[r_num][-1].center2side()
            continue
        if("axial" in line):
            Rods[r_num][-1].X=float_line(line)
            continue
        if("Temperature" in line):
            if(len(Rods[r_num][-1].R)==0): 
                print "Warning: No radial array"
                sys.exit()
            if(len(Rods[r_num][-1].X)==0): 
                print "Warning: No axial array"
                sys.exit()
            Rods[r_num][-1].T.append(float_line(line))
    return Rods

def main(file_name,output_dir):
    
    Rods = get_rods(file_name) 
    
    # Plot of Transeint behaviour
    for r_num in Rods:
        for i in range(len(Rods[r_num][0].X)):
            plot_rod_transient(Rods,r_num,i,output_dir)
            
    plot_rod_2D(Rods,output_dir)

    # Rod Profile
    for r_num in Rods:
        for i in range(len(Rods[r_num][0].X)):
                plot_rod_profile(Rods,r_num,i,output_dir)
                
if __name__=="__main__":
    file_name = "rod_temps.txt"
    output_dir="results"
    clean_directory(file_name,output_dir)
    
    main(file_name,output_dir)