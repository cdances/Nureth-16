#!/usr/bin/python
'''
Created on Jul 1, 2014

@author: cadance
'''

import os,sys,math
import numpy as np
import matplotlib.pyplot as plt

from plot_saved_data import get_results_data, clean_directory, data_convert_SI

def subtract_waves(Wave_1, Wave_2):
    
    N=Wave_1.N
    X=Wave_1.X
    Y1=Wave_1.Y
    Y2=Wave_2.Y
    Y_err=[]
    for i in range(N):
#         print i, len(Y1[i]), len(Y2[i]) 
        Y_err.append(Y1[i]-Y2[i])
    return Wave(X,Y_err,'r--o',label="Error")

def generate_L2Norm_Space(Wave_diff):
        N=len(Wave_diff.Y)
#         Wave_diff=subtract_waves(Wave_1,Wave_2)
        L2Norm=[]
        for i in range(N):#Wave1.Y:
            L2Norm.append( math.sqrt( np.sum( Wave_diff.Y[i]**2 ) ) )
        return np.array(L2Norm)
    
def generate_L2Norm_Time(Wave_diff):
#         Wave_diff=subtract_waves(Wave_1,Wave_2)
        Nx=len(Wave_diff.X)
        Ny=len(Wave_diff.Y)
        
        L2Norm=[]
        for i in range(Nx):#Wave1.Y:
            array=[]
            for j in range(Ny):
                array.append( Wave_diff.Y[j][i])
            array=np.array(array)
            L2Norm.append( math.sqrt( np.sum( array**2 ) ) )
#         L2=Wave(N,X,Y)
        return np.array(L2Norm)
    
    
def generate_L2Norm(Wave_diff):
    Nx=len(Wave_diff.X)-1
    Ny=len(Wave_diff.Y)-1
    L2Norm=0.0
    for i in range(Nx):#Wave1.Y:
        for j in range(Ny):
            L2Norm+=Wave_diff.Y[j+1][i+1]**2
    L2Norm=math.sqrt(L2Norm)
    return L2Norm
    
class Wave(object):
    def __init__(self,X,Y,*args,**kwargs):
        self.X=np.copy(X)
        self.Y=[]
        for data in Y:
            self.Y.append(np.copy(data))
#         print N,len(X),len(Y)
        self.N=len(self.Y)
        self.args=args
        self.kwargs=kwargs
    
    def plot(self, index, *args, **kwargs):
        if(index>self.N):
            print "Index",index,"is out of range",self.N
            sys.exit()
        if(args):
            self.args=args
        if(kwargs):
            self.kwargs=kwargs
#         plt.plot(self.X,self.Y[index], *self.args, **self.kwargs)
        
class Wave_Generator():
    """
    Generate square wave that propogates forward
    """
    def __init__(self,Nx,Ny):
        self.Nx=Nx
        self.Ny=Ny
    
    def setup(self,value1, value2):
        self.X=np.arange(0,self.Nx)
        self.Y=[]
        self.Y.append( np.ones_like(self.X)*value1 )
        self.index=0
        self.Y[0][0]=value2
        
    def step(self):
        self.index+=1
        self.Y.append(np.copy(self.Y[-1]))
        if(self.index<self.Nx):
            self.Y[self.index][self.index]=self.Y[self.index-1][self.index-1]
        self.Y[self.index][-1]=self.Y[self.index][-2]
        
    def build_wave(self,value1,value2,*args,**kwargs):
        self.setup(value1,value2)
        for _ in range(self.Ny-1):
            self.step()
        Gen_Wave=Wave(self.X,self.Y,*args,**kwargs)
        return Gen_Wave

class Wave_Generator2():
    def __init__(self,Nx,dx,dt,Nt,Vo):
        self.Nx=Nx
        self.dx=dx
        self.L=Nx*dx
        self.dt=dt
        self.Nt=Nt
        self.Vo=Vo
        self.CFL=Vo*dt/dx
        
    def setup(self, Y0, Y1 ):
        self.X=self.dx*np.arange(int(self.L/self.dx)) +self.dx/2.0
        self.Y=[]
        self.Y.append( np.ones_like(self.X)*Y0 )
        self.index = 0
        self.count = 0
        self.Y[0][0] = Y1
        self.time = [0.0]
        
    def step(self):
        self.count += 1
        self.time.append( self.dt*self.count)
        self.Y.append(np.copy(self.Y[-1]))
        dt1 = self.dx / self.Vo # exact time step when CFL = 1
        
        if( self.time[-1] >= dt1 * (self.index) ): # If the wave is SUPPOSSED to propogate. # You can't go faster than the CFL, so this is what the code assumes.
            self.index+=1
            if( self.index < self.Nx ):
                self.Y[-1][self.index]=self.Y[-2][self.index-1]
            self.Y[self.index][-1]=self.Y[self.index][-2]
        
    def build_wave(self,Y0,Y1,*args,**kwargs):
        self.setup(Y0,Y1)
        for _ in range(self.Nt - 1):
            self.step()
        Gen_Wave=Wave(self.X,self.Y,*args,**kwargs)
        return Gen_Wave
    
def verify_advection(file_name,dx,output_dir):
    print "Plotting results file", file_name
#     tmp_dir=os.path.join(output_dir,'tmp')
    data = get_results_data(file_name)
    data = data_convert_SI(data)
    
    Vo = np.average(data["velocity"][1])

    for field in data:
        
        plt.figure(figsize=(6.0, 6.0))

        plt.title("Plot of {0:}".format(field))
        plt.ylabel("Cell Number")
        plt.xlabel("Time [sec]")
        
        x=data[field][0] # Time
        y=range(len(data[field][1][0])) # Position
        X,Y=np.meshgrid(x, y)
        Z=np.array(data[field][1])
        
#         plt.pcolormesh(X,Y,Z.T, cmap='RdBu' )
#         plt.axis([X.min(), X.max(), Y.min(), Y.max()])
#         plt.colorbar()
#         plt.savefig(os.path.join(output_dir,field+"_2D.jpg"),bbox_inches='tight')
#         plt.close()
        
        N=int(len(Z[0,:]))/2-1
#         plt.figure(figsize=(6.0, 6.0))
# 
#         plt.title("Plot of {0:} at cell {1:d}".format(field,N+1))
#         plt.ylabel(field)
#         plt.xlabel("Time [sec]")
        value=[]
        for i in range(len(Z[:])):
            value.append(Z[i][N])
#         plt.plot( x , value, 'k-o' )
        
#         plt.grid()
#         plt.savefig(os.path.join(output_dir,field+"_end.jpg"),bbox_inches='tight')
#         plt.close()
        
        if(field in ["enthalpy","flow_rate","density","velocity"]): # ,"energy"
            plot_wave( field,dx,x,y,Z,Vo,output_dir )

def plot_wave_data(N,ref_value,W1,W2,field_name,folder):
    W_diff=subtract_waves(W1,W2)
    
    L2_norm_space=generate_L2Norm_Space(W_diff)*ref_value
    L2_norm_time=generate_L2Norm_Time(W_diff)*ref_value
    L2_norm = generate_L2Norm(W_diff)*ref_value
    
#     print "Error multiplied by ", ref_value
    
    tmp_folder=os.path.join(folder,'tmp')
#     print "Plotting ", field_name
    
    N = len(L2_norm_space) * len(L2_norm_time) 
    print " {0:} , {1:7.6e} ".format( field_name, L2_norm )
    
    # Gif Animation
#     for index in range(len(W1.Y)):
#             
#         plt.figure(figsize=(14.0, 8.0))
#             
#         plt.subplot(1,2,1)
#         plt.grid()
#         W1.plot(index)
#         W2.plot(index)
#         plt.axis([W1.X[0],W1.X[-1],np.amin(W1.Y),np.amax(W1.Y)])
#         plt.title("{0:} at time step {1:d}".format(field_name,index))
#         plt.ylabel("Relative value of {0:}".format(field_name))
#         plt.xlabel("Cell Number")
#             
#         plt.legend()
#             
#         plt.subplot(1,2,2)
#         plt.title("Error of {0:} at time step {1:d}".format(field_name,index))
#         W_diff.plot(index,'r--s')
#         plt.xlabel("Cell Number")
#         plt.ylabel("Error in "+field_name)
#         plt.grid()
#         plt.axis([W1.X[0],W1.X[-1],np.amin(W_diff.Y),np.amax(W_diff.Y)])
#             
#         plt.savefig(os.path.join(tmp_folder,"plot_{0:}_{1:04d}.jpg".format(field_name,index)),bbox_inches="tight")
#          plt.close()
    #
#     plt.figure(figsize=(6.0,6.0))
#     plt.title("Temporal Error for {0:}".format(field_name))
#     plt.plot(L2_norm_space,'k--o')
#     plt.axis([0,len(W_diff.Y),min(L2_norm_space),max(L2_norm_space)])
#     plt.grid()
#     plt.ylabel("Error for {0:}".format(field_name))
#     plt.xlabel("Time Step")
#     plt.savefig(os.path.join(folder,"Temporal_Error_{0:}.jpg".format(field_name)),bbox_inches="tight")
#     plt.close()
#     
#     plt.figure(figsize=(6.0,6.0))
#     plt.title("Spatial Error for {0:}".format(field_name))
#     plt.plot(L2_norm_time,'k--o')
#     plt.axis([0,len(W_diff.X),min(L2_norm_time),max(L2_norm_time)])
#     plt.grid()
#     plt.ylabel("Error for {0:}".format(field_name))
#     plt.xlabel("Cell Number")
#     plt.savefig(os.path.join(folder,"Spatial_Error_{0:}.jpg".format(field_name)),bbox_inches="tight")
#     plt.close()

def plot_wave(field,dx,temporal,position,Z,Vo,folder):
    
#     N=len(Z)
    Nt = len(temporal)
    Nx = len(position)
    dt = temporal[1]
#     dx = position[1]
#     Vo = Z["velocity"][0]
#     max_value=np.amax(Z)
    #min_value=np.amin(Z)
    min_value=1.0#Z[0][0]
    Z=Z/min_value
    first_value=Z[0][0]
#     for i,data in enumerate(Z):
#         Z[i]=data/min_value
        
    initial_value=Z[0][1] #np.amax(Z)
    
    Generator=Wave_Generator2(Nx,dx,dt,Nt,Vo) 
    W2=Generator.build_wave(initial_value,first_value,'b-x',label="Exact Solution")
    W1=Wave(W2.X,Z,'g-o',label="CTF Solution")
#     print initial_value, first_value
#     sys.exit()
    
    plot_wave_data(Nt,min_value,W1,W2,field,folder)
    
#     W_diff=subtract_waves(W1,W2)
#     X,Y=np.meshgrid(temporal, position)
#     Z=np.array(W_diff.Y)*min_value
# #     print "Sizing Enthalpy",min_value
#     
#     plt.figure(figsize=(6.0, 6.0))
# 
#     plt.title("Plot of {0:} error".format(field))
#     plt.ylabel("Cell Number")
#     plt.xlabel("Time [sec]")
#     
#     plt.pcolormesh(X,Y,Z.T, cmap='RdBu' )
#     plt.axis([X.min(), X.max(), Y.min(), Y.max()])
#     plt.colorbar()
#     plt.savefig(os.path.join(folder,field+"_error_2D.jpg"),bbox_inches='tight')    
#     plt.close()

def get_test_data(N):
    dxt = 1.0
    Vo = 1.0
    Generator=Wave_Generator2(N,dxt,dxt,N,Vo) 
    W1=Generator.build_wave(1.0,2.0,'g-o',label="Exact Solution")
    W2=Generator.build_wave(1.0,2.0,'b-x',label="Computer Data")
    
    # randomly change the second wave to give it "noise"
    for index in range(len(W1.Y)):
        W2.Y[index]+=0.1*(0.5-np.random.rand(len(W2.Y[index]))) 
    W3 = subtract_waves( W1, W2 )
    L2_norm=generate_L2Norm_Space(W3)
    return W1,W2,L2_norm

def get_test_data2(Nx,dx,dt,Nt,Vo):
    Generator=Wave_Generator2(Nx,dx,dt,Nt,Vo)
    W1=Generator.build_wave(1.0,2.0,'g-o',label="Exact Solution")
    W2=Generator.build_wave(1.0,2.0,'b-x',label="Computer Data")
    
    # randomly change the second wave to give it "noise"
    for index in range(len(W1.Y)):
        W2.Y[index]+=0.1*(0.5-np.random.rand(len(W2.Y[index]))) 
    W3 = subtract_waves( W1, W2 )
    L2_norm=generate_L2Norm_Space(W3)
    return W1,W2,L2_norm

def test():
    N=40
    file_name="results.txt"
    output_dir="results"
     
    files = clean_directory(file_name, output_dir)
    (W1,W2,L2_norm)=get_test_data(N)
    plot_wave_data(N,1.0,W1,W2,"field",output_dir)
    
def test2():
    Nx=40
    Nt=80
    Vo = 1.0
    dx = 1.0
    dt = 0.5
    
    file_name="results.txt"
    output_dir="results"
     
    files = clean_directory(file_name, output_dir)
    (W1,W2,L2_norm)=get_test_data2(Nx,dx,dt,Nt,Vo)
    plot_wave_data(Nx,1.0,W1,W2,"field",output_dir)

def main():
    file_name="results.txt"
    output_dir="results"
    N=30
    L=3.00
    dx = L/N
    print dx
    clean_directory(file_name, output_dir)
     
    verify_advection(file_name,dx,output_dir)

if __name__=="__main__":
#     test()
    main()
#     file_name="results.txt"
#     output_dir="results"
#     
#     files = clean_directory(file_name, output_dir)
#     
#     main(file_name,output_dir)