#!/usr/bin/python
'''
Created on Jul 15, 2014

@author: cadance

Reference 1. Anderson, D. John Jr., "Modern Compressible Flow with Historical Perspective", 
             McGrawHill Series in Areonautical and Aerospace Engineering, 
             McGraw-Hill Publishing Company, 1990
'''
import sys, os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton
from plot_saved_data import get_results_data,data_convert_SI,clean_directory

# Gas Properties for Air at temperature T = 30 C !!
gamma_o = 1.400   # Cp/Cv
Cp = 1005.0       # J/kg-K
Cv = Cp / gamma_o # J/kg-K
R= Cp-Cv
CFL=0.25000

# Problem Setup
P_ref = 1.0*10.0E+004 # Pa ( 1.0 bar )

def sound_speed(h):
    # Enthalpy in J/kg
#     print np.sqrt( (gamma_o - 1.0) * h )

    return np.sqrt( (gamma_o - 1.0) * h )

def EOS( P_abs, h):
    """
    Computes density as a function of pressure and enthalpy
    """
    rho = (gamma_o)/(gamma_o-1) * P_abs / h #/10.0
    return rho

def equation_1( x, *args ):
    # Equation 7.94. Modified to be used with newton iteration method to find the correct answer.
    y=args[0]
    r_a=args[1]
    return y - x*( 1 - (gamma_o-1)*(r_a)*(x-1)/(2*gamma_o+(gamma_o+1)*(x-1)))**(-2.0*gamma_o/(gamma_o-1.0))

class Shock_Tube(object):
    def __init__(self,N,dx,p1,p4,To):
        self.N=N
        self.dx=dx
        self.X=np.arange(0.0, (N-1)*dx+1.0e-3,dx)
        if(len(self.X) != N):
            print "What???"
            sys.exit()
            
        self.To= To + 273.15 # K
        self.p1 = p1
        self.p4 = p4
        
        self.temporal=[0.0]
        
        # Variables
        self.P=    [ np.zeros_like(self.X) ]
        self.U=    [ np.zeros_like(self.X) ]
        self.H=    [ np.zeros_like(self.X) ]
        self.RHO=  [ np.zeros_like(self.X) ]
        
    def get_min(self,array):
        
        min_value = 1.0e+012
        M = len(array)-1
        for i in range(M):
            t_array=array[i]
            tmp_min = min(t_array)
            min_value= min(tmp_min,min_value)
            
        return min_value
    
    def get_max(self,array):
        
        max_value = -1.0*1.0e+012
        M = len(array)-1
        for i in range(M):
            t_array=array[i]
            tmp_max = max(t_array)
            max_value= max(tmp_max,max_value)
            
        return max_value
        
    def plot_data(self,i,style='k--o',label=None):
    
        
        plt.subplot(2,2,1)
        plt.title("Pressure {0:f} sec".format(self.temporal[i]) )
        plt.plot(self.X,self.P[i],style,label=label)
        plt.axis([self.X[0], self.X[-1], self.get_min(self.P), self.get_max(self.P) ])
        if( label ):
            plt.legend(loc=6)
        
        plt.subplot(2,2,2)
        plt.title("Enthalpy {0:f} sec".format(self.temporal[i]))
        plt.plot(self.X,self.H[i],style,label=label)
        plt.axis([self.X[0], self.X[-1], self.get_min(self.H), self.get_max(self.H) ])
        if( label ):
            plt.legend(loc=6)
        
        plt.subplot(2,2,3)
        plt.title("Density {0:f} sec".format(self.temporal[i]))
        plt.plot(self.X,self.RHO[i],style,label=label)
        plt.axis([self.X[0], self.X[-1], self.get_min(self.RHO), self.get_max(self.RHO) ])
        if( label ):
            plt.legend(loc=6)
        
        plt.subplot(2,2,4)
        plt.title("Velocity {0:f} sec".format(self.temporal[i]))
        plt.plot(self.X,self.U[i],style,label=label)
        plt.axis([self.X[0], self.X[-1], self.get_min(self.U), self.get_max(self.U) ])
        if( label ):
            plt.legend(loc=6)

    def difference(self, other):
        Diff = Shock_Tube(self.N, self.dx, self.p1, self.p4, self.To)
        j=0
        Diff.P  = [  other.P[j] - self.P[j] ]
        Diff.H  = [ other.H[j] - self.H[j] ]
        Diff.U  = [ other.U[j] - self.U[j] ]
        Diff.RHO= [ other.RHO[j] - self.RHO[j] ]
        N = min( len(self.temporal), len(other.temporal))
        for j in range( N - 1):
            j+=1
#             if( j > len(other.temporal)-1 ):
#                 print "Error while taking a difference"
#                 print "The other shock tube has more time steps than this one"
#                 sys.exit()
            Diff.temporal.append( self.temporal[j] )
            Diff.P.append( other.P[j] - self.P[j] )
            Diff.H.append( other.H[j] - self.H[j] )
            Diff.U.append( other.U[j] - self.U[j] )
            Diff.RHO.append( other.RHO[j] - self.RHO[j] )
        print "Done Differencing"
        return Diff
    
    def L2_difference(self, other):
        Diff = Shock_Tube(self.N, self.dx, self.p1, self.p4, self.To)
        j=0
        Diff.P  = [ ( other.P[j]   - self.P[j]  )**2 ]
        Diff.H  = [ ( other.H[j]   - self.H[j]  )**2 ]
        Diff.U  = [ ( other.U[j]   - self.U[j]  )**2 ]
        Diff.RHO= [ ( other.RHO[j] - self.RHO[j])**2 ]
        for j in range( len(self.temporal) -1):
            j+=1
            if( j > len(other.temporal) ):
                print "Error while taking a difference"
                print "The other shock tube has more time steps than this one"
                break
            Diff.temporal.append( self.temporal[j] )
            Diff.P.append(   (other.P[j]   - self.P[j]   )**2 )
            Diff.H.append(   (other.H[j]   - self.H[j]   )**2 )
            Diff.U.append(   (other.U[j]   - self.U[j]   )**2 )
            Diff.RHO.append( (other.RHO[j] - self.RHO[j] )**2 )
        print "Done L2 Differencing"
        return Diff
    
    def L2_analysis(self, array):
                    # 0:pressure, 1:enthalpy, 2:density, 3:velocity
        L2_spatial =  [  ]
        L2_temporal = [  ]
        
        N = len(array[0]) # Number of spatial points
        M = len(array)    # Number of time steps
        
        for j in range(M):
#             print j
            L2_temporal.append( math.sqrt( sum( array[j]  )) )
            
        for i in range(N):
            number= 0.0
            for j in range(M):
                number += array[j][i]
            L2_spatial.append( math.sqrt(number) )
            
        return np.array(L2_spatial), (L2_temporal)
    
    def normalize(self, other):
        for j,_ in enumerate(self.temporal): 
            if( j+1 > len(other.temporal) ):
                print "The other shock tube has more time steps than this one"
                break
            Np = max ( other.P[j] )
            Nh = max ( other.H[j] )
            Nu = max ( max ( other.U[j] ) , 1.00e-6 )
            Nrho = max ( other.RHO[j] )
            self.P[j] = self.P[j] / Np 
            self.H[j] = self.H[j] / Nh
            self.U[j] = self.U[j] / Nu
            self.RHO[j] = self.RHO[j] / Nrho
        print "Done Normalizing"
    
    def average(self):
#         P_avg=0.0
#         U_avg=0.0
#         H_avg=0.0
#         RHO_avg=0.0
#         M=len(self.temporal)-1
#         P_avg = np.linalg.norm(self.P)
#         U_avg = np.linalg.norm(self.U)
#         H_avg = np.linalg.norm(self.H)
#         RHO_avg = np.linalg.norm(self.RHO)
        P_avg = np.sum(np.abs(self.P))/np.size(self.P)
        U_avg = np.sum(np.abs(self.U))/np.size(self.U)
        H_avg = np.sum(np.abs(self.H))/np.size(self.H)
        RHO_avg = np.sum(np.abs(self.RHO))/np.size(self.RHO)
        
#         for i in range(M):
#             P_avg = np.average(np.abs(self.P[i]))*1.0/M
#             U_avg = np.average(np.abs(self.U[i]))*1.0/M
#             H_avg = np.average(np.abs(self.H[i]))*1.0/M
#             RHO_avg = np.average(np.abs(self.RHO[i]))*1.0/M
        print "Done Averaging"
        return (P_avg,U_avg,H_avg,RHO_avg)
class CTF_Shock_Tube(Shock_Tube):
    
    def load_data(self, data):
        self.temporal = data["pressure"][0]
        self.P=    [ ]
        self.U=    [ ]
        self.H=    [ ]
        self.RHO=  [ ]
        for i in range(len(self.temporal)):
            self.P.append(  np.copy(data["pressure"][1][i][:]) + P_ref )
            self.U.append(  np.copy(data["velocity"][1][i][:]) )
            self.H.append(  np.copy(data["enthalpy"][1][i][:]) )
            self.RHO.append(np.copy(data["density"][1][i][:]) )
            
class Exact_Shock_Tube(Shock_Tube):
    """
    """
    
    def initialize(self):
        # Set the constant conditions
        
        ho= self.To * Cp # J/kg
        uo= 0.0 # m/s
        self.px=[self.p1+P_ref,0.0,0.0,self.p4+P_ref]
        
        self.hx=[ho,ho,ho,ho]
        self.ux=[uo,uo,uo,uo]
        self.rhox=[]
        self.ax=[]
        for i in range(4):
            self.rhox.append(EOS(self.px[i], ho))
            self.ax.append(sound_speed(self.hx[i]))
            
        # Spatial and Temporal Parameters
        
        self.temporal=[0.0]
        self.region = []
        self.region.append( np.ones_like(self.X) )
        self.region[0][0:self.N/2]=4
        
        
        # Variables
        self.P=    [ np.zeros_like(self.X) ]
        self.U=    [ np.zeros_like(self.X) ]
        self.H=    [ np.zeros_like(self.X) ]
        self.RHO=  [ np.zeros_like(self.X) ]
        
        self.P[-1][0:self.N/2-1]=self.px[3]
        self.P[-1][self.N/2-1:self.N]=self.px[0]
        
        self.H[-1][0:self.N/2-1]=self.hx[3]
        self.H[-1][self.N/2-1:self.N]=self.hx[0]
        
        self.RHO[-1][0:self.N/2-1]=self.rhox[3]
        self.RHO[-1][self.N/2-1:self.N]=self.rhox[0]
        
        self.U[-1][0:self.N/2-1]=self.ux[3]
        self.U[-1][self.N/2-1:self.N]=self.ux[0]
        
#         self.plot_data(0)

    def step1(self):
        # Newton iteration to find p2/p1. 
        # This defines the strength of the incident shock wave
        
        
        
        self.ax[0]=sound_speed(self.hx[0])
        self.ax[3]=sound_speed(self.hx[3])
        
        P2P1 = newton( equation_1 , 0.5*self.px[3]/self.px[0], args=(self.px[3]/self.px[0], self.ax[3]/self.ax[0]), tol=1.0e-4, maxiter=10 )
        
        self.px[1]=P2P1*self.px[0]
        
    def step2(self):
        # Calculate all other incident shock properties
        P2P1 = self.px[1]/self.px[0]
        
        #Eq 7.10
        c1= (gamma_o+1.0)/(gamma_o-1.0)
        T2T1 = P2P1 * ( (c1 + P2P1)/(1.0+c1*P2P1))
        self.hx[1] = self.hx[0] * T2T1
        self.ax[1]=sound_speed( self.hx[1] )
        
        # Eq 7.11
        rho2rho1 = ( 1.0 + c1 * P2P1)/(c1 + P2P1)
        self.rhox[1] = rho2rho1*self.rhox[0]
        
        # Eq. 7.14
        self.W=self.ax[0]*np.sqrt((gamma_o+1)/(2*gamma_o)*(P2P1-1)+1) # This is one of the region border speeds!!!
        
        # Eq. 7.16
        up = self.W * ( 1.0 -  1.0/rho2rho1 )  # This is one of the region border speeds!!! ???
        self.ux[2]=up
        self.ux[1]=self.ux[2]
        
    def step3(self):
        # This defines the strength of the incident expansion wave (p3/p4)
        
        self.px[2] = self.px[1] # 
        
    def step4(self):
        # All other thermodynamic properties immediately behind the expansion wave can be found from the istentropic equations.
        
        P3P4 = self.px[2]/self.px[3]
        
        rho3rho4 = P3P4 ** (1.0/gamma_o) # Isotropic Density Relation
        T3T4 =  P3P4 ** ((gamma_o-1)/gamma_o) # Isotropic Temperature Relation
        
        self.rhox[2] = rho3rho4 * self.rhox[3]
        self.hx[2]   = T3T4 * self.hx[3]
        self.ax[2]   = sound_speed( self.hx[2] )
        
    def step5(self, x, t):
        # The local properties inside the expansion wave.
        # The non-linear portion of the expansion wave that is dependent on position!!! ???
        
        return 2.0/( gamma_o + 1.0 )*( self.ax[3] + x/t ) # Equation 7.89

    def map_position(self, x, t):
        xot = x/t
        head= - self.ax[3]         # - a4
        tail= self.ux[2] - self.ax[2]   # v3 - a3
        contact = self.ux[1]        # v2
        
#         if (xot .lt. head) then
#           eexact = e4
#        elseif (xot .gt. sspd) then
#           eexact = e1
#        elseif (xot .gt. contact) then
#           eexact = e2
#        elseif (xot .gt. tail) then
#           eexact = e3
#        else
        
        region = None
        if( xot < head): # Region 4
            region = 3
        elif(xot > self.W): # Region 1
            region = 0
        elif( xot > contact ): # Region 2
            region = 1
        elif( xot > tail): # Region 3 , but is nearly identical to region 2
            region=2
        else: # Region 3 (  head < xot < tail ) Non-linear region
            region = -2
            
            
#             print head, tail, contact
#             
#             print x, "Region ", region+1,self.ux[region],x/t, head, tail
#             sys.exit()
        
        return region

    def step(self):
        """
        Steps on page 238 ref. 1
        """
                
        # Follow the steps
        
        # Step 1: Calc. p2
        self.step1()
        
        # Step 2: calc. h2 and rho2
        self.step2()
        
        # Step 3: calc. p3, rho3, h3
        self.step3()
        
        # Step 4: 
        self.step4()

    def shock_tube(self, Tf):
        
        self.initialize()
    
        ## Step 0: Initiaize the solution
        
        j = -1
        self.dt = 0.0
        while( self.temporal[-1] < Tf - self.dt):
            j+=1
            i=0
            self.step()
            
            self.dt = self.dx/max(self.ax) * CFL
            self.temporal.append( self.temporal[-1] + self.dt)
            
#             print "Time {0:d} {1:f}".format(j+1,self.temporal[-1])
            
            self.P.append(np.copy(self.P[-1]))
            self.H.append(np.copy(self.H[-1]))
            self.U.append(np.copy(self.U[-1]))
            self.RHO.append(np.copy(self.RHO[-1]))
            
            self.region.append( np.zeros_like(self.X) )
            
            for i in range(self.N):
                   
#                 self.ux[2] = self.ux[1]
                
                ## For each time step, go through the steps to calculate the new time variabels
                region = self.map_position(self.X[i]-self.N/2*self.dx, self.temporal[-1] )
                
                if ( region >= 0 ):
                    self.U[j+1][i]=self.ux[region]
                else:
                    # Step 5: Calculate new time velocities for the current position
                    region = 3
                    self.U[j+1][i] = self.step5( self.X[i]-self.N/2*self.dx, self.temporal[-1] )
                self.region[-1][i] = region + 1
                self.P[j+1][i]=self.px[region]
                self.H[j+1][i]=self.hx[region]
                self.RHO[j+1][i]=self.rhox[region]
#             print self.temporal[-1]
#             break
#             plt.plot(self.X,self.region[-1],'k--o')
#             plt.show()
        print "Done",j, self.temporal[-1], Tf
            
def test_Exact():
    p1 = 0.0
    p4 = (p1 + P_ref)* 10.0 - P_ref
    To = 30.0 # C
    N=50
    T_final =  0.0005 # seconds
    dx=0.1
    
    Exact=Exact_Shock_Tube(N, dx, p1, p4*1.01, To )
    Exact.shock_tube(T_final)
    
    Other = Exact_Shock_Tube(N, dx, p1, p4, To )
    Other.shock_tube(T_final)
    
    for i,t in enumerate(Exact.temporal):
        print "plot",i,t
        plt.figure(figsize=(20.0, 12.0))
        Exact.plot_data(i,style='b--s',label="exact")
        Other.plot_data(i,style='r--o',label="other")
        plt.savefig(os.path.join("tmp","plot_st_{0:04d}.jpg".format(i)),bbox_inches="tight")
        
    Diff = Exact.difference(Other)
    Diff.normalize(Exact)
    for i,t in enumerate(Exact.temporal):
        print "plot difference",i,t
        plt.figure(figsize=(20.0, 12.0))
        Diff.plot_data(i,style='r--v')
        plt.savefig(os.path.join("tmp","plot_diff_{0:04d}.jpg".format(i)),bbox_inches="tight")
                
def test_Load(file_name,output_folder):
    data=get_results_data(file_name)
    data=data_convert_SI(data)
    
    
    dx  = 0.5 # m
    N   = 52.0
    To  = 30.0
    p1  = min( data["pressure"][1][0] )
    p4  = max( data["pressure"][1][0] )
    N   = len( data["pressure"][1][0] )
    print "DATA SHAPE",p1, p4, N
    
    CTF_ST= CTF_Shock_Tube(N,dx,p1,p4,To)
    CTF_ST.load_data(data)
    
    plt.figure(figsize=(20.0,12.0))
    for i,t in enumerate(CTF_ST.temporal):
        CTF_ST.plot_data(i, style="k--o")
    plt.show()

def main(file_name,output_folder):
    data=get_results_data(file_name)
    data=data_convert_SI(data)
    
    tmp_folder = os.path.join(output_folder,"tmp")
    
    L=25.0 # m # Length of the tube
#     dx  = 2.50 # m
    N   = len( data["pressure"][1][0] )
    dx  = L / (N-2)
    To  = max( data["enthalpy"][1][0] )/Cp - 273.15
    p1  = min( data["pressure"][1][0] )
    p4  = max( data["pressure"][1][0] )
    print "DATA", To, p1, p4, N , dx
    T_final=max( data["pressure"][0] ) #3.9368324124744752E-002
    
    CTF_ST= CTF_Shock_Tube(N,dx,p1,p4,To)
    CTF_ST.load_data(data)
    
    Exact=Exact_Shock_Tube(N, dx, p1, p4, To )
    Exact.shock_tube(T_final)
    
    skip = 1# int( len(CTF_ST.X) / 1 )
        
    count = -1
#     for i,t in enumerate(CTF_ST.temporal):
#         #print i, (i+1), skip
#         if ( (i+1) % skip == 0 or i==0):
#             count+=1
#             print "Comparison",i+1,"time",t,"plot",count
#             plt.figure(figsize=(20.0,12.0))
#             Exact.plot_data(i, style="k--s", label="Exact")
#             CTF_ST.plot_data(i, style="b--o",label="CTF")
#             plt.savefig(os.path.join(tmp_folder,"plot_shocktube_{0:04d}.jpg".format(count)),bbox_inches="tight")
#             plt.close()
    print len(CTF_ST.temporal)

    plt.figure(figsize=(10.0,6.0))
    plt.title("Comparison of Times")
    plt.xlabel("Time Step")
    plt.ylabel("Time")
    plt.plot(Exact.temporal,'k--s',label="Exact")
    plt.plot(CTF_ST.temporal,'b--o',label="CTF")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(output_folder,"plot_st_time.jpg"),bbox_inches="tight")
    plt.close()
    
    Diff = Exact.difference(CTF_ST)
#     Diff.normalize(Exact)
     
#     count = -1
#     M = len(CTF_ST.temporal)-1
#     for i in range(M):
#         if ( (i+1) % skip == 0 or i==0):
#             count+=1
#             print "Error",i,"plot",count
#             plt.figure(figsize=(20.0,12.0))
#             Diff.plot_data(i, style="r--v", label="Error")
#             plt.savefig(os.path.join(tmp_folder,"plot_st_err_{0:04d}.jpg".format(i)),bbox_inches="tight")
             
    (P_avg,U_avg,H_avg,RHO_avg) = Diff.average()
     

#     CTF_ST.normalize(CTF_ST)
#     Exact.normalize(Exact)
    
#     Diff2 = Exact.L2_difference(CTF_ST)
#     Diff2.normalize(Exact)
#     Diff2.normalize(Exact)
#     (L2_space_P,     L2_time_P) = Diff2.L2_analysis(Diff2.P)
#     (L2_space_H,     L2_time_H) = Diff2.L2_analysis(Diff2.H)
#     (L2_space_U,     L2_time_U) = Diff2.L2_analysis(Diff2.U)
#     (L2_space_RHO, L2_time_RHO) = Diff2.L2_analysis(Diff2.RHO)
    
#     N = len(Diff2.X)*len(Diff2.temporal)
#     N = 1
#     N = N * len(Diff2.X)
#     N = N * len(Diff2.temporal)
#     
#     P_avg = sum(L2_time_P)     / N #/ np.amax(Diff2.P)
#     U_avg = sum(L2_time_U)     / N #/ np.amax(Diff2.U)
#     H_avg = sum(L2_time_H)     / N #/ np.amax(Diff2.H)
#     RHO_avg = sum(L2_time_RHO) / N #/ np.amax(Diff2.RHO)
#     
    print " N , {0:d}".format(N)
    print " dx , {0:f}".format(dx)
    print " {0:}, {1:}, {2:}, {3:}".format("pressure","velocity","enthalpy","density ")
#     print " {0:f} % , {1:f} %, {2:f} %, {3:f} %".format(P_avg*100.0, U_avg*100.0, H_avg*100.0, RHO_avg*100.0)
    print " {0:f}  , {1:f} , {2:f} , {3:f} ".format(P_avg, U_avg, H_avg, RHO_avg)
    (P_avg,U_avg,H_avg,RHO_avg) = CTF_ST.average()
    print " {0:f}  , {1:f} , {2:f} , {3:f} ".format(P_avg, U_avg, H_avg, RHO_avg)
#     
#     ttl_x = "L2 Spatial Error {0:}"
#     ttl_t = "L2 Temporal Error {0:}"
#     
#     plt.figure(figsize=(20.0,12.0))
#     
#     plt.subplot(2,2,1)
#     plt.title( ttl_x.format("Pressure [Pa]"))
#     plt.xlabel("Position [m]")
#     plt.plot( Diff2.X, L2_space_P )
#     
#     plt.subplot(2,2,2)
#     plt.title( ttl_x.format("Enthalpy [J/kg]"))
#     plt.xlabel("Position [m]")
#     plt.plot( Diff2.X, L2_space_H )
#     
#     plt.subplot(2,2,3)
#     plt.title( ttl_x.format("Density [kg/m^3]"))
#     plt.xlabel("Position [m]")
#     plt.plot( Diff2.X, L2_space_RHO )
#     
#     plt.subplot(2,2,4)
#     plt.title( ttl_x.format("Velocity [m/s]"))
#     plt.xlabel("Position [m]")
#     plt.plot( Diff2.X, L2_space_U )
#     plt.savefig(os.path.join(output_folder,"plot_st_L2_space.jpg"),bbox_inches="tight")
#     
#     plt.figure(figsize=(20.0,12.0))
#     
#     plt.subplot(2,2,1)
#     plt.title( ttl_t.format("Pressure [Pa]"))
#     plt.xlabel("Time [sec]")
#     plt.plot( Diff2.temporal, L2_time_P )
#     
#     plt.subplot(2,2,2)
#     plt.title( ttl_t.format("Enthalpy [J/kg]"))
#     plt.xlabel("Time [sec]")
#     plt.plot( Diff2.temporal, L2_time_H )
#     
#     plt.subplot(2,2,3)
#     plt.title( ttl_t.format("Density [kg/m^3]"))
#     plt.xlabel("Time [sec]")
#     plt.plot( Diff2.temporal, L2_time_RHO )
#     
#     plt.subplot(2,2,4)
#     plt.title( ttl_t.format("Velocity [m/s]"))
#     plt.xlabel("Time [sec]")
#     plt.plot( Diff2.temporal, L2_time_U )
#     plt.savefig(os.path.join(output_folder,"plot_st_L2_time.jpg"),bbox_inches="tight")
    
if __name__=="__main__":

    file_name = "results.txt"
    output_dir = "results"
    
    clean_directory(file_name,output_dir)
    
    main("results.txt","results")
    
    
    
