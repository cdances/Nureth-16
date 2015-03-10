#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys, os, shutil

t_kg_lbm = 2.204623    # lbm/kg
t_lbm_kg = 1.0 / 2.204623 # kg/lbm
t_BTU_J  = 1054.35     # J/BTU
t_J_BTU  = 1.0/t_BTU_J # BTU/J
t_ft_m   = 0.3048      # m/ft
t_m_ft   = 1.0/t_ft_m  # ft/m 
t_lbfft2_Pa = 47.88    # ( Pa ) / (lbf/ft^2)
t_Pa_lbfft2 = 1.0/t_lbfft2_Pa # (lbf/ft^2) / ( Pa )
t_enth_Eng_SI = 2324.44 # (J/kg) / (BTU/lbm)
t_enth_SI_Eng = 1.0/t_enth_Eng_SI # (BTU/lbm) / (J/kg)

# print t_BTU_J/t_lbm_kg, t_enth_Eng_SI

def data_convert_SI(data):
    SI_data = {}
    
    for key in data:
        SI_data[key]=[]
        SI_data[key].append(data[key][0][:])
        SI_data[key].append([])
        for i in range(len(data[key][1])):
            SI_data[key][1].append( np.copy( np.array(data[key][1][i][:]) ) )
    add = 0.0
    for key in SI_data:
        if(key=="pressure"):
            factor = t_lbfft2_Pa # Pa / (lbf/sq.-ft)
        if(key=="velocity"):
            factor =  t_ft_m # m / ft
        if(key=="enthalpy" or key=="energy"):
            factor =  t_BTU_J/t_lbm_kg #  (J/kg) / ( BTU / lbm ) = ( J/BTU ) / ( kg/lbm )
#             print factor
#             factor = t_enth_Eng_SI # (J/kg) / ( BTU / lbm )
#             print factor
        if(key=="flow_rate"):
            factor =  t_lbm_kg # (kg/lbm)
        if(key=="density"):
            factor = t_lbm_kg / (t_ft_m**3)
        if(key=="surf_temp"):
            factor = 5.0/9.0
            add = - 32.0*factor
#         print key
        for i in range(len(data[key][1])):
            SI_data[key][1][i] =  factor*np.array(SI_data[key][1][i]) + add
#             if("surf" in key):
#                 print SI_data[key][1][i], factor, add
            
    return SI_data
    
def get_results_data(file_name):
    data={}
    f=open(file_name,'r')
    lines=f.readlines()
    f.close()
    
    field=''
    for line in lines:
        content=line.split()
        if(len(content)==2):
            field=content[0]
            time_t=float(content[1])
            if(field not in data):
                print "Adding field {0:} to data structure".format(content[0])
                data[field]= [ [], [] ] # Array 1 is iteration number, array 2 is the data at that iteration
            data[field][0].append(time_t)
            data[field][1].append( [] )
        if(len(content)==1):
            data[field][1][-1].append( float(content[0]) )
    return data
    
def plot_results_file(file_name,output_dir,Trans=False):
    print "Plotting results file", file_name
    tmp_dir=os.path.join(output_dir,'tmp')
    data=get_results_data(file_name)
    data = data_convert_SI(data)

    for field in data:
        print field
#         if("surf_temp" not in field):
#             continue
        plt.figure(figsize=(6.0, 6.0))

        plt.title("Plot of {0:}".format(field))
        plt.ylabel("Cell Number")
        plt.xlabel("Time [sec]")
        
        x=data[field][0] # Time
        y=range(len(data[field][1][0])) # Position
        X,Y=np.meshgrid(x, y)
        Z=np.array(data[field][1])
        
        plt.pcolormesh(X,Y,Z.T, cmap='RdBu' )
        plt.axis([X.min(), X.max(), Y.min(), Y.max()])
        plt.colorbar()
        plt.savefig(os.path.join(output_dir,field+"_2D.jpg"),bbox_inches='tight')
        plt.close()
        
        N=int(len(Z[0,:]))/2-1
        plt.figure(figsize=(6.0, 6.0))

        plt.title("Plot of {0:} at cell {1:d}".format(field,N+1))
        plt.ylabel(field)
        plt.xlabel("Time [sec]")
        value=[]
        for i in range(len(Z[:])):
            value.append(Z[i][N])
        plt.plot( x , value, 'k-o' )
        
        plt.grid()
        plt.savefig(os.path.join(output_dir,field+"_end.jpg"),bbox_inches='tight')
        plt.close()
        
        time_array=x
        max_value=np.amax(Z)
        min_value=np.amin(Z)
        
        if(Trans):
            for i,t in enumerate(time_array):
                plt.figure(figsize=(8.0, 6.0))
                plt.title("Plot of {0:} at time {1:5.3f}".format(field,t) )
                print field,"step, time", i,t
                field_data=Z[i][:]
                plt.plot(field_data,'k--o')
                plt.axis([0, len(field_data), min_value, max_value])
                plt.grid()
                plt.savefig(os.path.join(tmp_dir,"plot_{0:}_{1:04d}.jpg".format(field,i)),bbox_inches="tight")
                plt.close()
            
def plot_convergence(file_name,output_dir):
    print "Ploting Convergence Parameters"
    if(file_name==""):
        return
    iterations    =[]
    Mass_Balance  =[]
    Mass_Storage  =[]
    Energy_Balance=[]
    Fluid_Energy  =[]
    Solid_Energy  =[]
    
    print file_name
    
    f=open(file_name,'r')
    lines=f.readlines()
    f.close()
    for line in lines:
        data=get_convergence_data(line)
        if(data):
            iterations    .append(int(data[0]))
            Mass_Balance  .append(float(data[1]))
            Mass_Storage  .append(float(data[2]))
            Energy_Balance.append(float(data[3]))
            Fluid_Energy  .append(float(data[4]))
            Solid_Energy  .append(float(data[5]))
            
    Mass_Balance   = np.array(Mass_Balance)
    Mass_Storage   = np.array(Mass_Storage)
    Energy_Balance = np.array(Energy_Balance)
    Fluid_Energy   = np.array(Fluid_Energy)
    Solid_Energy   = np.array(Solid_Energy)
    
    Max_value=[]
    Max_value.append( Mass_Balance   .max() )
    Max_value.append( Mass_Storage   .max() )
    Max_value.append( Energy_Balance .max() )
    Max_value.append( Fluid_Energy   .max() )
    Max_value.append( Solid_Energy   .max() )
    
#     colors="b,r,g,y,k,"
    labels=["Mass Balance  %4.3e",
            "Mass Storage  %4.3e", 
            "Energy Balance %4.3e", 
            "Fluid Energy  %4.3e",
            "Solid Energy  %4.3e"]
    plt.figure(figsize=(6.0, 6.0))
    
    plt.plot(iterations , Mass_Balance  /Max_value[0] , label= labels[0] % (Max_value[0]) )
    plt.plot(iterations , Mass_Storage  /Max_value[1] , label= labels[1] % (Max_value[1]) )
    plt.plot(iterations , Energy_Balance/Max_value[2] , label= labels[2] % (Max_value[2]) )
    plt.plot(iterations , Fluid_Energy  /Max_value[3] , label= labels[3] % (Max_value[3]) )
#     plt.plot(iterations , Solid_Energy  /Max_value[4] , label= labels[4] % (Max_value[4]) )
    
    plt.title("Convergence of Main Parameters")
    plt.xlabel("Time Step or Iteration")
    plt.ylabel("Convergence Parameter Value")
    
    plt.grid()
    plt.legend(loc="upper left")
    
    plt.savefig(os.path.join(output_dir,"convergence.jpg"),bbox_inches="tight")
    plt.close()
                
def get_convergence_data(line):
    if("#" in line):
        return False
    else:
        return line.split()

def clean_directory(file_name,output_dir):
    files = [ f for f in os.listdir('.') if os.path.isfile(f) ]
    directories = [ f for f in os.listdir('.') if os.path.isdir(f) ]
    
    if( output_dir  in directories):
#         os.removedirs(output_dir)
        shutil.rmtree(output_dir) 
    os.mkdir(output_dir)
    os.mkdir(os.path.join(output_dir,'tmp'))
    
    return files
    
if( __name__ == "__main__"):
    
    file_name="results.txt"
    output_dir="results"
    
    files = clean_directory(file_name, output_dir)
    
    plot_results_file(file_name,output_dir,Trans=False)
    
#     for file_name in files:
#         if (".convergence.out" in file_name):
#             plot_convergence(file_name,output_dir)

# plt.show()
