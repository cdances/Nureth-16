#!/usr/bin/python

import os
import sys
import subprocess
import time
import filecmp

def get_vtk_files(folder):
    vtk_files=[]
    for file in os.listdir(folder):
        if (".vtk" in file):
            vtk_files.append(file)
    return sorted(vtk_files)

def check_gold_files(run_dir):
    gold_dir=os.path.join(run_dir,"gold")

    if(not os.path.isdir(gold_dir)):
        print "GOLD DIRECTORY DOES NOT EXIST"
        print gold_dir
        sys.exit(-1)
        
    vtk_files=get_vtk_files(gold_dir)
    
    for vtk_file in vtk_files:
        run_file=os.path.join(run_dir,vtk_file)
        gold_file=os.path.join(gold_dir,vtk_file)
        if(not filecmp.cmp(run_file,gold_file)):
            print "\t Difference between test and gold files"
            print "\t "+vtk_file
            return False
    return True
    

CTF_exe="/home/dances/bin/cobratf-hdf5"
rundir=os.getcwd()

check_gold=True
produce_out=False
rebless=True

trans_tests=["null_transient","square_wave", "sine_wave","single_channel","transient","implicit_square_wave"]
ss_tests=["steady_state","ss_gravity","ss_cooling","ss_heating"]
cond_tests=["null_conduction","cooling","implicit_cooling", "heating", "implicit_heating"]#"implicit_conduction","ss_cooling", "ss_heating"
nucl_rod=["nuclear_rod","implicit_nuclear_rod","ss_nuclear_rod"]

test_names=[]
# test_names+=trans_tests
# test_names+=ss_tests
# test_names+=cond_tests
test_names+=nucl_rod

# test_names=[]
# test_names=["null_conduction"]

failed_runs=[]


for i,test in enumerate(test_names):
    print "Running Test: {0:d} - {1:s} ..".format(i+1, test)
    input_file=test+".inp"
    folder= os.path.join(rundir,test)
    
    clean_output=subprocess.check_output(["clean_CTF.py"],cwd=folder)
#     print folder, CTF_exe, input_file
    try:
        start_time=time.time()
        output_run=subprocess.check_output([CTF_exe,input_file],cwd=folder)
        end_time=time.time()
    except subprocess.CalledProcessError as e:
        print "\t test failed to run ... "
        failed_runs.append(i)
    else:
        if(rebless):
            print "REBLESSING FILE"
            output_bless=subprocess.check_output(["rebless.py"],cwd=folder)
        if(check_gold_files(folder)):
            print "\t test passed ... {0:f} seconds".format(end_time-start_time)
        else:
            print "\t test failed ... {0:f} seconds".format(end_time-start_time)
            failed_runs.append(i)
        if(produce_out):
            output_plot=subprocess.check_output(["plot_saved_data.py"],cwd=folder)

print "---------------------------------------------------------"
print "Run Summary"
if(len(failed_runs)==0):
    print "\t passed all tests"
else:
    print "\t failed {0:d} test(s) out of {1:d}".format(len(failed_runs),len(test_names))
    for i in failed_runs:
        print "\t test: {0:d} - {1:s} ..".format(i+1, test_names[i])
    
#     print output_run
    
#     plot_run=subprocess.check_output(["plot_saved_data.py"],cwd=folder)
    