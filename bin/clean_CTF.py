#!/usr/bin/python


import os, sys, shutil

result_there=False
result_files=["res_convergence.txt","results.txt","rod_temps.txt"]
del_ext=[".out",".vtk",".hf5",".h5",".png"]

files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
	if( f in result_files):
		print "removed ", f
		os.remove(f)
	for ext in del_ext:
		if( ext in f ):
			print "removed ", f
			os.remove(f)
			
for fname in result_files:
	f=open(fname,'w')
	f.close()
