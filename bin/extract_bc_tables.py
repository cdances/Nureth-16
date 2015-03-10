#!/usr/bin/python

import os, sys
import numpy as np

def get_table(linenum,lines):
	print "Table Detected at line:", linenum
	linenum+=1
	absc=[]
	ordint=[]
	while("*" not in lines[linenum]):
		data=lines[linenum].split()
		linenum+=1
		N=len(data)
		if(N%2==0):
			for i in range(N/2):
				absc.append(float(data[i*2]))
				ordint.append(float(data[i*2+1]))
		else:
			sys.exit("Tables need to be even numbers!")
	return (absc,ordint)


fname="sine_wave.inp"
outname="tables.txt"

f=open(fname,"r")
lines=f.readlines()
f.close()

tables=[]
for linenum,line in enumerate(lines):
	if("ABSC" in line and "ORDINT" in line):
		tables.append(get_table(linenum,lines))	


f=open(outname,"w")
for i,table in enumerate(tables):
	f.write("Table {0:d}\n".format(i+1))
	f.write("ABSC,ORDINT\n")
	for absc,ordint in zip(table[0],table[1]):
		f.write("{0:.12f},{1:.12f}\n".format(absc , ordint))
f.close()
