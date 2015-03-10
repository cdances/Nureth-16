#!/bin/bash

if [ -f CMakeCache.txt ]; then
        rm CMakeCache.txt
        rm -rf CMakeFiles
fi

#CTF_DIR_ABS=~/GitHub/COBRA-TF
#CTF_DIR_ABS=~/GitHub/PKA_COBRA_TF
CTF_DIR_ABS=~/GitHub/MY_COBRA-TF

cmake \
-D CMAKE_BUILD_TYPE:STRING=DEBUG \
-D CMAKE_Fortran_FLAGS:STRING="-fbacktrace -fbounds-check -ffpe-trap=zero -O0 -g --coverage " \
-D CTEST_BUILD_FLAGS:STRING="-j16"\
-D CTEST_PARALLEL_LEVEL:STRING="16" \
$CTF_DIR_ABS

#-D CMAKE_Fortran_FLAGS:STRING="-fbacktrace -fbounds-check -ffpe-trap=zero -O0 -g --coverage -Wall " \

