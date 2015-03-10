#!/bin/bash

# reset_lcov.sh

# Script to reset the lcov data

# Path to CTF Build Folder
export FOLDER=~/Programs/CTF-hdf5

# Command to reset the code coverage for that build
lcov -z -d ${FOLDER}/cobra_tf/ctf_src/CMakeFiles
