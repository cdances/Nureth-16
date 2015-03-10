#!/bin/bash

export GOLD=~/GitHub/COBRA-TF/cobra_tf/test_matrix/coverage_cases/gold
export GOLD=~/GitHub/MY_COBRA-TF/cobra_tf/test_matrix/coverage_cases/gold

export TM=/home/cad39/Programs/CTF/cobra_tf/test_matrix
export TM=/home/cad39/Programs/my_CTF/cobra_tf/test_matrix

cp $TM/COBRA_TF_run_steam01/vis_results.vtk $GOLD/steam01.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam02/vis_results.vtk $GOLD/steam02.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam03/vis_results.vtk $GOLD/steam03.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam04/vis_results.vtk $GOLD/steam04.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam05/vis_results.vtk $GOLD/steam05.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam06/vis_results.vtk $GOLD/steam06.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam07/vis_results.vtk $GOLD/steam07.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam08/vis_results.vtk $GOLD/steam08.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam09/vis_results.vtk $GOLD/steam09.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam10/vis_results.vtk $GOLD/steam10.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam11/vis_results.vtk $GOLD/steam11.vis_results.vtk.gold
cp $TM/COBRA_TF_run_steam12/vis_results.vtk $GOLD/steam12.vis_results.vtk.gold


