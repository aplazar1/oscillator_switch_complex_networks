#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 12:48:29 2024

@author: aplazar1
"""
import matplotlib 
import numpy as np
import math as mt
import matplotlib.pyplot as plt 
import os as os
import pandas as pd
import csv

joint_csv=open("ER_csv_multiple_reps.tsv", "a")
joint_csv.write('p')
joint_csv.write('\t')
joint_csv.write('C')
joint_csv.write('\t')
joint_csv.write('<l>')
joint_csv.write('\t')
joint_csv.write('G kmax node')
joint_csv.write('\t')
joint_csv.write('G kmax node type')
joint_csv.write('\t')
joint_csv.write('G kmax')
joint_csv.write('\t')
joint_csv.write('nodes_with_max_degree_centrality')
joint_csv.write('\t')
joint_csv.write('max_degree_centrality')
joint_csv.write('\t')
joint_csv.write('nodes with max betweenness centrality')
joint_csv.write('\t')
joint_csv.write(' max betweenness centrality')
joint_csv.write('\t')
joint_csv.write('louvain modularity')
joint_csv.write('\t')
joint_csv.write('G <k>')
joint_csv.write('\t')
joint_csv.write('K_t_init')
joint_csv.write('\t')
joint_csv.write('omega_0')
joint_csv.write('\t')
joint_csv.write('delta_scale')
joint_csv.write('\t')
joint_csv.write('K_max')
joint_csv.write('\t')
joint_csv.write('K_theta')
joint_csv.write('\t')
joint_csv.write('Kxx')
joint_csv.write('\t')
joint_csv.write('eta')
joint_csv.write('\t')
joint_csv.write('tau')
joint_csv.write('\t')
joint_csv.write('beta_l')
joint_csv.write('\t')
joint_csv.write('<r_x_eq>')
joint_csv.write('\t')
joint_csv.write('<r_theta_eq>')
joint_csv.write('\t')
joint_csv.write('num_switches')
joint_csv.write('\t')
joint_csv.write( 'num_oscillators')
joint_csv.write('\t')
joint_csv.write('mean_x_init')
joint_csv.write('\t')
joint_csv.write('SD_x_init')
joint_csv.write('\n')


rep_directories=[]
for filename in os.listdir("/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/ER_oscillator_switch/"):
    rep_directories.append(str(filename))
    #dir=str(filename)
    #os.chdir("/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/ER_oscillator_switch/" + dir)
  # with open(os.path.join(os.getcwd(), filename), 'r') as f: # open in readonly mode
      # do your stuff#
full_path=[]
for i in rep_directories:
    path="/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/ER_oscillator_switch/" + i
    full_path.append(path)
    
for i in range(len(full_path)):
    #print(i)
    #print(full_path[i])
    if full_path[i]+ "/ER_p_fixed_increasing_k_rtheta_10_28_24.tsv" != "/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/ER_oscillator_switch/.DS_Store/ER_p_fixed_increasing_k_rtheta_10_28_24.tsv":
        file=open(full_path[i]+ "/ER_p_fixed_increasing_k_rtheta_10_28_24.tsv", 'r')
        #rows=file.readlines()
        #print(rows)
        reader = csv.reader(file, delimiter="\t")
        row_num=0
        for row in reader:
            #print(row)
            row_num=row_num+1
            h_num=0
            if row_num==3:
                for h in row:
                    #print(h)
                    #print(h_num)
                    joint_csv.write(h)
                    h_num=h_num+1
                    #print(h)
                    # #item=i.split("\t")
                    # #print(item)
                    if h_num==27:
                        joint_csv.write('\n')
                    else:
                        joint_csv.write('\t')
                    
                   

   
            
   
joint_csv.close()



##SW section

joint_csv2=open("SW_csv_multiple_reps.tsv", "a")
joint_csv2.write('p')
joint_csv2.write('\t')
joint_csv2.write('C')
joint_csv2.write('\t')
joint_csv2.write('<l>')
joint_csv2.write('\t')
joint_csv2.write('G kmax node')
joint_csv2.write('\t')
joint_csv2.write('G kmax node type')
joint_csv2.write('\t')
joint_csv2.write('G kmax')
joint_csv2.write('\t')
joint_csv2.write('nodes_with_max_degree_centrality')
joint_csv2.write('\t')
joint_csv2.write('max_degree_centrality')
joint_csv2.write('\t')
joint_csv2.write('nodes with max betweenness centrality')
joint_csv2.write('\t')
joint_csv2.write(' max betweenness centrality')
joint_csv2.write('\t')
joint_csv2.write('louvain modularity')
joint_csv2.write('\t')
joint_csv2.write('G <k>')
joint_csv2.write('\t')
joint_csv2.write('K_t_init')
joint_csv2.write('\t')
joint_csv2.write('omega_0')
joint_csv2.write('\t')
joint_csv2.write('delta_scale')
joint_csv2.write('\t')
joint_csv2.write('K_max')
joint_csv2.write('\t')
joint_csv2.write('K_theta')
joint_csv2.write('\t')
joint_csv2.write('Kxx')
joint_csv2.write('\t')
joint_csv2.write('eta')
joint_csv2.write('\t')
joint_csv2.write('tau')
joint_csv2.write('\t')
joint_csv2.write('beta_l')
joint_csv2.write('\t')
joint_csv2.write('<r_x_eq>')
joint_csv2.write('\t')
joint_csv2.write('<r_theta_eq>')
joint_csv2.write('\t')
joint_csv2.write('num_switches')
joint_csv2.write('\t')
joint_csv2.write( 'num_oscillators')
joint_csv2.write('\t')
joint_csv2.write('mean_x_init')
joint_csv2.write('\t')
joint_csv2.write('SD_x_init')
joint_csv2.write('\n')


rep_directories=[]
for filename in os.listdir("/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/SW_oscillator_switch"):
    rep_directories.append(str(filename))

full_path=[]
for i in rep_directories:
    path="/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/SW_oscillator_switch/" + i
    full_path.append(path)
    
for i in range(len(full_path)):
    #print(i)
    #print(full_path[i])
    if full_path[i]+ "/SW_p_fixed_increasing_k_rtheta_10_28_24.tsv" != "/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/11_18_24/SW_oscillator_switch/.DS_Store/SW_p_fixed_increasing_k_rtheta_10_28_24.tsv":
        file=open(full_path[i]+ "/SW_p_fixed_increasing_k_rtheta_10_28_24.tsv", 'r')
        #rows=file.readlines()
        #print(rows)
        reader = csv.reader(file, delimiter="\t")
        row_num=0
        for row in reader:
            #print(row)
            row_num=row_num+1
            h_num=0
            if row_num==3:
                for h in row:
                    #print(h)
                    #print(h_num)
                    joint_csv2.write(h)
                    h_num=h_num+1
                    #print(h)
                    # #item=i.split("\t")
                    # #print(item)
                    if h_num==27:
                        joint_csv2.write('\n')
                    else:
                        joint_csv2.write('\t')
                    
                   

   
            
   
joint_csv2.close()