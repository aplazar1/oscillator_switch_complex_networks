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

print('input the base directory for ER graphs: this is the directory that will contain replicate simulations within different folders')
base_dir=input()

joint_csv=open("ER_csv_multiple_reps.tsv", "a")
joint_csv.write('p')
joint_csv.write('\t')
joint_csv.write('C')
joint_csv.write('\t')
joint_csv.write('<l>')
joint_csv.write('\t')
joint_csv.write('max_InfCen')
joint_csv.write('\t')
joint_csv.write('nodes_with_max_InfCen')
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
for filename in os.listdir(base_dir):
    rep_directories.append(str(filename))

full_path=[]
for i in rep_directories:
    path=base_dir + i
    full_path.append(path)
    
for i in range(len(full_path)):
    if full_path[i]+ "/ER_p_fixed_increasing_k_rtheta_10_28_24.tsv" != base_dir + ".DS_Store/ER_p_fixed_increasing_k_rtheta_10_28_24.tsv":
        file=open(full_path[i]+ "/ER_p_fixed_increasing_k_rtheta_10_28_24.tsv", 'r')
        reader = csv.reader(file, delimiter="\t")
        row_num=0
        for row in reader:
            row_num=row_num+1
            h_num=0
            ##select only simulations that ran completely and stored all of the data
            if len(row)==29:
                if row_num==3:
                    for h in row:
                        joint_csv.write(h)
                        h_num=h_num+1
                        if h_num==29: ##note 27 is the number of columns the csv file should have
                            joint_csv.write('\n')
                        else:
                            joint_csv.write('\t')
                    
                   

   
            
   
joint_csv.close()


##SW section
print('input the base directory for SW graphs: this is the directory that will contain replicate simulations within different folders')
base_dir2=input()

joint_csv2=open("SW_csv_multiple_reps.tsv", "a")
joint_csv2.write('p')
joint_csv2.write('\t')
joint_csv2.write('C')
joint_csv2.write('\t')
joint_csv2.write('<l>')
joint_csv2.write('\t')
joint_csv2.write('max_InfCen')
joint_csv2.write('\t')
joint_csv2.write('nodes_with_max_InfCen')
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


rep_directories2=[]
for filename in os.listdir(base_dir2):
    rep_directories2.append(str(filename))

full_path=[]
for i in rep_directories2:
    path=base_dir2 + i
    full_path.append(path)
    
for i in range(len(full_path)):
    if full_path[i]+ "/SW_p_fixed_increasing_k_rtheta_10_28_24.tsv" != base_dir2+ ".DS_Store/SW_p_fixed_increasing_k_rtheta_10_28_24.tsv":
        file=open(full_path[i]+ "/SW_p_fixed_increasing_k_rtheta_10_28_24.tsv", 'r')
        reader = csv.reader(file, delimiter="\t")
        row_num=0
        for row in reader:
            row_num=row_num+1
            h_num=0
            if len(row)==29:
                if row_num==3:
                    for h in row:
                        joint_csv2.write(h)
                        h_num=h_num+1
                        if h_num==29: ##note 27 is the number of columns the csv file should have
                            joint_csv2.write('\n')
                        else:
                            joint_csv2.write('\t')
                        
                       

   
            
   
joint_csv2.close()

##LFR section
print('input the base directory for LFR graphs: this is the directory that will contain replicate simulations within different folders')
base_dir3=input()

joint_csv3=open("LFR_csv_multiple_reps.tsv", "a")
joint_csv3.write('power_law1') 
joint_csv3.write('\t')
joint_csv3.write('power_law2') 
joint_csv3.write('\t')
joint_csv3.write('inter_community_edges')
joint_csv3.write('\t')
joint_csv3.write('C')
joint_csv3.write('\t')
joint_csv3.write('<l>')
joint_csv3.write('\t')
joint_csv3.write('max_InfCen')
joint_csv3.write('\t')
joint_csv3.write('nodes_with_max_InfCen')
joint_csv3.write('\t')
joint_csv3.write('G kmax node')
joint_csv3.write('\t')
joint_csv3.write('G kmax node type')
joint_csv3.write('\t')
joint_csv3.write('G kmax')
joint_csv3.write('\t')
joint_csv3.write('nodes_with_max_degree_centrality')
joint_csv3.write('\t')
joint_csv3.write('max_degree_centrality')
joint_csv3.write('\t')
joint_csv3.write('nodes with max betweenness centrality')
joint_csv3.write('\t')
joint_csv3.write(' max betweenness centrality')
joint_csv3.write('\t')
joint_csv3.write('louvain modularity')
joint_csv3.write('\t')
joint_csv3.write('G <k>')
joint_csv3.write('\t')
joint_csv3.write('K_t_init')
joint_csv3.write('\t')
joint_csv3.write('omega_0')
joint_csv3.write('\t')
joint_csv3.write('delta_scale')
joint_csv3.write('\t')
joint_csv3.write('K_max')
joint_csv3.write('\t')
joint_csv3.write('K_theta')
joint_csv3.write('\t')
joint_csv3.write('Kxx')
joint_csv3.write('\t')
joint_csv3.write('eta')
joint_csv3.write('\t')
joint_csv3.write('tau')
joint_csv3.write('\t')
joint_csv3.write('beta_l')
joint_csv3.write('\t')
joint_csv3.write('<r_x_eq>')
joint_csv3.write('\t')
joint_csv3.write('<r_theta_eq>')
joint_csv3.write('\t')
joint_csv3.write('num_switches')
joint_csv3.write('\t')
joint_csv3.write( 'num_oscillators')
joint_csv3.write('\t')
joint_csv3.write('mean_x_init')
joint_csv3.write('\t')
joint_csv3.write('SD_x_init')
joint_csv3.write('\n')


rep_directories3=[]
for filename in os.listdir(base_dir3):
    rep_directories3.append(str(filename))

full_path=[]
for i in rep_directories3:
    path=base_dir3 + i
    full_path.append(path)
    
for i in range(len(full_path)):
    #print(i)
    #print(full_path[i])
    if full_path[i]+ "/LFR_p_fixed_increasing_k_rtheta_10_28_24.tsv" != base_dir3+ ".DS_Store/LFR_p_fixed_increasing_k_rtheta_10_28_24.tsv":
        file=open(full_path[i]+ "/LFR_p_fixed_increasing_k_rtheta_10_28_24.tsv", 'r')
        reader = csv.reader(file, delimiter="\t")
        row_num=0
        for row in reader:
            row_num=row_num+1
            h_num=0
            if len(row)==31: ##note 29 is the number of columns the csv file should have
                if row_num==3: ##third line contains the data
                    for h in row:
                        joint_csv3.write(h)
                        h_num=h_num+1
                        if h_num==31: ##note 29 is the number of columns the csv file should have
                            joint_csv3.write('\n')
                        else:
                            joint_csv3.write('\t')
                        
                       

   
            
   
joint_csv3.close()
