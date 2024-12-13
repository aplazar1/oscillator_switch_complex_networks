#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:15:47 2024

@author: aplazar1
"""

##create figure2 heat map
from mpl_toolkits import mplot3d
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import csv
from itertools import count
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

####read in the file where parameters were ER saved
data=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/fig_1_2_data_correct/f1_2_plots/ER_csv_multiple_reps_11_18_24.tsv',sep='\t')

graph_summary_data=pd.DataFrame(data)
graph_summary_data=graph_summary_data.query("mean_x_init>0")

##calculate the mean and stanrdard deviation for the ER replicates
G_avg_k_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].G_avg_k)
G_avg_k_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.06].G_avg_k)
G_avg_k_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.16].G_avg_k)
G_avg_k_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.21].G_avg_k)
G_avg_k_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.26].G_avg_k)
G_avg_k_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.31].G_avg_k)
G_avg_k_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.36].G_avg_k)


G_avg_k_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].G_avg_k)
G_avg_k_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.06].G_avg_k)
G_avg_k_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.16].G_avg_k)
G_avg_k_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.21].G_avg_k)
G_avg_k_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.26].G_avg_k)
G_avg_k_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.31].G_avg_k)
G_avg_k_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.36].G_avg_k)

avg_r_theta_eq_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.06].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.16].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.21].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.26].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.31].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.36].avg_r_theta_eq)

avg_r_theta_eq_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.06].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.16].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.21].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.26].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.31].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.36].avg_r_theta_eq)

avg_l_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].avg_l)
avg_l_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.06].avg_l)
avg_l_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.16].avg_l)
avg_l_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.21].avg_l)
avg_l_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.26].avg_l)
avg_l_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.31].avg_l)
avg_l_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.36].avg_l)


avg_l_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].avg_l)
avg_l_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.06].avg_l)
avg_l_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.16].avg_l)
avg_l_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.21].avg_l)
avg_l_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.26].avg_l)
avg_l_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.31].avg_l)
avg_l_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.36].avg_l)


C_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].C)
C_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.06].C)
C_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.16].C)
C_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.21].C)
C_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.26].C)
C_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.31].C)
C_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.36].C)


C_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].C)
C_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.06].C)
C_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.16].C)
C_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.21].C)
C_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.26].C)
C_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.31].C)
C_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.36].C)


p_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.01].p)
p_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.06].p)
p_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.16].p)
p_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.21].p)
p_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.26].p)
p_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.31].p)
p_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.36].p)

p_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.01].p)
p_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.06].p)
p_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.16].p)
p_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.21].p)
p_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.26].p)
p_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.31].p)
p_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.36].p)

degCen_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.01].max_degree_centrality)
degCen_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.06].max_degree_centrality)
degCen_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.16].max_degree_centrality)
degCen_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.21].max_degree_centrality)
degCen_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.26].max_degree_centrality)
degCen_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.31].max_degree_centrality)
degCen_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.36].max_degree_centrality)

degCen_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.01].max_degree_centrality)
degCen_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.06].max_degree_centrality)
degCen_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.16].max_degree_centrality)
degCen_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.21].max_degree_centrality)
degCen_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.26].max_degree_centrality)
degCen_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.31].max_degree_centrality)
degCen_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.36].max_degree_centrality)


G_avg_k_ER=[G_avg_k_rep_p1_avg, G_avg_k_rep_p2_avg, G_avg_k_rep_p3_avg, G_avg_k_rep_p4_avg, G_avg_k_rep_p5_avg, G_avg_k_rep_p6_avg, G_avg_k_rep_p7_avg]
G_sd_k_ER=[G_avg_k_rep_p1_sd, G_avg_k_rep_p2_sd, G_avg_k_rep_p3_sd, G_avg_k_rep_p4_sd, G_avg_k_rep_p5_sd, G_avg_k_rep_p6_sd, G_avg_k_rep_p7_sd]
avg_r_theta_ER=[avg_r_theta_eq_rep_p1_avg,avg_r_theta_eq_rep_p2_avg, avg_r_theta_eq_rep_p3_avg, avg_r_theta_eq_rep_p4_avg, avg_r_theta_eq_rep_p5_avg, avg_r_theta_eq_rep_p6_avg, avg_r_theta_eq_rep_p7_avg]
sd_r_theta_ER=[avg_r_theta_eq_rep_p1_sd,avg_r_theta_eq_rep_p2_sd, avg_r_theta_eq_rep_p3_sd, avg_r_theta_eq_rep_p4_sd, avg_r_theta_eq_rep_p5_sd, avg_r_theta_eq_rep_p6_sd, avg_r_theta_eq_rep_p7_sd]
avg_l_ER=[avg_l_rep_p1_avg, avg_l_rep_p2_avg, avg_l_rep_p3_avg, avg_l_rep_p4_avg, avg_l_rep_p5_avg, avg_l_rep_p6_avg, avg_l_rep_p7_avg]
sd_l_ER=[avg_l_rep_p1_sd, avg_l_rep_p2_sd, avg_l_rep_p3_sd, avg_l_rep_p4_sd, avg_l_rep_p5_sd, avg_l_rep_p6_sd, avg_l_rep_p7_sd]
avg_C_ER=[C_rep_p1_avg, C_rep_p2_avg,C_rep_p3_avg, C_rep_p4_avg, C_rep_p5_avg, C_rep_p6_avg, C_rep_p7_avg]
C_sd_ER=[C_rep_p1_sd, C_rep_p2_sd, C_rep_p3_sd, C_rep_p4_sd, C_rep_p5_sd, C_rep_p6_sd, C_rep_p7_sd]
p_avg_ER=[p_rep_p1_avg, p_rep_p2_avg, p_rep_p3_avg, p_rep_p4_avg, p_rep_p5_avg, p_rep_p6_avg, p_rep_p7_avg]
p_sd_ER=[p_rep_p1_sd, p_rep_p2_sd, p_rep_p3_sd, p_rep_p4_sd, p_rep_p5_sd, p_rep_p6_sd, p_rep_p7_sd]
degCen_avg_ER=[degCen_rep_p1_avg, degCen_rep_p2_avg, degCen_rep_p3_avg, degCen_rep_p4_avg, degCen_rep_p5_avg, degCen_rep_p6_avg, degCen_rep_p7_avg]
degCen_sd_ER=[degCen_rep_p1_sd, degCen_rep_p2_sd, degCen_rep_p3_sd, degCen_rep_p4_sd, degCen_rep_p5_sd, degCen_rep_p6_sd, degCen_rep_p7_sd]



fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig1.suptitle('ER graphs')
ax1.errorbar(G_avg_k_ER, avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=G_sd_k_ER, fmt='d', label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.errorbar(avg_l_ER, avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=sd_l_ER, fmt='d', label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.errorbar(avg_C_ER, avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=C_sd_ER, fmt='x',label="C", color='m')
ax3.errorbar(p_avg_ER, avg_r_theta_ER, yerr=sd_r_theta_ER, xerr=p_sd_ER,fmt='d', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig1.show()
fig1.savefig("ER graph summary params_11_21_24.pdf" )


##plot rtheta vs k
figure7 = plt.figure(figsize=(8,8))
plt.title('rtheta vs Max degCen for ER')
plt.ylabel("r_theta")
plt.xlabel("Max degCen")
plt.errorbar(degCen_avg_ER,avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=degCen_sd_ER, label="degCen", fmt='d', color='k')
figure7.savefig("ER_MaxDegCen_vs_rtheta.pdf" )

##plot max degree centrality vs k
figure8= plt.figure(figsize=(8,8))
plt.title('Max degCen vs <k> for ER')
plt.ylabel("Max degCen")
plt.xlabel("<k>")
plt.errorbar(G_avg_k_ER,degCen_avg_ER, yerr=degCen_sd_ER, xerr=G_sd_k_ER, label="<k>", fmt='d', color='k')
figure8.savefig("ER_avg_k_vs_maxDegCen.pdf" )


##plot rtheta vs number hubs
num_hubs_ER_p1=[]
for i in graph_summary_data[graph_summary_data["p"]==0.01].nodes_with_max_degree_centrality:
    num_hubs_ER_p1.append(len(i.split(',')))
num_hubs_ER_p2=[]
for i in graph_summary_data[graph_summary_data["p"]==0.06].nodes_with_max_degree_centrality:
    num_hubs_ER_p2.append(len(i.split(',')))
num_hubs_ER_p3=[]
for i in graph_summary_data[graph_summary_data["p"]==0.16].nodes_with_max_degree_centrality:
    num_hubs_ER_p3.append(len(i.split(',')))
num_hubs_ER_p4=[]
for i in graph_summary_data[graph_summary_data["p"]==0.21].nodes_with_max_degree_centrality:
    num_hubs_ER_p4.append(len(i.split(',')))
num_hubs_ER_p5=[]
for i in graph_summary_data[graph_summary_data["p"]==0.26].nodes_with_max_degree_centrality:
    num_hubs_ER_p5.append(len(i.split(',')))
num_hubs_ER_p6=[]
for i in graph_summary_data[graph_summary_data["p"]==0.31].nodes_with_max_degree_centrality:
    num_hubs_ER_p6.append(len(i.split(',')))
num_hubs_ER_p7=[]
for i in graph_summary_data[graph_summary_data["p"]==0.36].nodes_with_max_degree_centrality:
    num_hubs_ER_p7.append(len(i.split(',')))    
 
avg_num_hubs_ER=[np.mean(num_hubs_ER_p1), np.mean(num_hubs_ER_p2), np.mean(num_hubs_ER_p3), np.mean(num_hubs_ER_p4), np.mean(num_hubs_ER_p5), np.mean(num_hubs_ER_p6), np.mean(num_hubs_ER_p7)]
std_num_hubs_ER=[np.std(num_hubs_ER_p1), np.std(num_hubs_ER_p2), np.std(num_hubs_ER_p3), np.std(num_hubs_ER_p4), np.std(num_hubs_ER_p5), np.std(num_hubs_ER_p6), np.std(num_hubs_ER_p7)]    

figure9 = plt.figure(figsize=(8,8))
plt.title('rtheta vs number of hubs for ER')
plt.ylabel("r_theta")
plt.xlabel("number of hubs")
plt.errorbar(avg_num_hubs_ER,avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=std_num_hubs_ER, label="r_theta", fmt='d', color='k')
figure9.savefig("ER_num_hubs_vs_rtheta.pdf" )

##plot max degree centrality vs num_hubs
figure10= plt.figure(figsize=(8,8))
plt.title('Max degCen vs <k> for ER')
plt.xlabel("number of hubs")
plt.ylabel("Max DegCen")
plt.errorbar(avg_num_hubs_ER,degCen_avg_ER, yerr=degCen_sd_ER, xerr=std_num_hubs_ER, label="<k>", fmt='o', color='k')
figure10.savefig("ER_Max degCen vs num_hubs.pdf" )


##plot Max DegCen vs <l>
figure15= plt.figure(figsize=(8,8))
plt.title('Max degCen vs <l> for ER')
plt.ylabel("Max degCen")
plt.xlabel("<l>")
plt.errorbar(avg_l_ER,degCen_avg_ER, yerr=degCen_sd_ER, xerr=sd_l_ER, label="<l>", fmt='d', color='k')
figure15.savefig("ER_avg_l_vs_maxDegCen.pdf" )


##plot C vs number of hubs
figure17= plt.figure(figsize=(8,8))
plt.title('C vs number_of_hubs for ER')
plt.ylabel("C")
plt.xlabel("num_hubs")
plt.errorbar(avg_num_hubs_ER,avg_C_ER, yerr=C_sd_ER, xerr=std_num_hubs_ER, label="C", fmt='d', color='k')
figure17.savefig("ER_avg_C_vs_num_hubs.pdf" )


figurey = plt.figure(figsize=(8,8))
plt.title('<k> vs <l> for ER')
plt.ylabel("<l>")
plt.xlabel("<k>")
plt.errorbar(G_avg_k_ER,avg_l_ER,yerr=sd_l_ER, xerr=G_sd_k_ER, label="<l>", fmt='-x', color='k')
figurey.savefig("ER_k_vs_l.pdf" )


##plot hub type vs rtheta for ER
####read in the file where parameters were ER saved
data_hub_type_ER=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/fig_1_2_data_correct/f1_2_plots/ER_csv_multiple_reps_11_18_24_hub_type.csv',sep=',')

graph_hub_type_data_ER=pd.DataFrame(data_hub_type_ER)
graph_hub_type_data_ER=graph_hub_type_data_ER.query("mean_x_init>0")
for i in graph_hub_type_data_ER.G_kmax_node_type:
    list(i)

r_theta_oscillator_hub=graph_hub_type_data_ER[graph_hub_type_data_ER["G_kmax_node_type"] == 'oscillator'].avg_r_theta_eq
r_theta_switch_hub=graph_hub_type_data_ER[graph_hub_type_data_ER["G_kmax_node_type"] == 'switch'].avg_r_theta_eq

figurez=plt.figure(figsize=(8,8))
plt.boxplot([r_theta_switch_hub, r_theta_oscillator_hub])
plt.title("rtheta vs dominant hub node type ER graphs") 
plt.ylabel('<r_theta_eq>')
plt.xlabel("switch hub                                                    oscillator hub")
figurez.savefig("ER_rtheta_vs_hub_type.pdf" )



####read in the file where parameters were SW saved
data2=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/fig_1_2_data_correct/f1_2_plots/SW_csv_multiple_reps_11_21_24.tsv',sep='\t')

graph_summary_data2=pd.DataFrame(data2)
graph_summary_data2=graph_summary_data2.query("mean_x_init>0")



##calculate the mean and stanrdard deviation for the SW replicates
G_avg_k_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].G_avg_k)
G_avg_k_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].G_avg_k)
G_avg_k_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].G_avg_k)
G_avg_k_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].G_avg_k)
G_avg_k_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].G_avg_k)
G_avg_k_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].G_avg_k)
G_avg_k_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].G_avg_k)
G_avg_k_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].G_avg_k)

G_avg_k_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].G_avg_k)
G_avg_k_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].G_avg_k)
G_avg_k_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].G_avg_k)
G_avg_k_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].G_avg_k)
G_avg_k_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].G_avg_k)
G_avg_k_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].G_avg_k)
G_avg_k_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].G_avg_k)
G_avg_k_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].G_avg_k)


avg_r_theta_eq_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].avg_r_theta_eq)
avg_r_theta_eq_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_r_theta_eq)

avg_r_theta_eq_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].avg_r_theta_eq)
avg_r_theta_eq_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_r_theta_eq)

avg_l_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_l)
avg_l_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_l)
avg_l_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_l)
avg_l_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_l)
avg_l_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_l)
avg_l_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_l)
avg_l_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].avg_l)
avg_l_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_l)


avg_l_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_l)
avg_l_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_l)
avg_l_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_l)
avg_l_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_l)
avg_l_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_l)
avg_l_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_l)
avg_l_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].avg_l)
avg_l_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_l)

C_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].C)
C_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].C)
C_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].C)
C_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].C)
C_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].C)
C_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].C)
C_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].C)
C_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].C)

C_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].C)
C_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].C)
C_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].C)
C_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].C)
C_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].C)
C_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].C)
C_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].C)
C_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].C)

degCen_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].max_degree_centrality)
degCen_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].max_degree_centrality)
degCen_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].max_degree_centrality)
degCen_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].max_degree_centrality)
degCen_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].max_degree_centrality)
degCen_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].max_degree_centrality)
degCen_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].max_degree_centrality)
degCen_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].max_degree_centrality)

degCen_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].max_degree_centrality)
degCen_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"]== 10].max_degree_centrality)
degCen_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].max_degree_centrality)
degCen_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].max_degree_centrality)
degCen_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].max_degree_centrality)
degCen_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].max_degree_centrality)
degCen_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].max_degree_centrality)
degCen_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].max_degree_centrality)


G_avg_k_SW=[G_avg_k_rep_p1_avg_SW, G_avg_k_rep_p2_avg_SW, G_avg_k_rep_p3_avg_SW, G_avg_k_rep_p4_avg_SW, G_avg_k_rep_p5_avg_SW, G_avg_k_rep_p6_avg_SW, G_avg_k_rep_p7_avg_SW, G_avg_k_rep_p8_avg_SW]
G_sd_k_SW=[G_avg_k_rep_p1_sd_SW, G_avg_k_rep_p2_sd_SW, G_avg_k_rep_p3_sd_SW, G_avg_k_rep_p4_sd_SW, G_avg_k_rep_p5_sd_SW, G_avg_k_rep_p6_sd_SW, G_avg_k_rep_p7_sd_SW, G_avg_k_rep_p8_sd_SW]
avg_r_theta_SW=[avg_r_theta_eq_rep_p1_avg_SW,avg_r_theta_eq_rep_p2_avg_SW, avg_r_theta_eq_rep_p3_avg_SW, avg_r_theta_eq_rep_p4_avg_SW, avg_r_theta_eq_rep_p5_avg_SW, avg_r_theta_eq_rep_p6_avg_SW, avg_r_theta_eq_rep_p7_avg_SW, avg_r_theta_eq_rep_p8_avg_SW ]
sd_r_theta_SW=[avg_r_theta_eq_rep_p1_sd_SW,avg_r_theta_eq_rep_p2_sd_SW, avg_r_theta_eq_rep_p3_sd_SW, avg_r_theta_eq_rep_p4_sd_SW, avg_r_theta_eq_rep_p5_sd_SW, avg_r_theta_eq_rep_p6_sd_SW, avg_r_theta_eq_rep_p7_sd_SW, avg_r_theta_eq_rep_p8_sd_SW]
avg_l_SW=[avg_l_rep_p1_avg_SW, avg_l_rep_p2_avg_SW, avg_l_rep_p3_avg_SW, avg_l_rep_p4_avg_SW, avg_l_rep_p5_avg_SW, avg_l_rep_p6_avg_SW, avg_l_rep_p7_avg_SW, avg_l_rep_p8_avg_SW]
sd_l_SW=[avg_l_rep_p1_sd_SW, avg_l_rep_p2_sd_SW, avg_l_rep_p3_sd_SW, avg_l_rep_p4_sd_SW, avg_l_rep_p5_sd_SW, avg_l_rep_p6_sd_SW, avg_l_rep_p7_sd_SW, avg_l_rep_p8_sd_SW]
avg_C_SW=[C_rep_p1_avg_SW, C_rep_p2_avg_SW,C_rep_p3_avg_SW, C_rep_p4_avg_SW, C_rep_p5_avg_SW, C_rep_p6_avg_SW, C_rep_p7_avg_SW, C_rep_p8_avg_SW]
C_sd_SW=[C_rep_p1_sd_SW, C_rep_p2_sd_SW, C_rep_p3_sd_SW, C_rep_p4_sd_SW, C_rep_p5_sd_SW, C_rep_p6_sd_SW, C_rep_p7_sd_SW, C_rep_p8_sd_SW]
p_avg_SW=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
avg_degCen_SW=[degCen_rep_p1_avg_SW, degCen_rep_p2_avg_SW, degCen_rep_p3_avg_SW, degCen_rep_p4_avg_SW, degCen_rep_p5_avg_SW, degCen_rep_p6_avg_SW, degCen_rep_p7_avg_SW, degCen_rep_p8_avg_SW]
sd_degCen_SW=[degCen_rep_p1_sd_SW, degCen_rep_p2_sd_SW, degCen_rep_p3_sd_SW, degCen_rep_p4_sd_SW, degCen_rep_p5_sd_SW, degCen_rep_p6_sd_SW,degCen_rep_p7_sd_SW,  degCen_rep_p8_sd_SW]


fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig2.suptitle('SW graphs')
ax1.errorbar(G_avg_k_SW, avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=G_sd_k_SW, fmt='d', label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.errorbar(avg_l_SW, avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=sd_l_SW, fmt='d',label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.errorbar(avg_C_SW, avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=C_sd_SW, fmt='x',label="C", color='m')
ax3.errorbar(p_avg_SW, avg_r_theta_SW, yerr=sd_r_theta_SW, xerr=None,fmt='d', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig2.show()
fig2.savefig("SW_graph_summary_params_11_22_24.pdf" )


##plot rtheta vs k
figure11 = plt.figure(figsize=(8,8))
plt.title('rtheta vs Max degCen for SW')
plt.ylabel("r_theta")
plt.xlabel("Max degCen")
#plt.yscale('log')
plt.errorbar(avg_degCen_SW,avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=sd_degCen_SW, label="degCen", fmt='d', color='b')
figure11.savefig("SW_MaxDegCen_vs_rtheta.pdf" )

##plot max degree centrality vs k
figure12= plt.figure(figsize=(8,8))
plt.title('Max degCen vs <k> for SW')
plt.ylabel("Max degCen")
plt.xlabel("<k>")
plt.errorbar(G_avg_k_SW,avg_degCen_SW, yerr=sd_degCen_SW, xerr=G_sd_k_SW, label="<k>", fmt='d', color='b')
figure12.savefig("SW_avg_k_vs_maxDegCen.pdf" )


##plot rtheta vs number hubs
num_hubs_SW_p1=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].nodes_with_max_degree_centrality:
    num_hubs_SW_p1.append(len(i.split(',')))
num_hubs_SW_p2=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].nodes_with_max_degree_centrality:
    num_hubs_SW_p2.append(len(i.split(',')))
num_hubs_SW_p3=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].nodes_with_max_degree_centrality:
    num_hubs_SW_p3.append(len(i.split(',')))
num_hubs_SW_p4=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].nodes_with_max_degree_centrality:
    num_hubs_SW_p4.append(len(i.split(',')))
num_hubs_SW_p5=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].nodes_with_max_degree_centrality:
    num_hubs_SW_p5.append(len(i.split(',')))
num_hubs_SW_p6=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].nodes_with_max_degree_centrality:
    num_hubs_SW_p6.append(len(i.split(',')))
num_hubs_SW_p7=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 34].nodes_with_max_degree_centrality:
    num_hubs_SW_p7.append(len(i.split(',')))   
num_hubs_SW_p8=[]
for i in graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].nodes_with_max_degree_centrality:
    num_hubs_SW_p8.append(len(i.split(',')))   
 
avg_num_hubs_SW=[np.mean(num_hubs_SW_p1), np.mean(num_hubs_SW_p2), np.mean(num_hubs_SW_p3), np.mean(num_hubs_SW_p4), np.mean(num_hubs_SW_p5), np.mean(num_hubs_SW_p6), np.mean(num_hubs_SW_p7), np.mean(num_hubs_SW_p8)]
std_num_hubs_SW=[np.std(num_hubs_SW_p1), np.std(num_hubs_SW_p2), np.std(num_hubs_SW_p3), np.std(num_hubs_SW_p4), np.std(num_hubs_SW_p5), np.std(num_hubs_SW_p6), np.std(num_hubs_SW_p7), np.std(num_hubs_SW_p8)]    

figure13 = plt.figure(figsize=(8,8))
plt.title('rtheta vs number of hubs for SW')
plt.ylabel("r_theta")
plt.xlabel("number of hubs")
plt.errorbar(avg_num_hubs_SW,avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=std_num_hubs_SW, label="r_theta", fmt='d', color='b')
figure13.savefig("SW_num_hubs_vs_rtheta.pdf" )

##plot max degree centrality vs num_hubs
figure14= plt.figure(figsize=(8,8))
plt.title('Max degCen vs number of hubs for SW')
plt.xlabel("number of hubs")
plt.ylabel("Max DegCen")
plt.errorbar(avg_num_hubs_SW,avg_degCen_SW, yerr=sd_degCen_SW, xerr=std_num_hubs_SW, label="<k>", fmt='o', color='b')
figure14.savefig("SW_Max degCen vs num_hubs.pdf" )


##plot max degree centrality vs k
figure16= plt.figure(figsize=(8,8))
plt.title('Max degCen vs <l> for SW')
plt.ylabel("Max degCen")
plt.xlabel("<l>")
plt.errorbar(avg_l_SW,avg_degCen_SW, yerr=sd_degCen_SW, xerr=sd_l_SW, label="<l>", fmt='d', color='b')
figure16.savefig("SW_avg_l_vs_maxDegCen.pdf" )


##plot max degree centrality vs k
figure17= plt.figure(figsize=(8,8))
plt.title('C vs number of hubs for SW')
plt.ylabel("C")
plt.xlabel("num_hubs")
plt.errorbar(avg_num_hubs_SW, avg_C_SW, yerr=C_sd_SW,xerr=std_num_hubs_SW, label="C", fmt='d', color='b')
figure17.savefig("SW_C_vs_num_hubs.pdf" )

##plot k vs <l>

figurex = plt.figure(figsize=(8,8))
plt.title('<k> vs <l> for SW')
plt.ylabel("<l>")
plt.xlabel("<k>")
plt.errorbar(G_avg_k_SW,avg_l_SW,yerr=sd_l_SW, xerr=G_sd_k_SW, label="<l>", fmt='-x', color='b')
figurex.savefig("SW_k_vs_l.pdf" )


figure3=plt.figure(figsize=(8,8))
plt.title(" SW graph p vs C")
plt.xlabel("p")
plt.ylabel("C")
plt.plot(graph_summary_data2.p, graph_summary_data2.C , 'x')
figure3.savefig("SW p vs C.pdf" )

figure4=plt.figure(figsize=(8,8))
plt.title(" ER graph p vs C")
plt.xlabel("p")
plt.ylabel("C")
plt.plot(graph_summary_data.p, graph_summary_data.C, 'x')
figure4.savefig("ER p vs C.pdf" )



##plot hub type vs rtheta for SW
####read in the file where parameters were SW saved
data_hub_type_SW=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/working_scripts/fig_1_2_data_correct/f1_2_plots/SW_csv_multiple_reps_11_21_24_hub_node_type.csv',sep=',')

graph_hub_type_data_SW=pd.DataFrame(data_hub_type_SW)
graph_hub_type_data_SW=graph_hub_type_data_SW.query("mean_x_init>0")
for i in graph_hub_type_data_SW.G_kmax_node_type:
    list(i)

r_theta_oscillator_hub_SW=graph_hub_type_data_SW[graph_hub_type_data_SW["G_kmax_node_type"] == 'oscillator'].avg_r_theta_eq
r_theta_switch_hub_SW=graph_hub_type_data_SW[graph_hub_type_data_SW["G_kmax_node_type"] == 'switch'].avg_r_theta_eq

figurez2=plt.figure(figsize=(8,8))
plt.boxplot([r_theta_switch_hub_SW, r_theta_oscillator_hub_SW])
plt.title("rtheta vs dominant hub node type SW graphs") 
plt.ylabel('<r_theta_eq>')
plt.xlabel("switch hub                                                    oscillator hub")
figurez2.savefig("SW_rtheta_vs_hub_type.pdf" )


####read in the file where parameters were SW saved
##kfixed while changing p
data3=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/additional_figure_SW_k_10_increasing_p/SW_k_fixed10_increasing_p_rtheta_10_1_24_mod_colnames.csv',sep=',')

graph_summary_data3=pd.DataFrame(data3)
fig5, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig5.suptitle('SW graphs')
ax1.plot(graph_summary_data3.avg_k, graph_summary_data3.avg_r_theta_eq, label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.plot(graph_summary_data3.avg_l, graph_summary_data3.avg_r_theta_eq,'--.' ,label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.plot(graph_summary_data3.C, graph_summary_data3.avg_r_theta_eq, '--x',label="C", color='m')
ax3.plot(graph_summary_data3.p, graph_summary_data3.avg_r_theta_eq, '.', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig5.show()
fig5.savefig("SW graph summary params_k_fixed.png" )





####read in the file where parameters were SW saved
##kfixed while changing p
data4=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/additional_figure_SW_k_increasing_p_fixed/SW_k_increasing_fixed_p_0_03_rtheta_10_1_24_mod_colnames.csv',sep=',')

graph_summary_data4=pd.DataFrame(data4)
fig6, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig6.suptitle('SW graphs')
ax1.plot(graph_summary_data4.avg_k, graph_summary_data4.avg_r_theta_eq, label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.plot(graph_summary_data4.avg_l, graph_summary_data4.avg_r_theta_eq,'--.' ,label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')



ax3.plot(graph_summary_data4.C, graph_summary_data4.avg_r_theta_eq, '--x',label="C", color='m')
ax3.plot(graph_summary_data4.p, graph_summary_data4.avg_r_theta_eq, '.', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig6.show()
fig6.savefig("SW graph summary params_k_increasing_p_fixed.png" )