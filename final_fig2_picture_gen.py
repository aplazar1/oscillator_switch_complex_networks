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
data=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/Fig1_2/ER_p_k_r_theta_9_26_24_nod_colname.csv',sep=',')

graph_summary_data=pd.DataFrame(data)


fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig1.suptitle('ER graphs')
ax1.plot(graph_summary_data.avg_k, graph_summary_data.avg_r_theta_eq, label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.plot(graph_summary_data.avg_l, graph_summary_data.avg_r_theta_eq, label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.plot(graph_summary_data.C, graph_summary_data.avg_r_theta_eq, 'x',label="C", color='m')
ax3.plot(graph_summary_data.p, graph_summary_data.avg_r_theta_eq, '.', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig1.show()
fig1.savefig("ER graph summary params.png" )

####read in the file where parameters were SW saved
data2=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/Fig1_2/SW_p_k_r_theta_10_1_24_nod_colname.csv',sep=',')

graph_summary_data2=pd.DataFrame(data2)

fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig2.suptitle('SW graphs')
ax1.plot(graph_summary_data2.avg_k, graph_summary_data2.avg_r_theta_eq, label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.plot(graph_summary_data2.avg_l, graph_summary_data2.avg_r_theta_eq, label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.plot(graph_summary_data2.C, graph_summary_data2.avg_r_theta_eq, '--x',label="C", color='m')
ax3.plot(graph_summary_data2.p, graph_summary_data2.avg_r_theta_eq, '.', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig2.show()
fig2.savefig("SW graph summary params2.pdf" )

figure3=plt.figure(figsize=(8,8))
plt.title(" SW graph p vs C")
plt.xlabel("p")
plt.ylabel("C")
plt.plot(graph_summary_data2.p, graph_summary_data2.C)
figure3.savefig("SW p vs C.png" )

figure4=plt.figure(figsize=(8,8))
plt.title(" ER graph p vs C")
plt.xlabel("p")
plt.ylabel("C")
plt.plot(graph_summary_data.p, graph_summary_data.C)
figure4.savefig("ER p vs C.png" )

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