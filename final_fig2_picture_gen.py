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
data=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/Fig1_2_data/ER_p_k_r_theta_9_26_24_and_10_28_24_mod_colname.csv',sep=',')

graph_summary_data=pd.DataFrame(data)

##calculate the mean and stanrdard deviation for the ER replicates
G_avg_k_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].G_avg_k)
G_avg_k_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.015].G_avg_k)
G_avg_k_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.02].G_avg_k)
G_avg_k_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.05].G_avg_k)
G_avg_k_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.1].G_avg_k)
G_avg_k_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.15].G_avg_k)
G_avg_k_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.2].G_avg_k)
G_avg_k_rep_p8_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.25].G_avg_k)
G_avg_k_rep_p9_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.3].G_avg_k)

G_avg_k_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].G_avg_k)
G_avg_k_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.015].G_avg_k)
G_avg_k_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.02].G_avg_k)
G_avg_k_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.05].G_avg_k)
G_avg_k_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.1].G_avg_k)
G_avg_k_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.15].G_avg_k)
G_avg_k_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.2].G_avg_k)
G_avg_k_rep_p8_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.25].G_avg_k)
G_avg_k_rep_p9_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.3].G_avg_k)

avg_r_theta_eq_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.015].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.02].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.05].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.1].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.15].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.2].avg_r_theta_eq)
avg_r_theta_eq_rep_p8_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.25].avg_r_theta_eq)
avg_r_theta_eq_rep_p9_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.3].avg_r_theta_eq)

avg_r_theta_eq_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.015].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.02].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.05].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.1].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.15].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.2].avg_r_theta_eq)
avg_r_theta_eq_rep_p8_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.25].avg_r_theta_eq)
avg_r_theta_eq_rep_p9_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.3].avg_r_theta_eq)

avg_l_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].avg_l)
avg_l_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.015].avg_l)
avg_l_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.02].avg_l)
avg_l_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.05].avg_l)
avg_l_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.1].avg_l)
avg_l_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.15].avg_l)
avg_l_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.2].avg_l)
avg_l_rep_p8_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.25].avg_l)
avg_l_rep_p9_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.3].avg_l)

avg_l_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].avg_l)
avg_l_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.015].avg_l)
avg_l_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.02].avg_l)
avg_l_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.05].avg_l)
avg_l_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.1].avg_l)
avg_l_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.15].avg_l)
avg_l_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.2].avg_l)
avg_l_rep_p8_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.25].avg_l)
avg_l_rep_p9_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.3].avg_l)

C_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.01].C)
C_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.015].C)
C_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.02].C)
C_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.05].C)
C_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.1].C)
C_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.15].C)
C_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.2].C)
C_rep_p8_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.25].C)
C_rep_p9_avg=np.mean(graph_summary_data[graph_summary_data["p"] == 0.3].C)

C_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.01].C)
C_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.015].C)
C_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.02].C)
C_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.05].C)
C_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.1].C)
C_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.15].C)
C_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.2].C)
C_rep_p8_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.25].C)
C_rep_p9_sd=np.std(graph_summary_data[graph_summary_data["p"] == 0.3].C)

p_rep_p1_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.010].p)
p_rep_p2_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.015].p)
p_rep_p3_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.02].p)
p_rep_p4_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.05].p)
p_rep_p5_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.1].p)
p_rep_p6_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.15].p)
p_rep_p7_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.2].p)
p_rep_p8_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.25].p)
p_rep_p9_avg=np.mean(graph_summary_data[graph_summary_data["p"]==0.3].p)

p_rep_p1_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.010].p)
p_rep_p2_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.015].p)
p_rep_p3_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.02].p)
p_rep_p4_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.05].p)
p_rep_p5_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.1].p)
p_rep_p6_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.15].p)
p_rep_p7_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.2].p)
p_rep_p8_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.25].p)
p_rep_p9_sd=np.std(graph_summary_data[graph_summary_data["p"]==0.3].p)


G_avg_k_ER=[G_avg_k_rep_p1_avg, G_avg_k_rep_p2_avg, G_avg_k_rep_p3_avg, G_avg_k_rep_p4_avg, G_avg_k_rep_p5_avg, G_avg_k_rep_p6_avg, G_avg_k_rep_p7_avg, G_avg_k_rep_p8_avg, G_avg_k_rep_p9_avg]
G_sd_k_ER=[G_avg_k_rep_p1_sd, G_avg_k_rep_p2_sd, G_avg_k_rep_p3_sd, G_avg_k_rep_p4_sd, G_avg_k_rep_p5_sd, G_avg_k_rep_p6_sd, G_avg_k_rep_p7_sd, G_avg_k_rep_p8_sd, G_avg_k_rep_p9_sd]
avg_r_theta_ER=[avg_r_theta_eq_rep_p1_avg,avg_r_theta_eq_rep_p2_avg, avg_r_theta_eq_rep_p3_avg, avg_r_theta_eq_rep_p4_avg, avg_r_theta_eq_rep_p5_avg, avg_r_theta_eq_rep_p6_avg, avg_r_theta_eq_rep_p7_avg, avg_r_theta_eq_rep_p8_avg, avg_r_theta_eq_rep_p9_avg]
sd_r_theta_ER=[avg_r_theta_eq_rep_p1_sd,avg_r_theta_eq_rep_p2_sd, avg_r_theta_eq_rep_p3_sd, avg_r_theta_eq_rep_p4_sd, avg_r_theta_eq_rep_p5_sd, avg_r_theta_eq_rep_p6_sd, avg_r_theta_eq_rep_p7_sd, avg_r_theta_eq_rep_p8_sd, avg_r_theta_eq_rep_p9_sd]
avg_l_ER=[avg_l_rep_p1_avg, avg_l_rep_p2_avg, avg_l_rep_p3_avg, avg_l_rep_p4_avg, avg_l_rep_p5_avg, avg_l_rep_p6_avg, avg_l_rep_p7_avg, avg_l_rep_p8_avg,avg_l_rep_p9_avg]
sd_l_ER=[avg_l_rep_p1_sd, avg_l_rep_p2_sd, avg_l_rep_p3_sd, avg_l_rep_p4_sd, avg_l_rep_p5_sd, avg_l_rep_p6_sd, avg_l_rep_p7_sd, avg_l_rep_p8_sd, avg_l_rep_p9_sd]
avg_C_ER=[C_rep_p1_avg, C_rep_p2_avg,C_rep_p3_avg, C_rep_p4_avg, C_rep_p5_avg, C_rep_p6_avg, C_rep_p7_avg, C_rep_p8_avg, C_rep_p9_avg]
C_sd_ER=[C_rep_p1_sd, C_rep_p2_sd, C_rep_p3_sd, C_rep_p4_sd, C_rep_p5_sd, C_rep_p6_sd, C_rep_p7_sd, C_rep_p8_sd, C_rep_p9_sd]
p_avg_ER=[p_rep_p1_avg, p_rep_p2_avg, p_rep_p3_avg, p_rep_p4_avg, p_rep_p5_avg, p_rep_p6_avg, p_rep_p7_avg, p_rep_p8_avg, p_rep_p9_avg]
p_sd_ER=[p_rep_p1_sd, p_rep_p2_sd, p_rep_p3_sd, p_rep_p4_sd, p_rep_p5_sd, p_rep_p6_sd, p_rep_p7_sd, p_rep_p8_sd, p_rep_p9_sd]




fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig1.suptitle('ER graphs')
ax1.errorbar(G_avg_k_ER, avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=G_sd_k_ER,  label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.errorbar(avg_l_ER, avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=sd_l_ER, label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.errorbar(avg_C_ER, avg_r_theta_ER,yerr=sd_r_theta_ER, xerr=C_sd_ER, fmt='-x',label="C", color='m')
ax3.errorbar(p_avg_ER, avg_r_theta_ER, yerr=sd_r_theta_ER, xerr=p_sd_ER,fmt='-.', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig1.show()
fig1.savefig("ER graph summary params_10_29_24.pdf" )

####read in the file where parameters were SW saved
data2=pd.read_csv('/Users/aplazar1/Documents/code/oscillator_project/APR_paper1/Fig1_2_data/SW_p_fixed_increasing_k_rtheta_10_28_24_mod_col_names.csv',sep=',')

graph_summary_data2=pd.DataFrame(data2)

##calculate the mean and stanrdard deviation for the SW replicates
G_avg_k_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].G_avg_k)
G_avg_k_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].G_avg_k)
G_avg_k_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].G_avg_k)
G_avg_k_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].G_avg_k)
G_avg_k_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].G_avg_k)
G_avg_k_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].G_avg_k)
G_avg_k_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].G_avg_k)
G_avg_k_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].G_avg_k)

G_avg_k_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].G_avg_k)
G_avg_k_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].G_avg_k)
G_avg_k_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].G_avg_k)
G_avg_k_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].G_avg_k)
G_avg_k_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].G_avg_k)
G_avg_k_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].G_avg_k)
G_avg_k_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].G_avg_k)
G_avg_k_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].G_avg_k)


avg_r_theta_eq_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_r_theta_eq)
avg_r_theta_eq_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_r_theta_eq)

avg_r_theta_eq_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].avg_r_theta_eq)
avg_r_theta_eq_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_r_theta_eq)
avg_r_theta_eq_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_r_theta_eq)
avg_r_theta_eq_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_r_theta_eq)
avg_r_theta_eq_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_r_theta_eq)
avg_r_theta_eq_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_r_theta_eq)
avg_r_theta_eq_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_r_theta_eq)
avg_r_theta_eq_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_r_theta_eq)

avg_l_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].avg_l)
avg_l_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_l)
avg_l_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_l)
avg_l_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_l)
avg_l_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_l)
avg_l_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_l)
avg_l_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_l)
avg_l_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_l)


avg_l_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].avg_l)
avg_l_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].avg_l)
avg_l_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].avg_l)
avg_l_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].avg_l)
avg_l_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].avg_l)
avg_l_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].avg_l)
avg_l_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].avg_l)
avg_l_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].avg_l)

C_rep_p1_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].C)
C_rep_p2_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].C)
C_rep_p3_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].C)
C_rep_p4_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].C)
C_rep_p5_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].C)
C_rep_p6_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].C)
C_rep_p7_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].C)
C_rep_p8_avg_SW=np.mean(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].C)

C_rep_p1_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 2].C)
C_rep_p2_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 4].C)
C_rep_p3_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 10].C)
C_rep_p4_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 14].C)
C_rep_p5_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 20].C)
C_rep_p6_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 24].C)
C_rep_p7_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 30].C)
C_rep_p8_sd_SW=np.std(graph_summary_data2[graph_summary_data2["G_avg_k"] == 40].C)





G_avg_k_SW=[G_avg_k_rep_p1_avg_SW, G_avg_k_rep_p2_avg_SW, G_avg_k_rep_p3_avg_SW, G_avg_k_rep_p4_avg_SW, G_avg_k_rep_p5_avg_SW, G_avg_k_rep_p6_avg_SW, G_avg_k_rep_p7_avg_SW, G_avg_k_rep_p8_avg_SW]
G_sd_k_SW=[G_avg_k_rep_p1_sd_SW, G_avg_k_rep_p2_sd_SW, G_avg_k_rep_p3_sd_SW, G_avg_k_rep_p4_sd_SW, G_avg_k_rep_p5_sd_SW, G_avg_k_rep_p6_sd_SW, G_avg_k_rep_p7_sd_SW, G_avg_k_rep_p8_sd_SW]
avg_r_theta_SW=[avg_r_theta_eq_rep_p1_avg_SW,avg_r_theta_eq_rep_p2_avg_SW, avg_r_theta_eq_rep_p3_avg_SW, avg_r_theta_eq_rep_p4_avg_SW, avg_r_theta_eq_rep_p5_avg_SW, avg_r_theta_eq_rep_p6_avg_SW, avg_r_theta_eq_rep_p7_avg_SW, avg_r_theta_eq_rep_p8_avg_SW ]
sd_r_theta_SW=[avg_r_theta_eq_rep_p1_sd_SW,avg_r_theta_eq_rep_p2_sd_SW, avg_r_theta_eq_rep_p3_sd_SW, avg_r_theta_eq_rep_p4_sd_SW, avg_r_theta_eq_rep_p5_sd_SW, avg_r_theta_eq_rep_p6_sd_SW, avg_r_theta_eq_rep_p7_sd_SW, avg_r_theta_eq_rep_p8_sd_SW]
avg_l_SW=[avg_l_rep_p1_avg_SW, avg_l_rep_p2_avg_SW, avg_l_rep_p3_avg_SW, avg_l_rep_p4_avg_SW, avg_l_rep_p5_avg_SW, avg_l_rep_p6_avg_SW, avg_l_rep_p7_avg_SW, avg_l_rep_p8_avg_SW]
sd_l_SW=[avg_l_rep_p1_sd_SW, avg_l_rep_p2_sd_SW, avg_l_rep_p3_sd_SW, avg_l_rep_p4_sd_SW, avg_l_rep_p5_sd_SW, avg_l_rep_p6_sd_SW, avg_l_rep_p7_sd_SW, avg_l_rep_p8_sd_SW]
avg_C_SW=[C_rep_p1_avg_SW, C_rep_p2_avg_SW,C_rep_p3_avg_SW, C_rep_p4_avg_SW, C_rep_p5_avg_SW, C_rep_p6_avg_SW, C_rep_p7_avg_SW, C_rep_p8_avg_SW]
C_sd_SW=[C_rep_p1_sd_SW, C_rep_p2_sd_SW, C_rep_p3_sd_SW, C_rep_p4_sd_SW, C_rep_p5_sd_SW, C_rep_p6_sd_SW, C_rep_p7_sd_SW, C_rep_p8_sd_SW]
p_avg_SW=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]


fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,6))
#fig.figure(figsize = (20, 10))
fig2.suptitle('SW graphs')
ax1.errorbar(G_avg_k_SW, avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=G_sd_k_SW,  label="<k>", color='b')
ax1.set(ylabel='r theta eq')
ax1.set(xlabel='<k>')
ax1.legend()




ax2.errorbar(avg_l_SW, avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=sd_l_SW, label="<l>", color='r')
ax2.legend()
ax2.set(xlabel='<l>')





ax3.errorbar(avg_C_SW, avg_r_theta_SW,yerr=sd_r_theta_SW, xerr=C_sd_SW, fmt='-x',label="C", color='m')
ax3.errorbar(p_avg_SW, avg_r_theta_SW, yerr=sd_r_theta_SW, xerr=None,fmt='D', label="p", color='g')
ax3.legend()
ax3.set(xlabel='C or p')

fig2.show()
fig2.savefig("SW_graph_summary_params_10_29_24.pdf" )


##plot k vs <l>

figurex = plt.figure(figsize=(8,8))
plt.title('<k> vs <l> for SW')
plt.ylabel("<l>")
plt.xlabel("<k>")
plt.errorbar(G_avg_k_SW,avg_l_SW,yerr=sd_l_SW, xerr=G_sd_k_SW, label="<l>", fmt='-x', color='m')
figurex.savefig("SW_k_vs_l.pdf" )


figurey = plt.figure(figsize=(8,8))
plt.title('<k> vs <l> for ER')
plt.ylabel("<l>")
plt.xlabel("<k>")
plt.errorbar(G_avg_k_ER,avg_l_ER,yerr=sd_l_ER, xerr=G_sd_k_ER, label="<l>", fmt='-x', color='m')
figurey.savefig("ER_k_vs_l.pdf" )


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
