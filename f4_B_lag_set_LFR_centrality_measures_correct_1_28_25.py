#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:16:15 2025

@author: aplaza-rodriguez
"""

## Figure 4 Script for Complex Network Oscillator switch paper
##LFR graphs

##Note: Here I corrected the code for mistakes found in normalization of coupling constants
##normalization should be done according to the  degree of node, taking into account the 
#number of neighborgs of each type
##I also corrected some indexing mistakes in the Euler scheme
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
plt.style.use("ggplot")
import scipy.stats as scp
import networkx as nx



##Model default params in the order described below:200,5,2.5,2.5,0.9,1,4,100,10,3.2,0,100,5,1,1.5,1,0.01,50
#The model parameter mean_x_init can be selected to be positive, zero, or a negative number by the user. Selecting a negative value will 
#initialize all the switches in the "off state" (smaller than -1), selecting zero will initialize some switches in "off" and others in "on state".
##Finally, selecting a positive value for this parameter (larger than 1) will initialize all switches in the "on state".


##model parameters from user input
print('here we sample X init and omega init from a normal distribution')
print('input the model parameters below')
print("enter  simulation parameters: total_nodes, k, power_law1, power_law2, inter_community_edges, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, eta, tau, dt, secs, respectively, separated by commas")
total_nodes, k, power_law1, power_law2, inter_community_edges, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, eta, tau, dt, secs =input().split(',')

#convert input to numbers
total_nodes=int(total_nodes)
k=float(k)
power_law1=float(power_law1)
power_law2=float(power_law2)
inter_community_edges=float(inter_community_edges)
rep=int(rep)
K_t_init=float(K_t_init)
K_max=float(K_theta_max)
Kx_theta=float(Kx_theta)
Kxx=float(Kxx)
mean_omega=float(mean_omega)
mean_x_init=float(mean_x_init)
sigma_omega=float(sigma_omega)
SD_x_init=float(SD_x_init)
eta=float(eta)
tau=float(tau)
dt=float(dt)
secs=float(secs)

##generate 1 Erdos Reny random graph
G1=nx.LFR_benchmark_graph(n=total_nodes,tau1=power_law1, tau2=power_law2, average_degree=k, mu=inter_community_edges) ##use a single graph for all the types of nodes


##sample from a uniform distribution to assign node type
import numpy.random as rn
assignment2=rn.choice(total_nodes-1, total_nodes)

##we want to assign some nodes to be switches and others to be oscillators
##classify some nodes as switches and others as oscillators by adding attribute
for i in range(0, int(G1.number_of_nodes())):
    if assignment2[i] < np.median(assignment2):
        G1.add_node(i, classification='oscillator', color='blue')
    else:
        G1.add_node(i, classification='switch', color='red')
        #print(i)

##create the list of colors for plotting
Node_colors=[]
for i in range(0, int(G1.number_of_nodes())):
        if G1.nodes.data("color")[i]=='blue':
               Node_colors.append('blue')
        elif G1.nodes.data("color")[i]=='red':
               Node_colors.append('red')
               
               
##count the number of each
list_classifications=[G1.nodes.data("classification")[n] for n in range(G1.number_of_nodes())]
num_oscillators=list_classifications.count("oscillator")
num_switches=list_classifications.count("switch")
print("number oscillators")
print(num_oscillators)
print("number switches")
print(num_switches)       


###plot the graph
pos1 = nx.circular_layout(G1)

 
figure0=plt.figure(figsize = (12, 12))
nx.draw_networkx(G1, pos1, node_color=Node_colors) 
figure0.savefig("full_LFR_graph_node_colors.pdf" )          ##10/31  


# Define the natural frequencies of the oscillators
omega_init =scp.cauchy.rvs(loc=mean_omega, scale=sigma_omega, size=num_oscillators)


##phase lag beta
Beta=rn.choice(np.random.uniform(0, 2*np.pi, num_switches), size=1)

# Define the initial phases of the oscillators
##asume that the oscillators are uniformly distributed, and switches are normally distributed.
theta_init = np.random.uniform(0, 2*np.pi, num_oscillators)

##define initial switch state
##Asumme switches were sampled from unimodal, random distribution

x_t_init = np.random.normal(mean_x_init,SD_x_init, num_switches)

# Define the number of time steps
T = int(secs/dt)  #number of steps needed= real time seconds/time step; 100/0.01=10000 total steps
               
               
##select out nodes with given attributes and plot the subgraphs
oscillator_oscillator_subgraph=nx.empty_graph()
oscillator_switch_subgraph=nx.empty_graph()
switch_switch_subgraph=nx.empty_graph()
switch_oscillator_subgraph=nx.empty_graph()

for i in range(G1.number_of_nodes()):
    if G1.nodes.data("classification")[i]=='oscillator':
        #print(G1.nodes.data("classification")[i])
        #print(i)
        neighbors=list(G1.neighbors(i))
        #print(neighbors)
        subgraph_list1=[i]
        subgraph_list2=[i]
        for n in neighbors:
            #print(n)
            if G1.nodes.data("classification")[n]=='oscillator':
                #print(n)
                #print(G1.nodes.data("classification")[n])
                subgraph_list1.append(n)
            if len(subgraph_list1)>1:
                H=G1.subgraph(subgraph_list1)
                #print(H.edges)
                oscillator_oscillator_subgraph=nx.compose(oscillator_oscillator_subgraph, H)
            #print(n)
            if G1.nodes.data("classification")[n]=='switch':
                #print(n)
                #print(G1.nodes.data("classification")[n])
                subgraph_list2.append(n)
            if len(subgraph_list2)>1:
                H=G1.subgraph(subgraph_list2)
                #print(H.edges)
                oscillator_switch_subgraph=nx.compose(oscillator_switch_subgraph, H)
    if G1.nodes.data("classification")[i]=='switch':
        #print(G1.nodes.data("classification")[i])
        #print(i)
        neighbors=list(G1.neighbors(i))
        #print(neighbors)
        subgraph_list1=[i]
        subgraph_list2=[i]
        for n in neighbors:
            #print(n)
            if G1.nodes.data("classification")[n]=='oscillator':
                #print(n)
                #print(type(n))
                #print(G1.nodes.data("classification")[n])
                subgraph_list1.append(n)
            if len(subgraph_list1)>1:
                H=G1.subgraph(subgraph_list1)
                #print(H.edges)
                switch_oscillator_subgraph=nx.compose(switch_oscillator_subgraph, H)
            #print(n)
            if G1.nodes.data("classification")[n]=='switch':
                #print(n)
                #print(G1.nodes.data("classification")[n])
                subgraph_list2.append(n)
            if len(subgraph_list2)>1:
                H=G1.subgraph(subgraph_list2)
                #print(H.edges)
                switch_switch_subgraph=nx.compose(switch_switch_subgraph, H)


##compute and store subgraph parameters
##Oscillator-oscillator subgraphs
LFR_p_K_r_theta_subgraph_log = open("LFR_p_fixed_increasing_k_rtheta_10_01_24_subgraph_parameters_log.txt", "a")
LFR_p_K_r_theta_subgraph_log.write("rep-%01d"%(rep))
LFR_p_K_r_theta_subgraph_log.write("\n")
##1 oscillator-oscillator subgraph         
###draw the oscillator-oscillator subgraph with a circular layout
print("oscillator_oscillator_subgraph number of  nodes and edges")
print(switch_switch_subgraph)
print("\n")
pos2 = nx.circular_layout(oscillator_oscillator_subgraph)


figure1=plt.figure(figsize = (12, 12))
plt.title('oscillator-oscillator subgraph')
nx.draw_networkx(oscillator_oscillator_subgraph, pos2) 
figure1.savefig("oscillator-oscillator subgraph.pdf" )

##compute descriptive grapoh properties for subgraph
C1=nx.average_clustering(oscillator_oscillator_subgraph)
print("C1") 
print(C1) 
print("\n")  
LFR_p_K_r_theta_subgraph_log.write("oscillator_oscillator_subgraph")
LFR_p_K_r_theta_subgraph_log.write("\n")
LFR_p_K_r_theta_subgraph_log.write("C1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(C1))
LFR_p_K_r_theta_subgraph_log.write("\n") 

if nx.is_connected(oscillator_oscillator_subgraph):
    l1=nx.average_shortest_path_length(oscillator_oscillator_subgraph)
    print("l1")
    print(l1)
    print("\n")
    LFR_p_K_r_theta_subgraph_log.write("l1")
    LFR_p_K_r_theta_subgraph_log.write(" ")
    LFR_p_K_r_theta_subgraph_log.write(str(l1))
    LFR_p_K_r_theta_subgraph_log.write("\n") 

    
else:
    LFR_p_K_r_theta_subgraph_log.write("<l1>")
    LFR_p_K_r_theta_subgraph_log.write(" ")
    LFR_p_K_r_theta_subgraph_log.write("N/A")
    LFR_p_K_r_theta_subgraph_log.write("\n")
    print('N/A')
    

##print max degree
degree_view1=dict(oscillator_oscillator_subgraph.degree())
degree_sequence1=dict(sorted(degree_view1.items(), key=lambda item: item[1],reverse=True))
kmax1_nodes=[k for k, v in degree_sequence1.items() if v == max(degree_sequence1.values())]
kmax1 = max(degree_sequence1.values())
print("kmax1_nodes")
print(kmax1_nodes)
print("kmax1")
print(kmax1)

LFR_p_K_r_theta_subgraph_log.write("kmax1 nodes")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(kmax1_nodes))
LFR_p_K_r_theta_subgraph_log.write("\n")   

LFR_p_K_r_theta_subgraph_log.write("kmax1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(kmax1))
LFR_p_K_r_theta_subgraph_log.write("\n")   


##get the max degree dentrality
#calculate degree centrality
degCent1 = nx.degree_centrality(oscillator_oscillator_subgraph)
#Descending order sorting centrality
degCent_sorted1=dict(sorted(degCent1.items(), key=lambda item: item[1],reverse=True))
##get the node with the highest degree centrality
nodes_max_degcen1=[k for k, v in degCent_sorted1.items() if v == max(degCent_sorted1.values())]
Max_degcen1=max(degCent_sorted1.values())
print("nodes_with_max_degree_centrality1")
print(nodes_max_degcen1)
print("Max_degree_centrality1")
print(Max_degcen1)

LFR_p_K_r_theta_subgraph_log.write("nodes_with_max_degree_centrality1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(nodes_max_degcen1))
LFR_p_K_r_theta_subgraph_log.write("\n")   

LFR_p_K_r_theta_subgraph_log.write("Max_degree_centrality1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(Max_degcen1))
LFR_p_K_r_theta_subgraph_log.write("\n") 

##look at the betweennes centrality
##obtain the node with the max betweennes centrality as above
betCent1=nx.betweenness_centrality(oscillator_oscillator_subgraph)
betCent_sorted1=dict(sorted(betCent1.items(), key=lambda item: item[1],reverse=True))
nodes_max_betCen1 = [k for k, v in betCent_sorted1.items() if v == max(betCent_sorted1.values())]
Max_betCen1=max(betCent_sorted1.values())
print("nodes_with_max_betweenness_centrality1")
print(nodes_max_betCen1)
print("Max_betweennes_centrality1")
print(Max_betCen1)

LFR_p_K_r_theta_subgraph_log.write("nodes_with_max_betweenness_centrality1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(nodes_max_betCen1))
LFR_p_K_r_theta_subgraph_log.write("\n")   

LFR_p_K_r_theta_subgraph_log.write("Max_betweennes_centrality1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(Max_betCen1))
LFR_p_K_r_theta_subgraph_log.write("\n") 

louvain_modularity1=nx.community.modularity(oscillator_oscillator_subgraph, nx.community.louvain_communities(oscillator_oscillator_subgraph))
print("louvain_modularity1=")
print(louvain_modularity1)
print("\n")
LFR_p_K_r_theta_subgraph_log.write("louvain_modularity1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(louvain_modularity1))
LFR_p_K_r_theta_subgraph_log.write("\n") 

degree_list=[]
for node in oscillator_oscillator_subgraph.nodes():
    degree=oscillator_oscillator_subgraph.degree[node]
    degree_list.append(degree)
average_degree_of_graph1=np.sum(degree_list)/len(oscillator_oscillator_subgraph.nodes())
print("average_degree_of_graph1=") 
print(average_degree_of_graph1) 
print("\n")   
LFR_p_K_r_theta_subgraph_log.write("average_degree_of_graph1")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph1))
LFR_p_K_r_theta_subgraph_log.write("\n")     



##Switch-Switch Subgraph
###draw the switch-switch subgraph with a circular layout
print('switch-switch subgraph number of nodes and edges')
print(switch_switch_subgraph)
print("\n")
pos4 = nx.circular_layout(switch_switch_subgraph)

 
figure3=plt.figure(figsize = (12, 12))
plt.title('switch_switch_subgraph subgraph')
nx.draw_networkx(switch_switch_subgraph, pos4)
figure3.savefig("switch-switch subgraph.pdf" )

C3=nx.average_clustering(switch_switch_subgraph)
print("C3=")   
print(C3)   
print("\n") 
LFR_p_K_r_theta_subgraph_log.write("switch_switch_subgraph")
LFR_p_K_r_theta_subgraph_log.write("\n")
LFR_p_K_r_theta_subgraph_log.write("C3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(C3))
LFR_p_K_r_theta_subgraph_log.write("\n") 

if nx.is_connected(switch_switch_subgraph):
    l3=nx.average_shortest_path_length(switch_switch_subgraph)
    print("l3")
    print(l3)
    print("\n")
    LFR_p_K_r_theta_subgraph_log.write("<l3>")
    LFR_p_K_r_theta_subgraph_log.write(" ")
    LFR_p_K_r_theta_subgraph_log.write(str(l3))
    LFR_p_K_r_theta_subgraph_log.write("\n") 
    
else:
    LFR_p_K_r_theta_subgraph_log.write("<l3>")
    LFR_p_K_r_theta_subgraph_log.write(" ")
    LFR_p_K_r_theta_subgraph_log.write("N/A")
    LFR_p_K_r_theta_subgraph_log.write("\n")
    print('N/A')


##print max degree
degree_view3=dict(switch_switch_subgraph.degree())
degree_sequence3=dict(sorted(degree_view3.items(), key=lambda item: item[1],reverse=True))
kmax3_nodes=[k for k, v in degree_sequence3.items() if v == max(degree_sequence3.values())]
kmax3 = max(degree_sequence3.values())
print("kmax3_nodes")
print(kmax3_nodes)
print("kmax3")
print(kmax3)

LFR_p_K_r_theta_subgraph_log.write("kmax3 nodes")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(kmax3_nodes))
LFR_p_K_r_theta_subgraph_log.write("\n")   

LFR_p_K_r_theta_subgraph_log.write("kmax3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(kmax3))
LFR_p_K_r_theta_subgraph_log.write("\n")   


##get the max degree dentrality
#calculate degree centrality
degCent3 = nx.degree_centrality(switch_switch_subgraph)
#Descending order sorting centrality
degCent_sorted3=dict(sorted(degCent3.items(), key=lambda item: item[1],reverse=True))
##get the node with the highest degree centrality
nodes_max_degcen3=[k for k, v in degCent_sorted3.items() if v == max(degCent_sorted3.values())]
Max_degcen3=max(degCent_sorted3.values())
print("nodes_with_max_degree_centrality3")
print(nodes_max_degcen3)
print("Max_degree_centrality3")
print(Max_degcen3)

LFR_p_K_r_theta_subgraph_log.write("nodes_with_max_degree_centrality3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(nodes_max_degcen3))
LFR_p_K_r_theta_subgraph_log.write("\n")   

LFR_p_K_r_theta_subgraph_log.write("Max_degree_centrality3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(Max_degcen3))
LFR_p_K_r_theta_subgraph_log.write("\n") 

##look at the betweennes centrality
##obtain the node with the max betweennes centrality as above
betCent3=nx.betweenness_centrality(switch_switch_subgraph)
betCent_sorted3=dict(sorted(betCent3.items(), key=lambda item: item[1],reverse=True))
nodes_max_betCen3 = [k for k, v in betCent_sorted3.items() if v == max(betCent_sorted3.values())]
Max_betCen3=max(betCent_sorted3.values())
print("nodes_with_max_betweenness_centrality3")
print(nodes_max_betCen3)
print("Max_betweennes_centrality3")
print(Max_betCen3)

LFR_p_K_r_theta_subgraph_log.write("nodes_with_max_betweenness_centrality3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(nodes_max_betCen3))
LFR_p_K_r_theta_subgraph_log.write("\n")   

LFR_p_K_r_theta_subgraph_log.write("Max_betweennes_centrality3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(Max_betCen3))
LFR_p_K_r_theta_subgraph_log.write("\n") 

louvain_modularity3=nx.community.modularity(switch_switch_subgraph, nx.community.louvain_communities(switch_switch_subgraph))
print("louvain_modularity3")
print(louvain_modularity3)
print("\n")
LFR_p_K_r_theta_subgraph_log.write("louvain_modularity3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(louvain_modularity3))
LFR_p_K_r_theta_subgraph_log.write("\n") 

degree_list=[]
for node in switch_switch_subgraph.nodes():
    degree=switch_switch_subgraph.degree[node]
    degree_list.append(degree)
average_degree_of_graph3=np.sum(degree_list)/len(switch_switch_subgraph.nodes())
print("average_degree_of_graph3")  
print(average_degree_of_graph3)  
print("\n")
LFR_p_K_r_theta_subgraph_log.write("average_degree_of_graph3")
LFR_p_K_r_theta_subgraph_log.write(" ")
LFR_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph3))
LFR_p_K_r_theta_subgraph_log.write("\n") 


##Obtain the overall adjacency matrix to ensure correct simulation

h=nx.to_numpy_array(G1)

# Define the array to store results at each time step
model_state_t = np.ndarray((T, total_nodes))
K_t = np.zeros((T))
K_t[0]=K_t_init
theta_t=np.zeros((T, num_oscillators))

x_t=np.zeros((T, num_switches))
x_t[0]=x_t_init
theta_t[0]=theta_init
x_t_discrete=np.zeros((T, num_switches))


# Run the simulation
print("euler scheme calculation")
for t in range(0, T):
    if t==0:
        oscillator_num=0
        switch_num=0
        #print(t)
        for i in range(G1.number_of_nodes()):
            if G1.nodes.data("classification")[i]=='oscillator':
                #print(i)
                #print(oscillator_num)
                model_state_t[t,i]=theta_init[oscillator_num]
                oscillator_num=oscillator_num+1
            elif G1.nodes.data("classification")[i]=='switch':
                #print(i)
                #print(switch_num)
                model_state_t[t,i]=x_t_init[switch_num]
                ##clasify states as boolean discrete
                if model_state_t[t,i] <= 0:
                    x_t_discrete[t,switch_num]=0
                elif model_state_t[t,i]>0:
                    x_t_discrete[t,switch_num]=1
                switch_num=switch_num+1
                 
    else:
        oscillator_num=0
        switch_num=0
        switch_oscillator_K_infl=[]
        r_link_oscillator_sum=[]
        for i in range(G1.number_of_nodes()):
            if G1.nodes.data("classification")[i]=='oscillator':
                oscillator_oscillator_neighborg_contributions=[]
                neighbors=list(G1.neighbors(i))
                #print(neighbors)
                if len(neighbors)> 0: 
                    neighborg_classification=[G1.nodes.data("classification")[n] for n in neighbors]
                    num_oscillator_oscillator_neighborgs=neighborg_classification.count("oscillator")
                    #r_link=[]
                    #print(neighbors)
                    #print(num_oscillator_oscillator_neighborgs)
                    for n in neighbors:
                        #print(G1.nodes.data("classification")[n])
                        #print(i)
                        #print(n)
                        if G1.nodes.data("classification")[n]=='oscillator':
                            #print((i,n))
                            #print(oscillator_oscillator_neighborg_contributions)
                            ##oscillator rate of phase change with time
                            oscillator_neighborg_contribution=(h[i][n]*np.sin(model_state_t[t-1, n] - model_state_t[t-1, i]))/num_oscillator_oscillator_neighborgs
                            oscillator_oscillator_neighborg_contributions.append(oscillator_neighborg_contribution)    
                    model_state_t[t,i] = (model_state_t[t-1, i] + dt*(omega_init[oscillator_num] + ((K_t[t-1])*np.sum(oscillator_oscillator_neighborg_contributions))))%(2*np.pi)
                    theta_t[t, oscillator_num]=model_state_t[t, i]
                    #print(len(oscillator_oscillator_neighborg_contributions))
                elif len(neighbors)==0: 
                    model_state_t[t, i] = (model_state_t[t-1, i] + dt*(omega_init[oscillator_num]))%(2*np.pi)
                    theta_t[t, oscillator_num]=model_state_t[t, i]
                oscillator_num=oscillator_num+1
            ##switch state vs time
            if G1.nodes.data("classification")[i]=='switch':
                neighbors=list(G1.neighbors(i))                                                                                               
                if len(neighbors)> 0:
                    neighborg_classification=[G1.nodes.data("classification")[n] for n in neighbors]
                    num_switch_oscillator_neighborgs=neighborg_classification.count('oscillator')
                    num_switch_switch_neighborgs=neighborg_classification.count('switch')
                    #print(num_switch_oscillator_neighborgs)
                    #print(num_switch_switch_neighborgs)
                    switch_neighborg_infl=[]
                    oscillator_neighborg_infl=[]
                    for n in neighbors:
                        #print(n)
                        #print((i,n))
                        #print(len(switch_neighborg_infl))
                        if G1.nodes.data("classification")[n]=='switch': 
                            
                            if model_state_t[t-1,n]>0:
                                l=1
                                switch_switch_influence=(h[i][n]*l)/num_switch_switch_neighborgs
                                switch_neighborg_infl.append(switch_switch_influence)
                              #print(i)
                              #print(n)
                            else: 
                                l=0  ###do i want this to be an else if statement?
                                switch_switch_influence=(h[i][n]*l)/num_switch_switch_neighborgs
                                switch_neighborg_infl.append(switch_switch_influence)
                        elif G1.nodes.data("classification")[n]=='oscillator':
                            oscillator_switch_influence=(h[i][n]*np.sin(model_state_t[t-1,n] - Beta))/num_switch_oscillator_neighborgs
                            oscillator_neighborg_infl.append(oscillator_switch_influence)
                            if model_state_t[t-1,i]>0:
                                l=1
                                switch_oscillator_influence=(h[i][n]*l)/num_switch_oscillator_neighborgs
                                switch_oscillator_K_infl.append(switch_oscillator_influence)
                            else:
                                l=0
                                switch_oscillator_influence=(h[i][n]*l)/num_switch_oscillator_neighborgs
                                switch_oscillator_K_infl.append(switch_oscillator_influence)           
                    model_state_t[t,i] = model_state_t[t-1, i] + dt*(-model_state_t[t-1,i] - eta +  (Kxx)*np.sum(switch_neighborg_infl) +  (Kx_theta)*np.sum(oscillator_neighborg_infl))  
                    x_t[t,switch_num]=model_state_t[t,i] 
                    #print(len(switch_neighborg_infl))                                                                                     
                elif len(neighbors)==0:
                    model_state_t[t,i] = model_state_t[t-1, i] + dt*(-model_state_t[t-1,i] - eta )
                    x_t[t,switch_num]=model_state_t[t,i] 
                ##clasify states as discrete
                if model_state_t[t,i] <= 0:
                    x_t_discrete[t,switch_num]=0
                elif model_state_t[t,i]>0:
                    x_t_discrete[t,switch_num]=1
                switch_num=switch_num+1     
        ##rate of change of oscillator coupling constant vs time
        K_t[t]= K_t[t-1] + dt*(-K_t[t-1] + ((K_max/len(switch_oscillator_K_infl))*np.sum(switch_oscillator_K_infl)))*(1/tau)
    
# Plot the oscillators in the simulation

f4,ax=plt.subplots(figsize=(10,10))
theta_t_pie=(theta_t)
plt.plot(theta_t_pie)
plt.xlabel('model time steps')
plt.ylabel('Phase')
f4.savefig("phase.pdf" )

##plot switch state
f5,ax=plt.subplots(figsize=(10,10))
plt.plot(x_t_discrete)
plt.xlabel('model time steps')
plt.ylabel('state')
f5.savefig("switch state.pdf" )


##calculate the average phase psi
psi=np.mean(theta_t_pie, axis=1)
f6,ax=plt.subplots(figsize=(10,10))
plt.plot(psi)
plt.xlabel('model time steps')
plt.ylabel('psi average phase angle')
f6.savefig("psi.pdf" )

##compute the order parameter(s) at each model timestep
r_theta_list = []
r_x_list=[]


#Note: I found that the mean field order parameters characterized the state of the system well
##for both well connected and sparsely connected graphs
for t in range(T):
        #print(t)
        ##for this order parameter, we sum over all oscillators frequencies at a given time
        #print(np.sum(np.exp(1j*theta_t_pie[t])))
        #print(np.exp(1j*psi[t]))
        r_theta=abs((1/num_oscillators)*np.sum(np.exp(1j*theta_t_pie[t]-1j*psi[t])))
        #print(r_theta)
        r_theta_list.append(r_theta)
        ##switch order parameter
        r_x=(np.sum(x_t_discrete[t])/num_switches)
        r_x_list.append(r_x)



#perform autocorrelation  and cross correlation analysis of order parameters, specifically the switch order param
if max(r_x_list)> 0: 
    rx_auto_corr=np.correlate(r_x_list, r_x_list, mode='same')/(max(np.correlate(r_x_list, r_x_list, mode='same')))
    order_param_crosscorr=np.correlate(r_x_list, r_theta_list, mode='same')/(max(np.correlate(r_x_list, r_theta_list, mode='same')))


r_theta_auto_corr=np.correlate(r_theta_list, r_theta_list, mode='same')/(max(np.correlate(r_theta_list, r_theta_list, mode='same')))


#perform fourier analysis of the switch order parameter
##calculate the real_time_steps
time=np.linspace(0,secs,int(secs/dt))

from scipy.fft import fft
from scipy.fft import fftfreq
x_fft=fft(np.mean(x_t, axis=1)) ##carry out fourier analysis on switch state, not switch order param
xf = fftfreq(1000, 0.01)[:1000//2]
rx_fft=fft(r_x_list)
r_theta_fft=fft(r_theta_list)
psi_fft=fft(psi)


##output time-series of fourier transforms
fourier_correlation_analysys_LFR=open("LFR_fourier_analysis_10_30_24.tsv", "a")
fourier_correlation_analysys_LFR.write('frequency domain xf')
fourier_correlation_analysys_LFR.write("\t")
fourier_correlation_analysys_LFR.write("r_theta_fft")
fourier_correlation_analysys_LFR.write("\t")
fourier_correlation_analysys_LFR.write("psi_fft")
if max(r_x_list)> 0: 
    fourier_correlation_analysys_LFR.write("\t")
    fourier_correlation_analysys_LFR.write("rx_fft")
    fourier_correlation_analysys_LFR.write("\t")
    fourier_correlation_analysys_LFR.write("<x(t)> fft")
fourier_correlation_analysys_LFR.write("\n")


for t in range(len(xf)):
    fourier_correlation_analysys_LFR.write(str(xf[t]))
    fourier_correlation_analysys_LFR.write("\t")
    fourier_correlation_analysys_LFR.write(str(r_theta_fft[t]))
    fourier_correlation_analysys_LFR.write("\t")
    fourier_correlation_analysys_LFR.write(str(psi_fft[t]))
 
    if max(r_x_list)> 0: 
        fourier_correlation_analysys_LFR.write("\t")
        fourier_correlation_analysys_LFR.write(str(rx_fft[t]))
        fourier_correlation_analysys_LFR.write("\t")
        fourier_correlation_analysys_LFR.write(str(x_fft[t]))
    fourier_correlation_analysys_LFR.write("\n")





##plot the order parameters
f7,ax=plt.subplots(figsize=(10,10))
plt.plot(time,r_theta_list,'r', label='r_theta')
plt.plot(time,r_x_list,'b', label='r_x')
plt.legend()
plt.title('order parameters')
plt.xlabel('t(secs)')
f7.savefig("order params.pdf" )



##calculate and export the r_theta_time_average_over_the_last 100 time steps
time_avg_r_theta_eq=np.mean(r_theta_list[-1000:], axis=0)
time_avg_r_x_eq=np.mean(r_x_list[-1000:], axis=0)

print("time_avg_r_theta_eq")
print(time_avg_r_theta_eq)
print("time_avg_r_x_eq")
print(time_avg_r_x_eq)


##plot the oscillator-oscillator coupling constant vs time 
import matplotlib.pyplot as plt
f8,ax=plt.subplots(figsize=(10,10))
plt.plot(time, K_t,'g', label='K_t')
plt.legend()
plt.title('oscillator coupling coefficient')
plt.xlabel('t(s)')
plt.ylabel('K_t')
f8.savefig("K_t.pdf" )



##plot the switch state vs time 
import matplotlib.pyplot as plt
f9,ax=plt.subplots(figsize=(10,10))
plt.plot(time,np.mean(x_t, axis=1),'m', label='<x_t>')
plt.plot(time,np.std(x_t, axis=1),'k', label='SD x_t')
plt.legend()
plt.xlabel('t(secs)')
plt.ylabel(' average switch state <x(t)>')  
f9.savefig("average_x_t.pdf" ) 

##plot the signal correlations
if max(r_x_list)> 0: 
    f10, (ax1, ax2, ax3)=plt.subplots(1, 3, figsize=(10,10))
    f10.suptitle('order parameter cross and autocorrelations')
    ax1.plot(rx_auto_corr[int((len(rx_auto_corr)/2)):],'b', label='r_x autocorr')
    ax2.plot(r_theta_auto_corr[int((len(r_theta_auto_corr)/2)):],'r', label='r_theta autocorr')
    ax3.plot(order_param_crosscorr[int(len(r_theta_auto_corr)/2):], 'g', label='order param crosscorr')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax2.set(xlabel='time lag')
    ax1.set(ylabel='signal correlation')
    f10.savefig("auto_and_cross_corr.pdf" ) 
else:
    f10, (ax2)=plt.subplots(1, 1, figsize=(10,10))
    f10.suptitle('order parameter cross and autocorrelations')
    
    ax2.plot(r_theta_auto_corr[int((len(r_theta_auto_corr)/2)):],'r', label='r_theta autocorr')
   
   
    ax2.legend()

    ax2.set(xlabel='time lag')
    ax2.set(ylabel='signal correlation')
    f10.savefig("auto_and_cross_corr.pdf" ) 

##plot the fourier transforms of the order parameters
##as well as fft of average phase psi and <x(t)>

f11, (ax1, ax2, ax3, ax4)=plt.subplots(1, 4, figsize=(15,10))
f11.suptitle('FFT decomposition of order parameters')
ax1.plot(xf,rx_fft[0:len(xf)],'b', label='r_x fft')
ax2.plot(xf,r_theta_fft[0:len(xf)],'r', label='r_theta fft')
ax3.plot(xf,x_fft[0:len(xf)], 'm', label='<x(t)> fft')
ax4.plot(xf, psi_fft[0:len(xf)], 'k', label='psi(t) fft')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax2.set(xlabel='frequency domain')
ax3.set(xlabel='frequency domain')
ax4.set(xlabel='frequency domain')
ax1.set(xlabel='frequency domain')
ax1.set(ylabel='amplitude domain')
f11.savefig("order_param_fft.pdf" ) 


##Compute graph properties for the overall network
LFR_p_K_r_theta = open("LFR_p_fixed_increasing_k_rtheta_10_28_24.tsv", "a")
LFR_p_K_r_theta.write("rep-%01d"%(rep))
LFR_p_K_r_theta.write("\n")
LFR_p_K_r_theta.write("power_law1")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("power_law2")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("inter_community_edges") 
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("C")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("<l>")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("G kmax node")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("kmax node type")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("G kmax")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("nodes_with_max_degree_centrality")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("max_degree_centrality")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("nodes_with_max_betweeness_centrality")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("max_betweeness_centrality")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("louvain modularity")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("G <k>")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("K_t_init")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("mean_omega")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("sigma_omega")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("K_max")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("Kx_theta")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("Kxx")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("eta")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("tau")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("beta_l")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("<r_x_eq>")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("<r_theta_eq>")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("num_switches")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("num_oscillators")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("mean_x_init")
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write("SD_x_init")
LFR_p_K_r_theta.write("\n")

LFR_p_K_r_theta.write(str(power_law1))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(power_law2))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(inter_community_edges))
LFR_p_K_r_theta.write("\t")

##compute the average clustering of the network
print("overall graph descriptive indices")
print("C")
C=nx.average_clustering(G1)
print(C)  
print("\n")
LFR_p_K_r_theta.write(str(C))
LFR_p_K_r_theta.write("\t")  

 

clustering_per_node=nx.clustering(G1, list(G1.nodes()))

if nx.is_connected(G1):
    print("<l>")
    l=nx.average_shortest_path_length(G1)
    print(l)
    print("\n") 
    LFR_p_K_r_theta.write(str(l))
    LFR_p_K_r_theta.write("\t")
    
else:
    LFR_p_K_r_theta.write("N/A")
    LFR_p_K_r_theta.write("\t")
    print('N/A')
    
##print max degree
degree_view=dict(G1.degree())
degree_sequence=dict(sorted(degree_view.items(), key=lambda item: item[1],reverse=True))
kmax_nodes=[k for k, v in degree_sequence.items() if v == max(degree_sequence.values())]
kmax = max(degree_sequence.values())
print("kmax_nodes")
print(kmax_nodes)
print("kmax")
print(kmax)

kmax_node_type=[]
for k in kmax_nodes: 
    kmax_node_type.append(G1.nodes.data("classification")[k])

LFR_p_K_r_theta.write(str(kmax_nodes))
LFR_p_K_r_theta.write("\t") 
LFR_p_K_r_theta.write(str(kmax_node_type))
LFR_p_K_r_theta.write("\t")  
LFR_p_K_r_theta.write(str(kmax))
LFR_p_K_r_theta.write("\t")   

##get the max degree dentrality
#calculate degree centrality
degCent = nx.degree_centrality(G1)
#Descending order sorting centrality
degCent_sorted=dict(sorted(degCent.items(), key=lambda item: item[1],reverse=True))
##get the node with the highest degree centrality
nodes_max_degcen=[k for k, v in degCent_sorted.items() if v == max(degCent_sorted.values())]
Max_degcen=max(degCent_sorted.values())
print("nodes_with_max_degree_centrality")
print(nodes_max_degcen)
print("Max_degree_centrality")
print(Max_degcen)


LFR_p_K_r_theta.write(str(nodes_max_degcen))
LFR_p_K_r_theta.write("\t")   


LFR_p_K_r_theta.write(str(Max_degcen))
LFR_p_K_r_theta.write("\t") 

##look at the betweennes centrality
##obtain the node with the max betweennes centrality as above
betCent=nx.betweenness_centrality(switch_switch_subgraph)
betCent_sorted=dict(sorted(betCent.items(), key=lambda item: item[1],reverse=True))
nodes_max_betCen = [k for k, v in betCent_sorted.items() if v == max(betCent_sorted.values())]
Max_betCen=max(betCent_sorted.values())
print("nodes_with_max_betweeness_centrality")
print(nodes_max_betCen)
print("Max_betweennes_centrality")
print(Max_betCen)


LFR_p_K_r_theta.write(str(nodes_max_betCen))
LFR_p_K_r_theta.write("\t")   


LFR_p_K_r_theta.write(str(Max_betCen))
LFR_p_K_r_theta.write("\t") 


louvain_modularity=nx.community.modularity(G1, nx.community.louvain_communities(G1))
print("louvain_modularity")
print(louvain_modularity)
print("\n") 
LFR_p_K_r_theta.write(str(louvain_modularity))
LFR_p_K_r_theta.write("\t")  

degree_list=[]
for node in G1.nodes():
    degree=G1.degree[node]
    degree_list.append(degree)
average_degree_of_graph=np.sum(degree_list)/len(G1.nodes())
print("<k> of graph")
print(average_degree_of_graph)
print("\n") 
LFR_p_K_r_theta.write(str(average_degree_of_graph))
LFR_p_K_r_theta.write("\t")  

##export simulation params and calculated model and graph proeprties to a tsv file
set_Beta=Beta

print('computing order params within hubs')
print('oscillator subgraph hub')
##compute the order parameters within the HUBS of the subgraphs and overall graph
##oscillator-oscillator subgraph
if len(nodes_max_degcen1)>1:
    print("many hubs")
    hub_num=0
    r_theta_hub_oscillator_subgraph=np.zeros((T, len(nodes_max_degcen1)))
    for i in oscillator_oscillator_subgraph.nodes():
        if i in nodes_max_degcen1:
            #print(i)
            for t in range(0,T):
                r_i_theta_list=[]
                for j in oscillator_oscillator_subgraph.nodes():
                    if i!=j:  
                        r_i_theta=abs(h[i][j]*np.exp(1j*model_state_t[t, j]-1j*model_state_t[t, i]))
                        r_i_theta_list.append(r_i_theta)
                        #print(r_i_theta)
                        #print(h[i][j])
                        #print((i,j))
                r_theta_hub=np.sum(r_i_theta_list)/oscillator_oscillator_subgraph.degree(i)
                #print(np.sum(r_i_theta_list))
                #print(oscillator_oscillator_subgraph.degree(i))
                #print(r_theta_hub)
                #print(hub_num)
                r_theta_hub_oscillator_subgraph[t,hub_num]=r_theta_hub
            #print(hub_num)
            hub_num=hub_num+1
        
elif len(nodes_max_degcen1)==1:
    print("1 hub")
    r_theta_hub_oscillator_subgraph=[]
    for i in oscillator_oscillator_subgraph.nodes():
        if i in nodes_max_degcen1:
            for t in range(0,T):
                r_i_theta_list=[]
                for j in oscillator_oscillator_subgraph.nodes():
                    if i!=j: 
                        r_i_theta=abs(h[i][j]*np.exp(1j*model_state_t[t, j]-1j*model_state_t[t, i]))
                        r_i_theta_list.append(r_i_theta)
                        #print(r_i_theta)
                        #print(r_i_theta_list)
                        #print(len(r_i_theta_list))
                        #print(np.sum(r_i_theta_list)/num_oscillators)
                        #print((i,j))
                #print(t)
                #print(len(r_i_theta_list))
                #print(np.sum(r_i_theta_list))
                #print(oscillator_oscillator_subgraph.degree(i))
                r_theta_hub=np.sum(r_i_theta_list)/oscillator_oscillator_subgraph.degree(i)
                #print(r_theta_hub)
                r_theta_hub_oscillator_subgraph.append(r_theta_hub)
            

print('switch subgraph hub')
##switch-switch subgraph 
if len(nodes_max_degcen3)>1:
    print("many hubs hub")
    hub_num=0
    rx_hub_switch_subgraph=np.zeros((T, len(nodes_max_degcen3)))
    for i in switch_switch_subgraph.nodes():
        if i in nodes_max_degcen3:
            for t in range(0,T):
                r_i_x_list=[]
                for j in switch_switch_subgraph.nodes():
                    if model_state_t[t, j]>0:
                        l=1
                        r_i_x_list.append(h[i][j]*l)
                    else:
                        l=0
                        r_i_x_list.append(h[i][j]*l)
                rx_hub_switch_subgraph[t, hub_num]=np.sum(r_i_x_list)/switch_switch_subgraph.degree(i)
            hub_num=hub_num+1
        
elif len(nodes_max_degcen3)==1:
    print("1  hub")
    rx_hub_switch_subgraph=[]
    for i in switch_switch_subgraph.nodes():
            if i in nodes_max_degcen3:
                for t in range(0,T):
                    r_i_x_list=[]
                    for j in switch_switch_subgraph.nodes():
                        if model_state_t[t, j]>0:
                            l=1
                            r_i_x_list.append(h[i][j]*l)
                        else:
                            l=0
                            r_i_x_list.append(h[i][j]*l)
                    #print(len(r_i_x_list))
                    rx_hub_switch_subgraph.append(np.sum(r_i_x_list)/switch_switch_subgraph.degree(i))



##calculate the HUB order parameters for hubs in the overall graph 
print("overall graph hub")   
if len(nodes_max_degcen)>1:
    print("many hubs")
    switch_hubs=0
    oscillator_hubs=0
    hub_classification=[G1.nodes.data("classification")[n] for n in nodes_max_degcen]
    num_oscillator_hubs=hub_classification.count("oscillator")
    num_switch_hubs=hub_classification.count("switch")
    rx_hub_overall=np.zeros((T,num_switch_hubs))
    r_theta_hub_overall=np.zeros((T, num_oscillator_hubs))
    for i in G1.nodes():
        if i in nodes_max_degcen:
            if G1.nodes.data("classification")[i]=='oscillator':
             for t in range(0,T):
                r_i_theta_list=[]
                for j in G1.nodes():
                    if G1.nodes.data("classification")[j]=='oscillator' and i!=j:  
                            r_i_theta=abs(h[i][j]*np.exp(1j*model_state_t[t, j]-1j*model_state_t[t, i]))
                            r_i_theta_list.append(r_i_theta)
                            #print(r_i_theta)
                            #print(h[i][j])
                #print(len(r_i_theta_list))
                r_theta_hub=np.sum(r_i_theta_list)/oscillator_oscillator_subgraph.degree(i)
                r_theta_hub_overall[t,oscillator_hubs]=r_theta_hub
                #print(hub_num)
             oscillator_hubs=oscillator_hubs+1
            if G1.nodes.data("classification")[i]=='switch':
             for t in range(0,T):
                r_i_x_list=[]
                for j in G1.nodes():
                    if G1.nodes.data("classification")[j]=='switch':  
                        if model_state_t[t, j]>0:
                            l=1
                            r_i_x_list.append(h[i][j]*l)
                        else:
                           l=0
                           r_i_x_list.append(h[i][j]*l)
                #print(len(r_i_x_list))
                rx_hub_overall[t,switch_hubs]=np.sum(r_i_x_list)/switch_switch_subgraph.degree(i)
             switch_hubs=switch_hubs+1
           
elif len(nodes_max_degcen)==1:
    print("1  hub")
    for i in G1.nodes():
        if i in nodes_max_degcen:
         #print(i)
         rx_hub_overall=[]
         r_theta_hub_overall=[]
         if G1.nodes.data("classification")[i]=='oscillator':
          for t in range(0,T):
             r_i_theta_list=[]
             for j in G1.nodes():
                 if G1.nodes.data("classification")[j]=='oscillator' and i!=j:  
                         r_i_theta=abs(h[i][j]*np.exp(1j*model_state_t[t, j]-1j*model_state_t[t, i]))
                         r_i_theta_list.append(r_i_theta)
                         #print(r_i_theta)
                         #print(h[i][j])
             #print(len(r_i_theta_list))
             r_theta_hub=np.sum(r_i_theta_list)/oscillator_oscillator_subgraph.degree(i) ##G1.degree(i)
             #print(np.sum(r_i_theta_list))
             #print(oscillator_oscillator_subgraph.degree(i))
             r_theta_hub_overall.append(r_theta_hub)
         if G1.nodes.data("classification")[i]=='switch':
          for t in range(0,T):
             r_i_x_list=[]
             for j in G1.nodes():
                 if G1.nodes.data("classification")[j]=='switch':  
                     if model_state_t[t, j]>0:
                         l=1
                         r_i_x_list.append(h[i][j]*l)
                     else:
                        l=0
                        r_i_x_list.append(h[i][j]*l)
             #print(len(r_i_x_list))
             rx_hub_overall.append(np.sum(r_i_x_list)/switch_switch_subgraph.degree(i)) ##G1.degree(i)

##convert hub time series to dataframe an export as csv
import pandas as pd
rx_hub_switch_subgraph_df=pd.DataFrame(rx_hub_switch_subgraph)
rx_hub_switch_subgraph_df.to_csv('switch_subgraph_hub_order_param_time_series.csv')

r_theta_hub_oscillator_subgraph_df=pd.DataFrame(r_theta_hub_oscillator_subgraph)
r_theta_hub_oscillator_subgraph_df.to_csv('oscillator_subgraph_hub_order_param_time_series.csv')

rx_hub_overall_df=pd.DataFrame(rx_hub_overall)
rx_hub_overall_df.to_csv('switch_full_graph_hub_order_param_time_series.csv')

r_theta_hub_overall_df=pd.DataFrame(r_theta_hub_overall)
r_theta_hub_overall_df.to_csv('oscillator_full_graph_hub_order_param_time_series.csv')

##now plot the HUB order parameters 
##oscillator-oscillator subgraph
f12=plt.figure(figsize=(10,10))
plt.title('r_theta_oscillator_oscillator_subgraph_hub')
plt.plot(time, r_theta_hub_oscillator_subgraph, 'r',label='r_theta')
plt.legend()
f12.savefig("r_theta_oscillator_oscillator_subgraph_hub.pdf")
##switch-switch subgraph
f13=plt.figure(figsize=(10,10))
plt.title('r_x_switch_switch_subgraph_hub')
plt.plot(time, rx_hub_switch_subgraph, 'b',label="r_x")
plt.legend()
f13.savefig("r_theta_switch_switch_subgraph_hub.pdf")



##plot HUB order parameters for the overall graph
if np.shape(rx_hub_overall) !=(0,) and np.shape(r_theta_hub_overall) !=(0,):
    f15=plt.figure(figsize=(10,10))
    plt.title('order_params_full_graph_hub')
    plt.plot(time, rx_hub_overall, 'b', label="r_x")
    plt.plot(time, r_theta_hub_overall, 'r', label='r_theta')
    plt.legend()
    f15.savefig("order_params_full_graph_hub.pdf")

elif np.shape(rx_hub_overall) !=(0,) :
    f15=plt.figure(figsize=(10,10))
    plt.title('order_params_full_graph_hub')
    plt.plot(time, rx_hub_overall, 'b', label="r_x")
    plt.legend()
    f15.savefig("order_params_full_graph_hub.pdf")
    
elif np.shape(r_theta_hub_overall) !=(0,):
    f15=plt.figure(figsize=(10,10))
    plt.title('order_params_full_graph_hub')
    plt.plot(time, r_theta_hub_overall, 'r', label='r_theta')
    plt.legend()
    f15.savefig("order_params_full_graph_hub.pdf")


##now store the average equilibrium hub order parameters in a file
print('exporting hub order params')
LFR_hub_order_params=open("LFR_hub_order_params.tsv", "a")
LFR_hub_order_params.write("switch-switch subgraph hub complex <rx_eq>")
LFR_hub_order_params.write("\t")
LFR_hub_order_params.write("oscillator-oscillator hub subgraph complex <rtheta_eq>")
LFR_hub_order_params.write("\t")
LFR_hub_order_params.write("full graph hub hub complex <rx_eq>")
LFR_hub_order_params.write("\t")
LFR_hub_order_params.write("full graph hub rtheta complex <rx_eq>")
LFR_hub_order_params.write("\n")

if len(np.shape(rx_hub_switch_subgraph))>1:
    LFR_hub_order_params.write(str(np.mean(np.mean(rx_hub_switch_subgraph[:-1000], axis=1), axis=0)))
    LFR_hub_order_params.write("\t")
else:
    LFR_hub_order_params.write(str(np.mean(rx_hub_switch_subgraph[:-1000])))
    LFR_hub_order_params.write("\t")

if len(np.shape(r_theta_hub_oscillator_subgraph))>1:
    LFR_hub_order_params.write(str(np.mean(np.mean(r_theta_hub_oscillator_subgraph[:-1000], axis=1), axis=0)))
    LFR_hub_order_params.write("\t")
else:
    LFR_hub_order_params.write(str(np.mean(r_theta_hub_oscillator_subgraph[:-1000])))
    LFR_hub_order_params.write("\t")
    
if len(np.shape(rx_hub_overall))>1:
    LFR_hub_order_params.write(str(np.mean(np.mean(rx_hub_overall[:-1000], axis=1), axis=0)))
    LFR_hub_order_params.write("\t")
else:
    LFR_hub_order_params.write(str(np.mean(rx_hub_overall[:-1000])))
    LFR_hub_order_params.write("\t")

if len(np.shape(r_theta_hub_overall))>1:
    LFR_hub_order_params.write(str(np.mean(np.mean(r_theta_hub_overall[:-1000], axis=1), axis=0)))
    LFR_hub_order_params.write("\n")
else:
    LFR_hub_order_params.write(str(np.mean(r_theta_hub_overall[:-1000])))
    LFR_hub_order_params.write("\n")


LFR_p_K_r_theta.write(str(K_t_init))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(mean_omega))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(sigma_omega))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(K_max))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(Kx_theta))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(Kxx))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(eta))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(tau))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(set_Beta))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(time_avg_r_x_eq))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(time_avg_r_theta_eq))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(num_switches))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(num_oscillators))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(mean_x_init))
LFR_p_K_r_theta.write("\t")
LFR_p_K_r_theta.write(str(SD_x_init))
LFR_p_K_r_theta.write("\n")



##output order parameter and auto-correlation time_series
##output time-series of order params and correlation analyses
order_param_corr_LFR=open("LFR_order_param_and_correlation_timeseries_analysis_10_30_24.tsv", "a")
order_param_corr_LFR.write('model time units')
order_param_corr_LFR.write('\t')
order_param_corr_LFR.write('K(t)')
order_param_corr_LFR.write('\t')
order_param_corr_LFR.write("r_theta")
order_param_corr_LFR.write("\t")
order_param_corr_LFR.write("r_x")
order_param_corr_LFR.write("\t")
order_param_corr_LFR.write("r_theta_autocorr")
if max(r_x_list)> 0: 
    order_param_corr_LFR.write("\t")
    order_param_corr_LFR.write("rx_autocorr")
    order_param_corr_LFR.write("\t")
    order_param_corr_LFR.write("order_param_cross_corr")
order_param_corr_LFR.write("\n")


for t in range(T):
    order_param_corr_LFR.write(str(time[t]))
    order_param_corr_LFR.write("\t")
    order_param_corr_LFR.write(str(K_t[t]))
    order_param_corr_LFR.write('\t')
    order_param_corr_LFR.write(str(r_theta_list[t]))
    order_param_corr_LFR.write("\t")
    order_param_corr_LFR.write(str(r_x_list[t]))
    order_param_corr_LFR.write("\t")
    order_param_corr_LFR.write(str(r_theta_auto_corr[t]))
 
    if max(r_x_list)> 0: 
        order_param_corr_LFR.write("\t")
        order_param_corr_LFR.write(str(rx_auto_corr[t]))
        order_param_corr_LFR.write("\t")
        order_param_corr_LFR.write(str(order_param_crosscorr[t]))
    order_param_corr_LFR.write("\n")

LFR_p_K_r_theta.close()
LFR_p_K_r_theta_subgraph_log.close()
fourier_correlation_analysys_LFR.close()
LFR_hub_order_params.close()
order_param_corr_LFR.close()

##export the generated graph in multi-line adjacency format for further use
nx.generate_gexf(G1, "LFR_10_1_24.gexf")
print('done running script')
