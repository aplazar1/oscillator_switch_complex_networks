#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 18:31:14 2024

@author: aplazar1
"""


## Figure 1 Script for Complex Network Oscillator switch paper
##SW graphs

##Note: Here I corrected the code for mistakes found in normalization of coupling constants
##normalization shoudl be done according to the  degree of node, taking into account the 
#number of neighborgs of each type
import numpy as np
import math as mt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
plt.style.use("ggplot")
import scipy.stats as scp
import networkx as nx


##testing params
#200,0.3,1,4,100,10,3.2,0,-100,5,1,1.5,1,0.01,50

##model parameters from user input
print('here we sample X init and omega init from a normal distribution')
print('input the model parameters below')
print("enter  simulation parameters: total_nodes, p, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, eta, tau, dt, secs, respectively, separated by commas")
total_nodes, p, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, eta, tau, dt, secs =input().split(',')

#convert input to numbers
total_nodes=int(total_nodes)
p=float(p)
rep=int(rep)
K_t_init=float(K_t_init)
K_theta_max=float(K_theta_max)
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


##Generate the ER random graphs
##Note: for this function p is the probability for edge creation. In the Gomez-Gardenes paper, this corresponds to 
##alpha, the probability that each of the m edeges is connected to a randomly chosen node.

##generate 1 Erdos Reny random graph
G1=nx.erdos_renyi_graph(n=total_nodes, p=p) ##use a single graph for all the types of nodes

##sample from a uniform distribution to assign node type
import numpy.random as rn
assignment2=np.random.choice(total_nodes-1, total_nodes)

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

##plot the graph
###plot the graph
pos1 = nx.circular_layout(G1)

 
figure0=plt.figure(figsize = (12, 12))
nx.draw_networkx(G1, pos1, node_color=Node_colors) 
figure0.savefig("full_SW_graph_node_colors.pdf" )          ##10/31  

##initial oscillator-oscillator coupling k(0)
K_t_init=4


omega_0=5
delta_scale=1
# Define the natural frequencies of the oscillators
omega_init =scp.cauchy.rvs(loc=omega_0, scale=delta_scale, size=num_oscillators)

# Define the coupling strength
K=100
K_max=K ##K_max is the oscillator-oscillator critical coupling constant
Kx_theta=10
Kxx=3.2

#threshold for switch activation
eta=1.5

##time for oscillator-oscillator coupling adaptation
#tau=np.max([1, omega_0])
##for Figure 2 set, otherwise, tau=np.max([1, omega_0])
tau=1

##phase lag beta
Beta=np.zeros(num_switches)

# Define the initial phases of the oscillators
##asume that the oscillators are uniformly distributed, and switches are normally distributed.
theta_init = np.random.uniform(0, 2*np.pi, num_oscillators)

##define initial switch state
##Asumme switches were sampled from unimodal, random distribution

x_t_init = np.random.normal(-100, 1, num_switches)

# Define the time step and the number of time steps
dt = 0.01
secs=50  ###number of "real time" seconds to simulate
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
SW_p_K_r_theta_subgraph_log = open("SW_p_fixed_increasing_k_rtheta_10_01_24_subgraph_parameters_log.txt", "a")
SW_p_K_r_theta_subgraph_log.write("rep-%01d"%(rep))
SW_p_K_r_theta_subgraph_log.write("\n")
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
SW_p_K_r_theta_subgraph_log.write("oscillator_oscillator_subgraph")
SW_p_K_r_theta_subgraph_log.write("\n")
SW_p_K_r_theta_subgraph_log.write("C1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(C1))
SW_p_K_r_theta_subgraph_log.write("\n") 

if nx.is_connected(oscillator_oscillator_subgraph):
    l1=nx.average_shortest_path_length(oscillator_oscillator_subgraph)
    ##get the node(s) with the highest Eigenvector  centrality
    EigCen1=nx.eigenvector_centrality(oscillator_oscillator_subgraph, max_iter=200)
    EigCen_sorted1=dict(sorted(EigCen1.items(), key=lambda item: item[1],reverse=True))
    nodes_max_EigCen1=[k for k, v in EigCen_sorted1.items() if v == max(EigCen_sorted1.values())]
    Max_EigCen1=max(EigCen_sorted1.values())
    print("l1")
    print(l1)
    print("\n")
    print("Nodes with max Eigenvector_centrality1")
    print(nodes_max_EigCen1)
    print("\n")
    print("Max Eigenvector_centrality1")
    print(Max_EigCen1)
    print("\n")
    SW_p_K_r_theta_subgraph_log.write("l1")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(l1))
    SW_p_K_r_theta_subgraph_log.write("\n") 
    SW_p_K_r_theta_subgraph_log.write("Nodes with max Eigenvector_centrality1")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(nodes_max_EigCen1))
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Max Eigenvector_centrality2")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(Max_EigCen1))
    SW_p_K_r_theta_subgraph_log.write("\n")
    
else:
    SW_p_K_r_theta_subgraph_log.write("<l1>")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Nodes with max Eigenvector_centrality1")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Max Eigenvector_centrality1")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
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

SW_p_K_r_theta_subgraph_log.write("kmax1 nodes")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(kmax1_nodes))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("kmax2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(kmax1))
SW_p_K_r_theta_subgraph_log.write("\n")   


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

SW_p_K_r_theta_subgraph_log.write("nodes_with_max_degree_centrality1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(nodes_max_degcen1))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("Max_degree_centrality1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(Max_degcen1))
SW_p_K_r_theta_subgraph_log.write("\n") 

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

SW_p_K_r_theta_subgraph_log.write("nodes_with_max_betweenness_centrality1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(nodes_max_betCen1))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("Max_betweennes_centrality1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(Max_betCen1))
SW_p_K_r_theta_subgraph_log.write("\n") 

louvain_modularity1=nx.community.modularity(oscillator_oscillator_subgraph, nx.community.louvain_communities(oscillator_oscillator_subgraph))
print("louvain_modularity1=")
print(louvain_modularity1)
print("\n")
SW_p_K_r_theta_subgraph_log.write("louvain_modularity1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(louvain_modularity1))
SW_p_K_r_theta_subgraph_log.write("\n") 

degree_list=[]
for node in oscillator_oscillator_subgraph.nodes():
    degree=oscillator_oscillator_subgraph.degree[node]
    degree_list.append(degree)
average_degree_of_graph1=np.sum(degree_list)/len(oscillator_oscillator_subgraph.nodes())
print("average_degree_of_graph1=") 
print(average_degree_of_graph1) 
print("\n")   
SW_p_K_r_theta_subgraph_log.write("average_degree_of_graph1")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph1))
SW_p_K_r_theta_subgraph_log.write("\n")     


### 2 draw the mixed node connection oscillator-switch subgraph with a circular layout
print("oscillator_switch_mixed_node_subgraph number of nodes and edges")
print(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
pos3 = nx.circular_layout(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
print("\n")

 
figure2=plt.figure(figsize = (12, 12))
plt.title('oscillator-switch mixed node subgraph')
nx.draw_networkx(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph), pos3)
figure2.savefig("oscillator-switch subgraph.pdf" )

C2=nx.average_clustering(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
print("C2")   
print(C2)   
print("\n")


SW_p_K_r_theta_subgraph_log.write("oscillator_switch_subgraph")
SW_p_K_r_theta_subgraph_log.write("\n") 
SW_p_K_r_theta_subgraph_log.write("C2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(C2))
SW_p_K_r_theta_subgraph_log.write("\n")    


if nx.is_connected(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph)):
    l2=nx.average_shortest_path_length(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
    ##get the node(s) with the highest Eigenvector  centrality
    EigCen2=nx.eigenvector_centrality(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph), max_iter=200)
    EigCen_sorted2=dict(sorted(EigCen2.items(), key=lambda item: item[1],reverse=True))
    nodes_max_EigCen2=[k for k, v in EigCen_sorted2.items() if v == max(EigCen_sorted2.values())]
    Max_EigCen2=max(EigCen_sorted2.values())
    print("l2")
    print(l2)
    print("\n")
    print("Nodes with max Eigenvector_centrality2")
    print(nodes_max_EigCen2)
    print("\n")
    print("Max Eigenvector_centrality2")
    print(Max_EigCen2)
    print("\n")
    
    SW_p_K_r_theta_subgraph_log.write("<l2>")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(l2))
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Nodes with max Eigenvector_centrality2")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(nodes_max_EigCen2))
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Max Eigenvector_centrality2")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(Max_EigCen2))
    SW_p_K_r_theta_subgraph_log.write("\n")
    
else:
    SW_p_K_r_theta_subgraph_log.write("<l2>")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Nodes with max Eigenvector_centrality2")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Max Eigenvector_centrality2")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    print('N/A')    



#extract the kmax (max degre) and the node(s) with the largest kmax
degree_view2=dict(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph).degree())
degree_sequence2=dict(sorted(degree_view2.items(), key=lambda item: item[1],reverse=True))
kmax2_nodes=[k for k, v in degree_sequence2.items() if v == max(degree_sequence2.values())]
kmax2 = max(degree_sequence2.values())
print("kmax2_nodes")
print(kmax2_nodes)
print("kmax2")
print(kmax2)

SW_p_K_r_theta_subgraph_log.write("kmax2 nodes")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(kmax2_nodes))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("kmax2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(kmax2))
SW_p_K_r_theta_subgraph_log.write("\n")   


##get the max degree dentrality
#calculate degree centrality
degCent2 = nx.degree_centrality(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
#Descending order sorting centrality
degCent_sorted2=dict(sorted(degCent2.items(), key=lambda item: item[1],reverse=True))
##get the node with the highest degree centrality
nodes_max_degcen2=[k for k, v in degCent_sorted2.items() if v == max(degCent_sorted2.values())]
Max_degcen2=max(degCent_sorted2.values())
print("nodes_with_max_degree_centrality2")
print(nodes_max_degcen2)
print("Max_degree_centrality2")
print(Max_degcen2)

SW_p_K_r_theta_subgraph_log.write("nodes_with_max_degree_centrality2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(nodes_max_degcen2))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("Max_degree_centrality2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(Max_degcen2))
SW_p_K_r_theta_subgraph_log.write("\n") 

##look at the betweennes centrality
##obtain the node with the max betweennes centrality as above
betCent2=nx.betweenness_centrality(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
betCent_sorted2=dict(sorted(betCent2.items(), key=lambda item: item[1],reverse=True))
nodes_max_betCen2 = [k for k, v in betCent_sorted2.items() if v == max(betCent_sorted2.values())]
Max_betCen2=max(betCent_sorted2.values())
print("nodes_with_max_betweenness_centrality2")
print(nodes_max_betCen2)
print("Max_betweennes_centrality2")
print(Max_betCen2)

SW_p_K_r_theta_subgraph_log.write("nodes_with_max_betweenness_centrality2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(nodes_max_betCen2))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("Max_betweennes_centrality2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(Max_betCen2))
SW_p_K_r_theta_subgraph_log.write("\n") 


louvain_modularity2=nx.community.modularity(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph), nx.community.louvain_communities(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph)))
print("louvain_modularity2")
print(louvain_modularity2)
print("\n")
SW_p_K_r_theta_subgraph_log.write("louvain_modularity2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(louvain_modularity2))
SW_p_K_r_theta_subgraph_log.write("\n")    

degree_list=[]
for node in oscillator_switch_subgraph.nodes():
    degree=nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph).degree[node]
    degree_list.append(degree)
average_degree_of_graph2=np.sum(degree_list)/len(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph).nodes())
print("average_degree_of_graph2")  
print(average_degree_of_graph2)  
print("\n")
SW_p_K_r_theta_subgraph_log.write("average_degree_of_graph2")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph2))
SW_p_K_r_theta_subgraph_log.write("\n")  

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
SW_p_K_r_theta_subgraph_log.write("switch_switch_subgraph")
SW_p_K_r_theta_subgraph_log.write("\n")
SW_p_K_r_theta_subgraph_log.write("C3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(C3))
SW_p_K_r_theta_subgraph_log.write("\n") 

if nx.is_connected(switch_switch_subgraph):
    l3=nx.average_shortest_path_length(switch_switch_subgraph)
    ##get the node with the highest Eigenvector  centrality
    EigCen3=nx.eigenvector_centrality(switch_switch_subgraph, max_iter=200)
    EigCen_sorted3=dict(sorted(EigCen3.items(), key=lambda item: item[1],reverse=True))
    nodes_max_EigCen3=[k for k, v in EigCen_sorted3.items() if v == max(EigCen_sorted3.values())]
    Max_EigCen3=max(EigCen_sorted3.values())
    print("l3")
    print(l3)
    print("\n")
    print("Nodes with max Eigenvector_centrality3")
    print(nodes_max_EigCen3)
    print("\n")
    print("Max Eigenvector_centrality3")
    print(Max_EigCen3)
    print("\n")
    SW_p_K_r_theta_subgraph_log.write("<l3>")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(l3))
    SW_p_K_r_theta_subgraph_log.write("\n") 
    SW_p_K_r_theta_subgraph_log.write("Nodes with max Eigenvector_centrality3")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(nodes_max_EigCen3))
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Max Eigenvector_centrality3")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write(str(Max_EigCen3))
    SW_p_K_r_theta_subgraph_log.write("\n")
else:
    SW_p_K_r_theta_subgraph_log.write("<l3>")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Nodes with max Eigenvector_centrality3")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
    SW_p_K_r_theta_subgraph_log.write("Max Eigenvector_centrality3")
    SW_p_K_r_theta_subgraph_log.write(" ")
    SW_p_K_r_theta_subgraph_log.write("N/A")
    SW_p_K_r_theta_subgraph_log.write("\n")
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

SW_p_K_r_theta_subgraph_log.write("kmax3 nodes")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(kmax3_nodes))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("kmax3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(kmax3))
SW_p_K_r_theta_subgraph_log.write("\n")   


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

SW_p_K_r_theta_subgraph_log.write("nodes_with_max_degree_centrality3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(nodes_max_degcen3))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("Max_degree_centrality3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(Max_degcen3))
SW_p_K_r_theta_subgraph_log.write("\n") 

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

SW_p_K_r_theta_subgraph_log.write("nodes_with_max_betweenness_centrality3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(nodes_max_betCen3))
SW_p_K_r_theta_subgraph_log.write("\n")   

SW_p_K_r_theta_subgraph_log.write("Max_betweennes_centrality3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(Max_betCen3))
SW_p_K_r_theta_subgraph_log.write("\n") 

louvain_modularity3=nx.community.modularity(switch_switch_subgraph, nx.community.louvain_communities(switch_switch_subgraph))
print("louvain_modularity3")
print(louvain_modularity3)
print("\n")
SW_p_K_r_theta_subgraph_log.write("louvain_modularity3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(louvain_modularity3))
SW_p_K_r_theta_subgraph_log.write("\n") 

degree_list=[]
for node in switch_switch_subgraph.nodes():
    degree=switch_switch_subgraph.degree[node]
    degree_list.append(degree)
average_degree_of_graph3=np.sum(degree_list)/len(switch_switch_subgraph.nodes())
print("average_degree_of_graph3")  
print(average_degree_of_graph3)  
print("\n")
SW_p_K_r_theta_subgraph_log.write("average_degree_of_graph3")
SW_p_K_r_theta_subgraph_log.write(" ")
SW_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph3))
SW_p_K_r_theta_subgraph_log.write("\n") 


##Obtain the overall adjacency matrix to ensure correct simulation

h=nx.to_numpy_array(G1)

# Define the array to store results at each time step
theta_t = np.zeros((T, num_oscillators))
theta_t[0] = theta_init
K_t = np.zeros((T))
K_t[0]=K_t_init
x_t = np.zeros((T, num_switches))
x_t[0] = x_t_init
x_t_discrete=np.zeros((T, num_switches))
#r_link_t=[]

# Run the simulation
for t in range(1, T):
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
                        ##oscillator rate of phase change with time
                        oscillator_neighborg_contribution=(h[i][n]*np.sin(theta_t[t-1] - theta_t[t-1, oscillator_num]))/num_oscillator_oscillator_neighborgs
                        oscillator_oscillator_neighborg_contributions.append(oscillator_neighborg_contribution)    
                theta_t[t, oscillator_num] = (theta_t[t-1, oscillator_num] + dt*(omega_init[oscillator_num] + ((K_t[t-1])*np.sum(oscillator_oscillator_neighborg_contributions))))%(2*np.pi)
            elif len(neighbors)==0: 
                theta_t[t, oscillator_num] = (theta_t[t-1, oscillator_num] + dt*(omega_init[oscillator_num]))%(2*np.pi)
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
                    if G1.nodes.data("classification")[n]=='switch': 
                        #print(i)
                        #print(n)
                        switch_switch_influence=(h[i][n]*x_t_discrete[t-1, switch_num])/num_switch_switch_neighborgs
                        switch_neighborg_infl.append(switch_switch_influence)
                    elif G1.nodes.data("classification")[n]=='oscillator':
                        oscillator_switch_influence=(h[i][n]*np.sin(theta_t[t-1] - Beta[switch_num]))/num_switch_oscillator_neighborgs
                        oscillator_neighborg_infl.append(oscillator_switch_influence)
                        switch_oscillator_influence=(h[i][n]*x_t_discrete[t-1, switch_num])/num_switch_oscillator_neighborgs
                        switch_oscillator_K_infl.append(switch_oscillator_influence)
                x_t[t,switch_num] = x_t[t-1, switch_num] + dt*(-x_t[t-1,switch_num] - eta +  (Kxx)*np.sum(switch_neighborg_infl) +  (Kx_theta)*np.sum(oscillator_neighborg_infl))                                                                                        
            elif len(neighbors)==0:
                x_t[t,switch_num] = x_t[t-1, switch_num] + dt*(-x_t[t-1,switch_num] - eta )
            ##clasify states as discrete
            if x_t[t,switch_num] <= 0:
                x_t_discrete[t,switch_num]=0
            elif x_t[t,switch_num]>0:
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
#plt.ylim(0, 2*np.pi)
#plt.show()
f4.savefig("phase.pdf" )

##plot switch state
f5,ax=plt.subplots(figsize=(10,10))



plt.plot(x_t_discrete)
plt.xlabel('model time steps')
plt.ylabel('state')
f5.savefig("switch state.pdf" )
#plt.ylim(0, 2*np.pi)
#plt.show()

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
r_omega_list=[]

for t in range(T):
        #print(t)
        ##for this order parameter, we sum over all oscillator frequencies at a given time
        #print(np.sum(np.exp(1j*theta_t_pie[t])))
        #print(np.exp(1j*psi[t]))
        r_theta=abs((1/num_oscillators)*np.sum(np.exp(1j*theta_t_pie[t]-1j*psi[t])))
        #print(r_theta)
        r_theta_list.append(r_theta)
        #if r_theta < 0:
        #    print(t)
        #    print(theta_t_pie[t])
        #    print(psi[t])
        #    print(r_theta)
        ##note that this is a mean-field switch order parameter
        r_x=(np.sum(x_t_discrete[t])/num_switches)
        r_x_list.append(r_x)


#perform autocorrelation  and cross correlation analysis of order parameters, specifically the switch order param
rx_auto_corr=np.correlate(r_x_list, r_x_list, mode='same')/(max(np.correlate(r_x_list, r_x_list, mode='same')))
r_theta_auto_corr=np.correlate(r_theta_list, r_theta_list, mode='same')/(max(np.correlate(r_theta_list, r_theta_list, mode='same')))
order_param_crosscorr=np.correlate(r_x_list, r_theta_list, mode='same')/(max(np.correlate(r_x_list, r_theta_list, mode='same')))



#perform fourier analysis of the switch order parameter
from scipy.fft import fft
rx_fft=fft(r_x_list)
r_theta_fft=fft(r_theta_list)

##calculate the real_time_steps
time=np.linspace(0,secs,int(secs/dt))

##output time-series of order parameters, their autocorrelations, and cross correlations
fourier_correlation_analysys_SW=open("SW_fourier_correlation_analysis_10_30_24.tsv", "a")
fourier_correlation_analysys_SW.write("model time (s)")
fourier_correlation_analysys_SW.write('\t')
fourier_correlation_analysys_SW.write("r_theta")
fourier_correlation_analysys_SW.write("\t")
fourier_correlation_analysys_SW.write("r_x")
fourier_correlation_analysys_SW.write("\t")
fourier_correlation_analysys_SW.write("rx_auto_corr")
fourier_correlation_analysys_SW.write("\t")
fourier_correlation_analysys_SW.write("r_theta_auto_orr")
fourier_correlation_analysys_SW.write("\t")
fourier_correlation_analysys_SW.write("cross_corr")
fourier_correlation_analysys_SW.write("\t")
fourier_correlation_analysys_SW.write("rx_fft")
fourier_correlation_analysys_SW.write("\t")
fourier_correlation_analysys_SW.write("r_theta_fft")
fourier_correlation_analysys_SW.write("\n")

for t in range(T):
    fourier_correlation_analysys_SW.write(str(time[t]))
    fourier_correlation_analysys_SW.write('\t')
    fourier_correlation_analysys_SW.write(str(r_theta_list[t]))
    fourier_correlation_analysys_SW.write("\t")
    fourier_correlation_analysys_SW.write(str(r_x_list[t]))
    fourier_correlation_analysys_SW.write("\t")
    fourier_correlation_analysys_SW.write(str(rx_auto_corr[t]))
    fourier_correlation_analysys_SW.write("\t")
    fourier_correlation_analysys_SW.write(str(r_theta_auto_corr[t]))
    fourier_correlation_analysys_SW.write("\t")
    fourier_correlation_analysys_SW.write(str(order_param_crosscorr[t]))
    fourier_correlation_analysys_SW.write("\t")
    fourier_correlation_analysys_SW.write(str(rx_fft[t]))
    fourier_correlation_analysys_SW.write("\t")
    fourier_correlation_analysys_SW.write(str(r_theta_fft[t]))
    fourier_correlation_analysys_SW.write("\n")






##plot the order parameters
f7,ax=plt.subplots(figsize=(10,10))
plt.plot(time,r_theta_list,'r', label='r_theta')
plt.plot(time,r_x_list,'b', label='r_x')
#plt.plot(r_link_t, 'm', label='r_link')
plt.legend()

plt.title('order parameters')
plt.xlabel('t(secs)')
f7.savefig("order params.pdf" )
#plt.ylabel('r theta')
#plt.show()  

##calculate and export the r_theta_time_average_over_the_last 100 time steps

#print(time[-100:])
time_avg_r_theta_eq=np.mean(r_theta_list[-100:], axis=0)
time_avg_r_x_eq=np.mean(r_x_list[-100:], axis=0)
print(time_avg_r_theta_eq)
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

#plt.show() 

##plot the switch state vs time 
import matplotlib.pyplot as plt
f9,ax=plt.subplots(figsize=(10,10))
plt.plot(time,np.mean(x_t, axis=1),'m', label='<x_t>')
plt.plot(time,np.std(x_t, axis=1),'k', label='SD x_t')
plt.legend()


plt.xlabel('t(secs)')
plt.ylabel(' average switch state <x(t)>')
#plt.show()   
f9.savefig("average_x_t.pdf" ) 

##plot the signal correlations
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

##plot the fourier transforms of the order parameters
f11, (ax1, ax2)=plt.subplots(1, 2, figsize=(10,10))
f10.suptitle('FFT decomposition of order parameters')
ax1.plot(rx_fft,'b', label='r_x fft')
ax2.plot(r_theta_fft,'r', label='r_theta fft')

ax1.legend()
ax2.legend()

ax2.set(xlabel='frequency domain')
ax1.set(xlabel='frequency domain')
ax1.set(ylabel='amplitude domain')
f11.savefig("order_param_fft.pdf" ) 


##Compute graph properties for the overall network
SW_p_K_r_theta = open("SW_p_fixed_increasing_k_rtheta_10_28_24.tsv", "a")
SW_p_K_r_theta.write("rep-%01d"%(rep))
SW_p_K_r_theta.write("\n")
SW_p_K_r_theta.write("p")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("C")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("<l>")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("nodes with max Eigenvector Centrality")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("Max Eigenvector Centrality")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("G kmax node")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("G kmax")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("nodes_with_max_degree_centrality")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("max_degree_centrality")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("nodes_with_max_betweeness_centrality")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("max_betweeness_centrality")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("louvain modularity")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("G <k>")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("K_t_init")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("omega_0")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("delta_scale")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("K_max")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("Kx_theta")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("Kxx")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("eta")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("tau")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("beta_l")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("<r_x_eq>")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("<r_theta_eq>")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("num_switches")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("num_oscillators")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("mean_x_init")
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write("SD_x_init")
SW_p_K_r_theta.write("\n")

SW_p_K_r_theta.write(str(p))
SW_p_K_r_theta.write("\t")

##compute the average clustering of the network
print("overall graph descriptive indices")
print("C")
C=nx.average_clustering(G1)
print(C)  
print("\n")
SW_p_K_r_theta.write(str(C))
SW_p_K_r_theta.write("\t")  

 

clustering_per_node=nx.clustering(G1, list(G1.nodes()))

if nx.is_connected(G1):
    print("<l>")
    l=nx.average_shortest_path_length(G1)
    print(l)
    print("\n") 
    SW_p_K_r_theta.write(str(l))
    SW_p_K_r_theta.write("\t")

    EigCen=nx.eigenvector_centrality(G1, max_iter=200)
    EigCen_sorted=dict(sorted(EigCen.items(), key=lambda item: item[1],reverse=True))
    nodes_max_EigCen=[k for k, v in EigCen_sorted.items() if v == max(EigCen_sorted.values())]
    Max_EigCen=max(EigCen_sorted.values())
    print("nodes with max Eigenvector centrality")
    print(nodes_max_EigCen)
    print("\n")
    print("max Eigenvector centrality")
    print(Max_EigCen)
    print("\n")
    SW_p_K_r_theta.write(str(nodes_max_EigCen))
    SW_p_K_r_theta.write("\t")
    SW_p_K_r_theta.write(str(Max_EigCen))
    SW_p_K_r_theta.write("\t")
    
else:
    SW_p_K_r_theta.write("N/A")
    SW_p_K_r_theta.write("\t")
    SW_p_K_r_theta.write("N/A")
    SW_p_K_r_theta.write("\t")
    SW_p_K_r_theta.write("N/A")
    SW_p_K_r_theta.write("\t")
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


SW_p_K_r_theta.write(str(kmax_nodes))
SW_p_K_r_theta.write("\t")  
SW_p_K_r_theta.write(str(kmax))
SW_p_K_r_theta.write("\t")   

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


SW_p_K_r_theta.write(str(nodes_max_degcen))
SW_p_K_r_theta.write("\t")   


SW_p_K_r_theta.write(str(Max_degcen))
SW_p_K_r_theta.write("\t") 

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


SW_p_K_r_theta.write(str(nodes_max_betCen))
SW_p_K_r_theta.write("\t")   


SW_p_K_r_theta.write(str(Max_betCen))
SW_p_K_r_theta.write("\t") 


louvain_modularity=nx.community.modularity(G1, nx.community.louvain_communities(G1))
print("louvain_modularity")
print(louvain_modularity)
print("\n") 
SW_p_K_r_theta.write(str(louvain_modularity))
SW_p_K_r_theta.write("\t")  

degree_list=[]
for node in G1.nodes():
    degree=G1.degree[node]
    degree_list.append(degree)
average_degree_of_graph=np.sum(degree_list)/len(G1.nodes())
print("<k> of graph")
print(average_degree_of_graph)
print("\n") 
SW_p_K_r_theta.write(str(average_degree_of_graph))
SW_p_K_r_theta.write("\t")  

##export simulation params and calculated model and graph proeprties to a tsv file
set_Beta=0


SW_p_K_r_theta.write(str(K_t_init))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(omega_0))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(delta_scale))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(K_max))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(Kx_theta))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(Kxx))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(eta))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(tau))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(set_Beta))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(time_avg_r_x_eq))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(time_avg_r_theta_eq))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(num_switches))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(num_oscillators))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(mean_x_init))
SW_p_K_r_theta.write("\t")
SW_p_K_r_theta.write(str(SD_x_init))
SW_p_K_r_theta.write("\n")

SW_p_K_r_theta.close()
SW_p_K_r_theta_subgraph_log.close()
fourier_correlation_analysys_SW.close()

##export the generated graph in multi-line adjacency format for further use
nx.write_gexf(G1, "SW_10_1_24.gexf")
