#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:57:58 2024

@author: aplazar1
"""

## Figure 1 Script for Complex Network Oscillator switch paper
##ER graphs

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
print("enter  simulation parameters: total_nodes, p, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, M, N, eta, tau, dt, secs, respectively, separated by commas")
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

 
plt.figure(figsize = (12, 12))
nx.draw_networkx(G1, pos1, node_color=Node_colors)        

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
ER_p_K_r_theta_subgraph_log = open("ER_p_k_r_theta_9_26_24_subgraph_parameters_log.txt", "a")
ER_p_K_r_theta_subgraph_log.write("rep-%01d"%(rep))
ER_p_K_r_theta_subgraph_log.write("\n")
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
ER_p_K_r_theta_subgraph_log.write("oscillator_oscillator_subgraph")
ER_p_K_r_theta_subgraph_log.write("\n")
ER_p_K_r_theta_subgraph_log.write("C1")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(C1))
ER_p_K_r_theta_subgraph_log.write("\n") 

if nx.is_connected(oscillator_oscillator_subgraph):
    l1=nx.average_shortest_path_length(oscillator_oscillator_subgraph)
    print("l1")
    print(l1)
    print("\n")
    ER_p_K_r_theta_subgraph_log.write("l1")
    ER_p_K_r_theta_subgraph_log.write(" ")
    ER_p_K_r_theta_subgraph_log.write(str(l1))
    ER_p_K_r_theta_subgraph_log.write("\n") 


##print max degree
degree_sequence = sorted((d for n, d in oscillator_oscillator_subgraph.degree()), reverse=True)
kmax1 = max(degree_sequence)
print("kmax1")
print(kmax1)
print("\n")
ER_p_K_r_theta_subgraph_log.write("kmax1")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(kmax1))
ER_p_K_r_theta_subgraph_log.write("\n") 

louvain_modularity1=nx.community.modularity(oscillator_oscillator_subgraph, nx.community.louvain_communities(oscillator_oscillator_subgraph))
print("louvain_modularity1=")
print(louvain_modularity1)
print("\n")
ER_p_K_r_theta_subgraph_log.write("louvain_modularity1")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(louvain_modularity1))
ER_p_K_r_theta_subgraph_log.write("\n") 

degree_list=[]
for node in oscillator_oscillator_subgraph.nodes():
    degree=oscillator_oscillator_subgraph.degree[node]
    degree_list.append(degree)
average_degree_of_graph1=np.sum(degree_list)/len(oscillator_oscillator_subgraph.nodes())
print("average_degree_of_graph1=") 
print(average_degree_of_graph1) 
print("\n")   
ER_p_K_r_theta_subgraph_log.write("average_degree_of_graph1")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph1))
ER_p_K_r_theta_subgraph_log.write("\n")     


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
ER_p_K_r_theta_subgraph_log.write("oscillator_switch_subgraph")
ER_p_K_r_theta_subgraph_log.write("\n") 
ER_p_K_r_theta_subgraph_log.write("C2")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(C2))
ER_p_K_r_theta_subgraph_log.write("\n")    


if nx.is_connected(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph)):
    l2=nx.average_shortest_path_length(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph))
    print("l2")
    print(l2)
    print("\n")
    ER_p_K_r_theta_subgraph_log.write("<l2>")
    ER_p_K_r_theta_subgraph_log.write(" ")
    ER_p_K_r_theta_subgraph_log.write(str(l2))
    ER_p_K_r_theta_subgraph_log.write("\n")    

##nx.is_connected(G)

##print max degree
degree_sequence = sorted((d for n, d in nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph).degree()), reverse=True)
kmax2 = max(degree_sequence)
print("kmax2")
print(kmax2)
print("\n")
ER_p_K_r_theta_subgraph_log.write("kmax2")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(kmax2))
ER_p_K_r_theta_subgraph_log.write("\n")    

louvain_modularity2=nx.community.modularity(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph), nx.community.louvain_communities(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph)))
print("louvain_modularity2")
print(louvain_modularity2)
print("\n")
ER_p_K_r_theta_subgraph_log.write("louvain_modularity2")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(louvain_modularity2))
ER_p_K_r_theta_subgraph_log.write("\n")    

degree_list=[]
for node in oscillator_switch_subgraph.nodes():
    degree=nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph).degree[node]
    degree_list.append(degree)
average_degree_of_graph2=np.sum(degree_list)/len(nx.intersection(oscillator_switch_subgraph, switch_oscillator_subgraph).nodes())
print("average_degree_of_graph2")  
print(average_degree_of_graph2)  
print("\n")
ER_p_K_r_theta_subgraph_log.write("average_degree_of_graph2")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph2))
ER_p_K_r_theta_subgraph_log.write("\n")  

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
ER_p_K_r_theta_subgraph_log.write("switch_switch_subgraph")
ER_p_K_r_theta_subgraph_log.write("\n")
ER_p_K_r_theta_subgraph_log.write("C3")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(C3))
ER_p_K_r_theta_subgraph_log.write("\n") 

if nx.is_connected(switch_switch_subgraph):
    l3=nx.average_shortest_path_length(switch_switch_subgraph)
    print("l3")
    print(l3)
    print("\n")
    ER_p_K_r_theta_subgraph_log.write("<l3>")
    ER_p_K_r_theta_subgraph_log.write(" ")
    ER_p_K_r_theta_subgraph_log.write(str(l3))
    ER_p_K_r_theta_subgraph_log.write("\n") 


##print max degree
degree_sequence = sorted((d for n, d in switch_switch_subgraph.degree()), reverse=True)
kmax3 = max(degree_sequence)
print("kmax3")
print(kmax3)
print("\n")
ER_p_K_r_theta_subgraph_log.write("kmax3")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(kmax3))
ER_p_K_r_theta_subgraph_log.write("\n") 

louvain_modularity3=nx.community.modularity(switch_switch_subgraph, nx.community.louvain_communities(switch_switch_subgraph))
print("louvain_modularity3")
print(louvain_modularity3)
print("\n")
ER_p_K_r_theta_subgraph_log.write("louvain_modularity3")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(louvain_modularity3))
ER_p_K_r_theta_subgraph_log.write("\n") 

degree_list=[]
for node in switch_switch_subgraph.nodes():
    degree=switch_switch_subgraph.degree[node]
    degree_list.append(degree)
average_degree_of_graph3=np.sum(degree_list)/len(switch_switch_subgraph.nodes())
print("average_degree_of_graph3")  
print(average_degree_of_graph3)  
print("\n")
ER_p_K_r_theta_subgraph_log.write("average_degree_of_graph3")
ER_p_K_r_theta_subgraph_log.write(" ")
ER_p_K_r_theta_subgraph_log.write(str(average_degree_of_graph3))
ER_p_K_r_theta_subgraph_log.write("\n") 


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
                        num_switch_oscillator_neighborgs=num_switch_oscillator_neighborgs+1
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

##plot the order parameters
##calculate the real_time_steps
time=np.linspace(0,secs,int(secs/dt))
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
f9.savefig("K_t.pdf" ) 

##Compute graph properties for the overall network
##compute the average clustering of the network
print("overall graph descriptive indices")
print("C")
C=nx.average_clustering(G1)
print(C)  
print("\n")   

clustering_per_node=nx.clustering(G1, list(G1.nodes()))

if nx.is_connected(G1):
    print("<l>")
    l=nx.average_shortest_path_length(G1)
    print(l)
    print("\n") 

##print max degree
degree_sequence = sorted((d for n, d in G1.degree()), reverse=True)
kmax = max(degree_sequence)
print("kmax")
print(kmax)
print("\n") 

louvain_modularity=nx.community.modularity(G1, nx.community.louvain_communities(G1))
print("louvain_modularity")
print(louvain_modularity)
print("\n") 

degree_list=[]
for node in G1.nodes():
    degree=G1.degree[node]
    degree_list.append(degree)
average_degree_of_graph=np.sum(degree_list)/len(G1.nodes())
print("<k> of graph")
print(average_degree_of_graph)
print("\n") 

##export simulation params and calculated model and graph proeprties to a tsv file
set_Beta=0


ER_p_K_r_theta = open("ER_p_k_r_theta_9_26_24.tsv", "a")
ER_p_K_r_theta.write("rep-%01d"%(rep))
ER_p_K_r_theta.write("\n")
ER_p_K_r_theta.write("p")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("<k>")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("G kmax")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("C")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("<l>")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("louvain_modularity")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("K_t_init")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("omega_0")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("delta_scale")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("K_max")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("Kx_theta")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("Kxx")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("eta")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("tau")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("beta_l")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("<r_x_eq>")
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write("<r_theta_eq>")
ER_p_K_r_theta.write("\n")


ER_p_K_r_theta.write(str(p))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(average_degree_of_graph))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(kmax))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(C))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(l))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(louvain_modularity))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(K_t_init))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(omega_0))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(delta_scale))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(K_max))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(Kx_theta))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(Kxx))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(eta))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(tau))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(set_Beta))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(time_avg_r_x_eq))
ER_p_K_r_theta.write("\t")
ER_p_K_r_theta.write(str(time_avg_r_theta_eq))
ER_p_K_r_theta.write("\n")

ER_p_K_r_theta.close()
ER_p_K_r_theta_subgraph_log.close()

##export the generated graph in multi-line adjacency format for further use
nx.write_gexf(G1, "ER_9_27_24.gexf")
