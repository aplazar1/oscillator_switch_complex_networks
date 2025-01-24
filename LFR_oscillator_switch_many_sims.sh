#!/bin/bash
##create a new directory indicative of the condition run
##and run the code with the given conditions/constants
##Here we will run the Tarlor, Fertig, Restrepo model on LFR graphs varying both k and mean_x_init
initial_k=5
inc_k=5
end_k=40

initial_mean_x_init=-3
inc_mean_x_init=1
end=3

initial_rep=1
inc_rep=1
end_rep=4



for i in $(seq $initial_k $inc_k $end_k)
do
  for j in $(seq $initial_mean_x_init $inc_mean_x_init $end_mean_x_init)
  do
    for h in $(seq $initial_rep $inc_rep $end_rep)
    do 
     mkdir LFR_k${i}_mean_x_init_${j}_rep_${h}
     cp APR_f1_2_LFR_script_1_21_25_centrality_measures.py /home/aplazar1/data_gsteino1/aplazar1/oscillator_code/LFR_oscillator_switch/LFR_k${i}_mean_x_init_${j}_rep_${h}
     cd LFR_k${i}_mean_x_init_${j}_rep_${h}
     ##run the script and input the constants
     ##user input order: total_nodes, k, power_law1, power_law2, inter_community_edges, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, eta, tau, dt, secs, respectively, separated by commas
     ./APR_f1_2_LFR_script_1_21_25_centrality_measures.py <<< "200,${i},2.5,2.5,0.8,${h},4,100,10,3.2,0,${j},5,1,1.5,1,0.01,50"
     cd ../
    done
  done  
done
echo "finished sequence"
