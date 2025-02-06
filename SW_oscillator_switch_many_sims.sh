#!/bin/bash
##create a new directory indicative of the condition run
##and run the code with the given conditions/constants
##Here we will run the Tarlor, Fertig, Restrepo model on SW graphs varying both p and mean_x_init

#Note: This bash script is meant to be run in the directory where the python script f1_2_SW_script_11_9_24_centrality_measures_correct_index.py is located.
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
     mkdir SW_k${i}_mean_x_init_${j}_rep_${h}
     cp f1_2_SW_script_11_9_24_centrality_measures_correct_index.py ./SW_k${i}_mean_x_init_${j}_rep_${h}
     cd SW_k${i}_mean_x_init_${j}_rep_${h}
     ##run the script and input the constants
     ##user input order: total_nodes,k, p, rep, K_t_init, K_theta_max,Kx_theta, Kxx, mean_omega, mean_x_init, sigma_omega, SD_x_init, eta, tau, dt, secs
     ./f1_2_SW_script_11_9_24_centrality_measures_correct_index.py <<< "200,${i},0.03,${h},4,100,10,3.2,0,${j},5,1,1.5,1,0.01,50"
     cd ../
    done
  done  
done
echo "finished sequence"
