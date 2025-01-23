# Paper1_oscillator_switch_theory
Repo containing code for runing simulations, data analysis and figure generation for my first post-doc theory paper of the Taylor-Fertig-Restrepo oscillator switch model adapted to complex networks.

Steps to reproduce data: 
1) Run the f1_2 scripts to run the TFR model on Erdos-Renyi (ER), Small-world (SW),  and Lancichinetti–Fortunato–Radicchi graphs (LFR).
2) Run the collect_metadata_for_large_scale_analysis.py script to condense metadata from many separate simulation csv files and group into a single csv file.
3) Run the figure_gen script to plot data (from many aggregate simulations).
