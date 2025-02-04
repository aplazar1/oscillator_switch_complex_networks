# Paper1_oscillator_switch_theory
Repo containing code for runing simulations, data analysis and figure generation for my first post-doc theory paper of the Taylor-Fertig-Restrepo oscillator switch model adapted to complex networks.

Prepare the environment, e.g for running in a Docker container interactively:
```
docker run -it -v ./:/ost --workdir /ost continuumio/miniconda3 /bin/bash
conda create -n python_env python=3.12.7 numpy matplotlib scipy networkx
conda activate python_env
```

Main script to run simulations: The file "General_script_input_adjacency_1_28_25_APR.py" solves the TFR model using an adjacency matrix input as csv file. It also takes as input model coupling constants and initial conditions to initialize the model. In addition, the script also asks for the user to input the ratio of oscillators to switches. This is a non-trivial parameter and will affect the model output. This ratio of oscillators to switches would need to be curated from prior biological knowledge.

Steps to reproduce paper data: 
1) Run the f1_2 (figure 1,2) scripts to run the TFR model on Erdos-Renyi (ER), Small-world (SW),  and Lancichinetti–Fortunato–Radicchi graphs (LFR).
2) Run the collect_metadata_for_large_scale_analysis.py script to condense metadata from many separate simulation csv files and group into a single csv file.
3) Run the figure_gen script to plot data (from many aggregate simulations).

Note: The f3 (figure 3) B_lag scripts implement differently uniformly sampled phase-lags between pairs of connected oscillators and switches, while the B_set_lag scripts implement the same phase-lag for all pairs of connected oscillators and switches (selected by sampling once from a uniform distribution).

*The Many_sims.sh shell scripts run many iterations of the python scripts (above) solving an euler scheme of the TFR model while varying over graph parameters and model initialization parameters. This is for running many simulations on a server to reproduce paper figures 1,2,3. It is a parameter scan.



Requirements to run scripts: python 3.12.7, numpy, matplotlib, matplotlib.pyplot, matplotlib.ticker, scipy.stats, networkx
