# Reinforcement Learning for Molecular Dynamics
This project aim to use a reinforcement learing approach to solve sampling problems for metastable dynamical systems. The project catins two main packages. The algorithm package contains some state of the art reinforcement learning algorihtms (see below for more info). The environment package contains differnt dynmical system which can be simulated with different time evolution methods (see also below for more info)

## Algorihtms

This folder contains differen state of the art reinforcement learning algorithms. At the moment there are
	
- REINFORCE
- Actor Critic
- Cross Entropy
- DDPG

A demo for an low dimensional SDE environment can be found under 
```
"algorihtm*"/src/run_"alg"_sde.py
```

## Environments
Environments for this RL package are implemented in a different python package which is called molecules. The package can be found here <a href="https://github.com/jannes002A/molecules" target="_blank">Molecules</a>. To use it just follwo the install instructions on the Molecules page and after that pip install the sderl package in the new virtual environment using 
```
pip install -e .
```
Of course you can use the algorithms also with other self designed environments. Feel free to do your own experiments.

