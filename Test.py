#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:33:04 2020

@author: charlesbarbizet
"""

# Import the Model file
import Model as model


#%% Initialization of the parameters

# Arrival rates 
lambda_1 = 0.8705
lambda_2 = 0.8364
lambda_3 = 0.7835
lambda_4 = 0.7335
l = [lambda_1, lambda_2, lambda_3, lambda_4]

# Capacities
M_1 = 8
M_2 = 8
M_3 = 8
M_4 = 8
M = [M_1, M_2, M_3, M_4]

# Service rate
mu = 0.5

#%% Initilisation of the input for the SOR method and the relocation scenarios

# Initial guess method: 
    # 1 is for unifrom initial guess
    # 2 is for random initial guess
initial_guess_method = 2

# Initiate the relocation method:
    # 0 is no-relocation
    # 1 is unifrom relocation
    # 2 is random relocation
    # 3 is unique relocation to optimal warehouse
relocation_method = 1

# Initiate the input for the SOR

# Relaxation weigth
w = 1
# Maximum number of iteration for the SOR
max_it = 10**5
# threshold for the convergence tests of the SOR 
    # with this value, the system converges in 120 seconds aproximately
threshold = 10**-8

# This function handles the creation of the system, 
# its global initialization and the computation of the performance measures
system = model.Stationarity(l, M, mu, relocation_method, initial_guess_method, w, max_it, threshold)

# This is the stationary distribution
stat = system.stationary_distribution
    

#%% Results

# There is built in function to display the system performances, the SOR performance 
# and basic information on the system. Print() call all of those functions.
system.Print()

#%% Remarks

# The method 'Stationarity' is practical 
# but it may provoke a bug in Python, where the instance of the class System
# is not visible in teh variable explorer

# you can create a system 'by hand like that:
    
# Generate System
# Initiate the system
system_2 = model.System(l, M, mu)
# Generate the relocation scenario
system_2.RelocationMatrix(relocation_method)
# Generate the transition rate matrix
system_2.GenerateTransitionMatrix()
    
# Evaluate performance
system_2.Evaluate(initial_guess_method, w, max_it, threshold)
