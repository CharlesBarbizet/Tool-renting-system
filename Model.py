#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 08:18:46 2020

@author: charlesbarbizet
"""

# Libraries:
import numpy as np
from numpy import linalg as LA
import math as m
import time as t

#%% class HarwellBoeing

class HarwellBoeing:
    # Attributes -------------------------------------------------------------
    
    def __init__(self,
                 value_double_tuple, 
                 i_coef_int_tuple, 
                 j_coef_int_tuple,
                 m_int, 
                 n_int):
        
        self.aa = value_double_tuple
        self.ia= i_coef_int_tuple
        self.ja = j_coef_int_tuple
        
        # Matrix size : (m,n)
        self.m = m_int
        self.n = n_int
    
    # Methods ----------------------------------------------------------------

    #   Method to compute left vector self.multiplication in sparse form
    def multiplication(self,vec):
        # Left vector multiplication for sparse matrix : z = A*vec
        # Input:    multiplied vector vec
        # Output:   z
        
        # Check if multiplication is legit
        if self.n!= np.size(vec):
            print('Sparse multiplication: SIZE ERROR')
            return
        
        z = np.zeros(self.m)
        for i in range(self.m):
            for j in range(self.ia[i],self.ia[i+1]):
                z[i] += self.aa[j]*vec[self.ja[j]]
        return z

#%% class Warehouse

class Warehouse:
    # Attributes -------------------------------------------------------------
    def __init__(self, 
                 arrival_rate, 
                 capacity, 
                 occupation = 0 ):
        self.l = arrival_rate
        self.k = occupation
        self.M = capacity

#%% class System
        
class System:
    # Attributes -------------------------------------------------------------
    
    def __init__(self, 
                 arrival_rate_list, 
                 capacity_list, 
                 servcice_rate):
        # Number of warehouses
        self.N = len(arrival_rate_list)
        
        # Initiate a list of warehouses
        self.warehouses = []
        for i in range(self.N):
            self.warehouses.append(Warehouse(arrival_rate_list[i],capacity_list[i]))
        
        # Initiate the service rate of the system
        self.mu = servcice_rate
        
        # Size of the State Space
        self.StateSpaceSize = 1
        for i in range(self.N):
            self.StateSpaceSize *= (self.warehouses[i].M + 1)
            
        # Maximum occupation of the SYstem
        self.max_occupation = 0
        for warehouse_index in range(self.N):
            self.max_occupation += self.warehouses[warehouse_index].M
    
    def GenerateTransitionMatrix(self):
        # We want to generate the TRANSPOSED transition rate matrix, hence we
        # need to generate Lambda by columns, which means that for each state 
        # we need to find the states that can access this state in one step
        text = False
        
        # Performance measure for generation time
        start = t.time()
        
        self.Q = HarwellBoeing( [] , [] ,  [] , self.StateSpaceSize , self.StateSpaceSize )
        
        row_pointer = 0
        for state_coef in range(self.StateSpaceSize):
            # Add pointer to ia for each new state
            self.Q.ia.append(row_pointer)
            
            if text:
               print("\tAdd of ia pointer: ", row_pointer)
            
            # Transfrom the coef into a state to find adjacent states
            state = self.Order2State(state_coef)
            
            if text:
                print("New line for state", state, " or coef ", state_coef)
            
            # Set the stable transition rate to zero 
            # and substract relevant transitions to it
            stable = 0
            
            # Check all adjacent states
            for i in range(self.N):
                
                # Compute transition arrival rate
                rate = self.warehouses[i].l
                for j in range(self.N):
                    if state[j] == self.warehouses[j].M:
                        rate += self.warehouses[j].l * self.P[i][j]
                
                # Arrival in warehouse i (k_i>0):
                if state[i] > 0:
                    # Define origine state                    
                    state[i] -= 1
                    
                    # Add coeficient to ja
                    self.Q.ja.append(self.State2Order(state))
                    # Keep track for ia pointers
                    row_pointer+=1
                    
                    if text:
                        print("\tAdd of arrival transition from\t", state,"\twith rate {0:1.3} \tand ja {1:}".format(rate, self.State2Order(state)) )
                    
                    # Back to current state                    
                    state[i] += 1
                    
                    # Add transition rate to aa
                    self.Q.aa.append(rate)
                    
                    # Keep track for futur stable transition rate (arrival - 0<k_i<M_i )
                    if state[i] < self.warehouses[i].M:
                        stable -= rate
                    
                    # Keep track for futur stable transition rate (service - 0<k_i )
                    stable -= state[i] * self.mu
                else:
                    # For futur stable transition rate (arrival k_i+0)
                    stable -= rate
                
                # Service in warehouse i (k_i<M_i):
                if state[i] < self.warehouses[i].M:
                    # Define origine state                    
                    state[i] += 1
                    
                    # Add coeficient to ja
                    self.Q.ja.append(self.State2Order(state))
                    # Keep track for ia pointers
                    row_pointer+=1
                    
                    if text:
                        print("\tAdd of service transition from\t", state,"\twith rate {0:1.3} \tand ja {1:}".format(state[i]*self.mu, self.State2Order(state))) 
                    
                    # Compute and add transition rate to aa
                    self.Q.aa.append(state[i] * self.mu)
                    
                    # Back to current state                    
                    state[i] -= 1
            
            # Add stabel transition rate to aa
            self.Q.aa.append(stable)
            
            # Add coeficient to ja
            self.Q.ja.append(state_coef)
            # Keep track for ia pointers
            row_pointer+=1
            
            if text:
                print("\tAdd of stable transition from\t", state,"\twith rate {0:1.3} \tand ja {1:}".format(stable, self.State2Order(state)) )
        
        self.Q.ia.append(row_pointer)
        
        # Performance measure for generation time
        end = t.time()
        self.Generation_time = end - start
        
        return self.Q    
    
    # Methods -----------------------------------------------------------------
    
    #%% SOR Methods
        
    def Gen_Initial_Guess(self, method = 2):
        if method == 1:
            self.initial_guess = (1/self.StateSpaceSize)*np.ones(self.StateSpaceSize)
        if method == 2:
            self.initial_guess = np.random.randint(self.StateSpaceSize, size=(self.StateSpaceSize, 1))
            self.initial_guess = self.initial_guess/np.sum(self.initial_guess)
        
        # Check if initial_guess is legit
        if self.Q.n!= np.size(self.initial_guess):
            print('SOR: Initial guess size ERROR')
            print("\tInitial guess by method {O} and value:".format(method), self.initial_guess)
            return
        
        return self.initial_guess
    
    def SOR(self, 
            w = 1, 
            max_it = 10**5, 
            threshold = 10**-8 ):
        # Time measurement for performance measures
        start = t.time()
        
        # Initialisation of variables
        self.stationary_distribution = np.copy(self.initial_guess)
        dif_test = m.inf
        self.SOR_it = 0
        
        self.test_1 = []
        self.test_2 = []
        self.test_3 = []
        
        while(True):
            # Keep in memory two succesive aproximation:
            self.former_stationary_distribution = np.copy( self.stationary_distribution )
            
            # SOR iteration loop to obtain next aproximation
            for i in range(self.Q.m):
                sum_tmp = 0
                for j in range(self.Q.ia[i],self.Q.ia[i+1]-1):
                    sum_tmp += self.Q.aa[j]*self.stationary_distribution[self.Q.ja[j]]
                j += 1
                aa_ii = self.Q.aa[j]
                self.stationary_distribution[i] = (1-w)*self.stationary_distribution[i] - (w/aa_ii)*sum_tmp
            
            # Keep the number of iteration in memory:
            self.SOR_it += 1
            
            # Normalization:
            norm_stationary = np.sum(self.stationary_distribution)
            if norm_stationary < 10**-25: 
                print('SOR: Underflow Error')
                print("\tFor w = {0}, max_it = {1}, and threshold = {2}".format(w,max_it,threshold))
            self.stationary_distribution /= norm_stationary            
            
            # dif_test is the diference between the first coeficient 2 succesive iterations
            dif_test = abs(self.former_stationary_distribution[0] - self.stationary_distribution[0])
            
            # Testing convergence ---------------
            
            # 1-st test (basic) - low computational power convergence test
            if dif_test<threshold:                
                # If test is passed, do a battery of more advanced tests
                # 2-nd test - Relative convergence test:
                norm_2 = self.stationary_distribution-self.former_stationary_distribution
                norm_2 /= self.stationary_distribution
                norm_2 = abs(norm_2)
                norm_2 = max(norm_2)
                
                if norm_2<threshold:
                    # This part is the third convergence test discussed in the thesis
                    # 3-rd test - Size of residuals:
                    #residuals = self.Q.multiplication(self.stationary_distribution)
                    #norm_3 = LA.norm(residuals)
                    #if norm_3<threshold*10:
                    break
                
            if (self.SOR_it>=max_it):
                print("SOR : max_it reached")
                print("\tFor w = {0}, max_it = {1}, and threshold = {2}".format(w,max_it,threshold))
                break
        
        # Time measurement for performance measures
        end = t.time()
        self.SOR_time = end - start
        return (self.stationary_distribution,self.SOR_it)
    
    def SOR_Optimalw(self, max_it, threshold, range_value = 5000):
        min_it = m.inf      
        for potential_w in np.linspace(0,2,range_value):
            #SYS2 = stat3.Stationarity2(l, M, mu, P2, method, potential_w, max_it, threshold)
            self.SOR(potential_w, max_it, threshold)
            
            #if SYS.SOR_it != SYS2.SOR_it:
            #    print("\tDifferent number of iteration for w = ", potential_w)
            #    error += 1
            
            if self.SOR_it < min_it:
                min_it = self.SOR_it
                min_w = potential_w
        self.SOR(min_w, max_it, threshold)
        self.Print()
        return (min_w, min_it)
    
    #%% Method for generation
    
    def State2Order(self, state):
        # Transforms a state into a coeficient according to a given mapping
        order = 0 
        prod = 1
        for i in range(self.N):
            order += prod*state[i]
            prod *= self.warehouses[i].M +1
        return int(order)
    
    def Order2State(self, indice):
        # Transforms a indice into a state according to a given mapping
        Quotient = self.StateSpaceSize
        State = [0] * self.N
        for i in range(self.N-1,-1,-1):
            Quotient /= (self.warehouses[i].M + 1)
            (State[i], indice) = divmod(indice,Quotient)
        return State
    
    #%% Method for performance
    
    # Total performance
    
    def TotalDistribution(self,n):
        # Find marginal density for a given warehouse index_warehouse
        if n>self.max_occupation or n<0:
            print('ERROR: TotalDistribution with wrong indice')
        p = 0
        for state_index in range(self.StateSpaceSize):
            state = self.Order2State(state_index)
            K = 0
            for warehouse_index in range(self.N):
                K += state[warehouse_index]
            if K==n:
                p += self.stationary_distribution[self.State2Order(state)]
        return p
    
    def TotalMean(self):
        # Define the mean total number of element
        self.total_mean = 0
        for n in range(self.max_occupation+1):
            self.total_mean += n*self.TotalDistribution(n)
        return self.total_mean

    def TotalVariance(self):
        # Define the total variance of the occupation
        # First compute the mean of the squared total occupation
        EN2 = 0
        for n in range(self.max_occupation+1):
            EN2 += n**2*self.TotalDistribution(n)
        # Variance = E[N2] - E[N]2
        self.total_variance = EN2 - self.total_mean**2
        return self.total_variance
    
    def TotalEmpty(self):
        # Define the probability of empty total occupation
        self.total_empty = self.stationary_distribution[0]
        return self.total_empty
                
    def TotalFull(self):
        # Define the probability of full total occupation
        self.total_full = self.stationary_distribution[self.StateSpaceSize-1]
        return self.total_full
    
    # Marginal performance 
    
    def MarginalDistribution(self,index_warehouse,n):
        # Find marginal density for a given warehouse index_warehouse
        if n>self.warehouses[index_warehouse].M or n<0:
            print('ERROR: MarginalDistribution with wrong indice')
        p = 0
        for state_index in range(self.StateSpaceSize):
            state = self.Order2State(state_index)
            if state[index_warehouse]==n:
                p += self.stationary_distribution[self.State2Order(state)]
        return p
    
    def ErlangDistribution(self,index_warehouse,n):
        # Find Erlang's probability according to the loss model
        # Keep in mind that we are using Erlang'model to compare without relocation
        # Hence we consider the arrival rate to simply be lambda
        p = self.MarginalDistribution(index_warehouse,0)
        p *= ((self.warehouses[index_warehouse].l/self.mu)**n)
        p/= m.factorial(n)
        return p
    
    def OccupationMeans(self):
        # Define the mean number of element in each warehouses
        self.occupation_means = []
        for i in range(self.N):
            E = 0
            for n in range(self.warehouses[i].M+1):
                #print(n, self.MarginalDistribution(i,n))
                E += n*self.MarginalDistribution(i,n)
            self.occupation_means.append(E)
        return self.occupation_means

    def OccupationVariance(self):
        # Define the variance of the occupation
        self.occupation_variances = []
        for i in range(self.N):
            EN2 = 0
            # First compute the mean of the squared occupation
            for n in range(self.warehouses[i].M+1):
                EN2 += (n**2)*self.MarginalDistribution(i,n)
            # Variance = E[N2] - E[N]2
            self.occupation_variances.append(EN2 - self.occupation_means[i]**2)
    
    def OccupationEmpty(self):
        # Define the probability of empty occupation
        self.occupation_empty = [] 
        for i in range(self.N):
            self.occupation_empty.append(self.MarginalDistribution(i,0))
        return self.occupation_empty
                
    def OccupationFull(self):
        # Define the probability of full occupation
        self.occupation_full = [] 
        for i in range(self.N):
            self.occupation_full.append(self.MarginalDistribution(i,self.warehouses[i].M))
        return self.occupation_full                
    
    def Performance(self):
        # Total performance
        self.TotalMean()
        self.TotalVariance()
        self.TotalEmpty()
        self.TotalFull()
        # Marginal performance
        self.OccupationMeans()
        self.OccupationVariance()
        self.OccupationEmpty()
        self.OccupationFull()
    
        #%% Display method Methods
    
    def PrintPerformance(self, details = True):
        print('Performance display: \n')
        
        # Total performance
        print(' - Total Performance:')
        print('\t mean occupation = {0} or {1}%'.format(self.total_mean, 
                                                        (self.total_mean/self.max_occupation)*100))
        print('\t variance occupation = {0}'.format(self.total_variance))
        print('\t empty probability = {0}'.format(self.total_empty))
        print('\t full probability = {0} \n'.format(self.total_full))
        
        if details:
            # Marginal performance
            for i in range(self.N):
                print(' - warehouse {0}:'.format(i))
                print('\t mean occupation = {0} or {1}%'.format(self.occupation_means[i],
                                                        (self.occupation_means[i]/self.warehouses[i].M)*100))
                print('\t variance occupation = {0}'.format(self.occupation_variances[i]))
                print('\t empty probability = {0}'.format(self.occupation_empty[i]))
                print('\t full probability = {0} \n'.format(self.occupation_full[i]))
    
    def PrintSystem(self):
        print('System display: \n')
        
        print(' - Number of warehouses = {0}'.format(self.N))
        print(' - State space size = {0}'.format(self.StateSpaceSize))
        #print('Stationnary distribution = ', self.stationary_distribution)
        
    def PrintTime(self):
        print('Time display: \n')
        
        # Generation performance
        print(' - Generation:')
        print('\t time of generation = {0} \n'.format(self.Generation_time))
        
        # SOR performance
        print(' - SOR:')
        print('\t number of iteration for SOR = {0}'.format(self.SOR_it))
        print('\t time of SOR = {0} \n'.format(self.SOR_time))
    
    def Print(self):
        print('\n')
        self.PrintSystem()
        print('\n')
        self.PrintPerformance()
        print('\n')
        self.PrintTime()
    
    #%% Relocation Matrix Methods
    
    def RelocationMatrix(self, method):
        # Method to generate the relocation matrix according to a given method
        #   Method 0 : no relocation 
        #   Method 1 : uniform relocation
        #   Method 2 : random relocation
        #   Method 3 : unique relocation toward warehouse j
        
        if method == 0:
            #   Method 0 : no relocation
            self.P = np.identity(self.N)
            
        elif method == 1:
            #   Method 1 : uniform relocation
            self.P = (1/self.N)*np.ones((self.N,self.N))
            
        elif method == 2:
            #   Method 2 : random relocation
            self.P = np.random.randint(10, size=(self.N, self.N))
            self.P = self.P/np.sum(self.P,1)[:,None]
            
        elif method == 3:
            #   Method 3 : unique relocation toward optimal warehouse
            #   Search for optimal warehouse unique_optimal
            #   Generate system without relocation
            self.RelocationMatrix(0)
            self.GenerateTransitionMatrix()
            self.Evaluate(2, 1, 10**5, 10**-8)
            
            self.unique_optimal = np.argmin(self.occupation_means)
            #   Relocation to j (best index)
            if self.unique_optimal<0 or self.unique_optimal>self.N :
                print('ERROR: unique relocation toward non existing warehouse')
            self.P = np.zeros((self.N,self.N))
            self.P[:,self.unique_optimal] = np.ones(self.N)
                
    def Evaluate(self, initial_guess_method, w, max_it, threshold):
        # Find stationary distribution
        self.Gen_Initial_Guess(initial_guess_method)
        self.SOR(w, max_it, threshold)
        
        # Compute performance measures
        self.Performance()
            
    
#%% Summary function

def Stationarity( arrival_rate_list, capacity_list, servcice_rate, relocation_method, initial_guess_method, w, max_it, threshold):
    # This function is suppose to handle all the initialisation of the system and
    # the computation of all the performance meausres
    
    # Generate System
    SYS = System(arrival_rate_list, capacity_list, servcice_rate)
    SYS.RelocationMatrix(relocation_method)
    SYS.GenerateTransitionMatrix()
    
    # Evaluate performance
    SYS.Evaluate(initial_guess_method, w, max_it, threshold)
    return SYS
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    