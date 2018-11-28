"""
@author: Mhamed
"""
import numpy as np
import time

class MDP():
    
    def __init__(self, proba, reward, gamma=0.95, epsilon_vi = 1e-2):
        self.proba = proba #matrix probability p(s_j | s_i,a_k)
        self.reward = reward #reward matrix r(s_i,a_k)
        self.gamma = gamma #discount factor

        self.ns = self.proba.shape[0] #number of possible states
        self.na = self.proba.shape[1] #number of possible actions 
        self.states = range(self.ns)
        self.actions = range(self.na)
        self.epsilon_vi = epsilon_vi #threshold for value iteration convergence
        

    def value_iteration(self, initial_V, verbose=True):
        V_old = initial_V
        V = V_old.copy()
        iterations = 0
        V_hist = list()
        
        if verbose:
            start = time.time()
        while np.abs(V - V_old).max() > self.epsilon_vi or iterations == 0:
            V_old = V.copy()
            for s in self.states:
                V[s] = np.max(self.reward[s, :] + self.gamma * (self.proba[:, s, :].T).dot(V))
                
            V_hist.append(V.copy())
            iterations += 1

        if verbose:
            print("{0} iterations before convergence in {1:.7f} seconds".format(iterations, time.time() - start))
            
        optimal_policy = list()
        for s in self.states : 
            optimal_policy.append(np.argmax(self.reward[s, :] + self.gamma * (self.proba[:, s, :].T).dot(V),axis=0))
        return np.array(optimal_policy), V ,V_hist
        
    
    def policy_iteration(self, initial_policy, verbose = True):
        old_pi = initial_policy
        pi = old_pi.copy()
        
        iterations = 0
        if verbose:
            start = time.time()
        
        while not np.array_equal(pi, old_pi) or iterations == 0:
            old_pi = pi.copy()
            # 2. policy evauation
            V_pi = self._policy_evaluation(pi)
            # 3. policy improvement
            for s in self.states:
                pi[s] =  np.argmax(self.reward[s, :] + self.gamma * (self.proba[:, s, :].T).dot(V_pi),axis=0)
            iterations += 1
        
        if verbose:
            print("{0} iterations before convergence in {1:.7f} seconds".format(iterations, time.time() - start))
        
        return pi
    
    
    def _policy_evaluation(self, policy):  
        
        Ppi = np.array([self.proba[: , j, policy[j]] for j in range(self.ns)])
        Rpi = np.array([self.reward[j, policy[j]] for j in range(self.ns)])
        Vpi = np.dot(np.linalg.inv(np.eye(self.ns)-self.gamma*Ppi),Rpi)
        
        return Vpi
    
    
    
    