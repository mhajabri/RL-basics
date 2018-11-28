# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 13:40:08 2018

@author: Mhamed
"""

import numpy as np

def simulate_trajectories(env, policy, n, Tmax):
    # n is the number of trajectories to simulate
    # policy is the policy that we'll use (provide a list of action to take in any state)
    states = []
    actions = []
    rewards = []
    for i in range(n):
        #We simulate one trajectory here
        tmp_states = []
        tmp_actions = []
        tmp_rewards = []
        tmp_states.append(env.reset())
        t = 0
        is_state_terminal = False
        while (t < Tmax and not is_state_terminal):
            tmp_actions.append(policy[tmp_states[t]])
            update_state, update_reward, is_state_terminal = env.step(tmp_states[t], tmp_actions[t])
            tmp_states.append(update_state)
            tmp_rewards.append(update_reward)
            t += 1
            
        states.append(tmp_states)
        actions.append(tmp_actions)
        rewards.append(tmp_rewards)
        
    return states, actions, rewards


def MC(states, actions, rewards, gamma, ns, verbose = False) : 
    """
    INPUT : 
    list of states, actions and rewards on "n" trajectories
    factor gamma 
    number of possible states, ns
    OUTPUT : 
    MC estimate of the value of each state
    """
    V = list()
    for s in range(ns):
        rewards_s = [rewards[i] for i in range(len(states)) if states[i][0]==s]
        Vs=0
        for traj in range(len(rewards_s)) : 
            Vs+=sum([gamma**(t-1)*rewards_s[traj][t] for t in range(len(rewards_s[traj]))])
        try : 
            V.append(Vs/len(rewards_s))
        except : 
            if verbose==True : 
                print('No trajectory found that starts from the state {0}, try increasing nb of trajectories')
            V.append(np.inf)
            
    return np.array(V)
            
def Qlearning(env, n, Tmax, eps, func_alpha, gamma, ns, na):
    
    """
    INPUT : 
    -------------------------------------------------------
    env : environment (in this example, a particular grid)
    Tmax : Lenght of each episode
    n : number of episodes
    eps : probability of exploration VS exploitation
    ns/na : number of states / actions
    gamma : discount rate 
    func_alpha : learning rate as a function of t, s and a
    
    OUTPUT:
    -------------------------------------------------------
    Q : of the Q-learning algorithm
    Vn : array of shape (n, ns), containing n value functions
    cum_rewards : list of len =n, cumulated reward for each episode
    """
    
    Q = np.zeros((ns, na))
    N = np.zeros((ns, na)) # Count the number of times (s, a) has been generated
    Vn = np.zeros((n, ns)) # Each row corresponds to v_pin found at episode n
    cum_rewards = [0]*n 
    
    #start with impossible (s,a) : 
    for s in range(ns):
        impossible_a = [i for i in range(na) if i not in env.state_actions[s]]
        Q[s, impossible_a] = np.nan
        
    # simulate n episodes of length Tmax 
    for i in range(n):
        state = env.reset()
        t = 0
        is_state_terminal = False
        while (t < Tmax and not is_state_terminal):
            strategy = np.random.choice(['greedy', 'explore'], size=None, p=[1-eps, eps])
            
            if strategy=="greedy" and np.argmax(Q[state]) in env.state_actions[state]:
                a = np.argmax(Q[state])
            else : 
                a = np.random.choice(env.state_actions[state],size=1)
                
            N[state, a] += 1
            s_new, reward, is_state_terminal = env.step(state, a)
            cum_rewards[i] += reward
            t += 1
            alpha = func_alpha(N[state,a],state,a)
            Q[state, a] = (1-alpha)*Q[state,a] + alpha*(reward + gamma*np.nanmax(Q[s_new]))
            state = s_new
            
        Vn[i:,]=np.nanmax(Q, axis=1)
            
    return Q, Vn, cum_rewards