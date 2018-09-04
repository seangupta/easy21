#easy21 RL project

from __future__ import division
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def step(s, a, verbose = False):
    '''state s is dealer's first card 1-10 and player's sum 1-21, action a (of player) is
    hit or stick. returns a sample of the next state s' and reward r'''
    dealer_sum, player_sum = s[0], s[1]
    assert dealer_sum in range(1,11)
    assert player_sum in range(1,22)
    assert a in ("hit","stick")
    if verbose:
        print "dealer_sum = ", dealer_sum, "player_sum = ", player_sum, "action = ", a
    if a == "hit":
        new_card = random.choice(range(1,11))
        if verbose: 
            print "new_card = ", new_card
        if random.random() < 1./3:
            #colour red
            if verbose: print "red"
            player_sum -= new_card
            if player_sum < 1:
                if verbose: print "player loses"
                return ("terminal",-1)
        else:
            #colour black
            if verbose: print "black"
            player_sum += new_card
            if player_sum > 21:
                if verbose: print "player loses"
                return ("terminal",-1)
        if verbose: 
            print "player didn't go bust"
            print ((dealer_sum,player_sum),0)
        return ((dealer_sum,player_sum),0) #player didn't go bust
    else:
        #player sticks so dealer starts taking turns
        while dealer_sum < 17:
            if verbose: print "dealer_sum = ", dealer_sum
            new_card = random.choice(range(1,11))
            if verbose: print "new_card = ", new_card
            if random.random() < 1./3:
                #colour red
                if verbose: print "red"
                dealer_sum -= new_card
                if dealer_sum < 1:
                    if verbose: print "player wins"
                    return ("terminal",1)
            else:
                #colour black
                if verbose: print "black"
                dealer_sum += new_card
                if dealer_sum > 21:
                    if verbose: print "player wins"
                    return ("terminal",1)
        #dealer didn't go bust
        if verbose: print "dealer didn't go bust"
        if player_sum > dealer_sum:
            r = 1
        elif player_sum == dealer_sum:
            r = 0
        else:
            r = -1
        if verbose: print "reward = ",r
        return ("terminal",r)

step((5,6),"hit")        
step((5,6),"stick")

#action-value function (query as Q[dealer_card][player_sum][action] where
#action 0 is hit and action 1 is stick
Q = [[[0,0] for col in range(21)] for row in range(10)]

def play_game(Q,epsilon,verbose = False):
    '''plays easy21 under epsilon-greedy policy'''
    dealer_card = random.choice(range(1,11))
    player_card = random.choice(range(1,11))
    s = (dealer_card, player_card)
    episode = []
    if verbose: print "new game"
    #while state not terminal
    while s != "terminal":
        if verbose: print "state = ", s
        #a = random.choice(("hit","stick"))
        hit_val, stick_val = Q[s[0]-1][s[1]-1]
        greedy_a = "hit" if hit_val > stick_val else "stick"
        if random.random() < epsilon/2 + 1 - epsilon:
            a = greedy_a
        else:
            a = "hit" if greedy_a == "stick" else "stick"
        episode.append((s,a)) #disregard terminal state
        s, reward = step(s,a)
    #game has ended
    if verbose: print "game has ended, reward = ", reward
    return episode, reward

#state-action counts
N_sa = [[[0,0] for col in range(21)] for row in range(10)]
num_episodes = 10**6
for i in range(num_episodes):
    epsilon = 1/(i+1) #control (policy improvement): new policy is epsilon-greedy wrt new Q
    episode, reward = play_game(Q,epsilon)
    #print "episode = ", episode
    #print "reward = ", reward
    #given sample episode. reward does not depend on time-step 
    #(no discounting and only one reward signal at end)
    
    #prediction
    for s,a in episode:
        old_count = N_sa[s[0]-1][s[1]-1][0 if a == "hit" else 1]
        old_Q = Q[s[0]-1][s[1]-1][0 if a == "hit" else 1]
        N_sa[s[0]-1][s[1]-1][0 if a == "hit" else 1] += 1
        Q[s[0]-1][s[1]-1][0 if a == "hit" else 1] += 1/(old_count + 1) * (reward - old_Q)
    
V = [[max(Q[row][col]) for col in range(21)] for row in range(10)]
N_s = [[sum(N_sa[row][col]) for col in range(21)] for row in range(10)]
learned_pol = [[(0 if Q[row][col][0] > Q[row][col][1] else 1) \
for col in range(21)] for row in range(10)]

plt.imshow(V, cmap='hot')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = range(1,11)
y = range(1,22)
Y, X = np.meshgrid(y,x)
ax.plot_surface(X,Y,V)
ax.set_xlabel('dealer card')
ax.set_ylabel('player sum')
ax.set_zlabel('Q')
plt.show()

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.imshow(N_s)
plt.show()

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.imshow(learned_pol)
plt.show()