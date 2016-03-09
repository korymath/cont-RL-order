import mountaincar
from Tilecoder import numTilings, tilecode, numTiles
from Tilecoder import numTiles as n
import numpy as np
# from pylab import *
import random

import matplotlib.pyplot as plt

numRuns = 5
numEpisodes = 50
alpha = 0.4/numTilings
gamma = 1
lmbda = 0.9
epsilon = 0.001
n = numTiles * 3
zerovec = np.zeros(n)
state = [-1]*numTilings
actions = [0, 1, 2]
runSum = 0.0

# output arrays
bigReturn = np.zeros(shape=(numRuns, numEpisodes))
bigSteps = np.zeros(shape=(numRuns, numEpisodes))

def Qs(state):
    Q = [0, 0, 0]
    for a in actions:
        for index in state:
            Q[a] += w[index + (a*12*9*9)]
    return Q

def writeF():
    fout = open('value500ep', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps, F)
            height = -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

# represent actions decelerate, coast, accelerate as integers
for run in range(numRuns):
    w = -0.01*np.random.rand(n)
    returnSum = 0.0
    for episodeNum in range(numEpisodes):
        G = 0
        step = 0

        # From Figure 9.9 in Sutton RL 2014
        # n-component eligibility trace vector
        e = np.zeros(n)

        # initialize observation
        observation = mountaincar.init()

        # use function approximation to generate next state
        tilecode(observation[0], observation[1], state)

        # compute the Q values for the state and every action
        Q = Qs(state)



        # repeat for each step of episode
        while True:
            
            # choose action a (epsilon-greedy)
            if np.random.uniform(0, 1) < epsilon:
                A = np.random.choice(actions)
                e = np.zeros(n)
            else:
                # choose A from max, Q
                A = Q.index(max(Q))

            # take some time with the world changing
            someRandomAmountOfTime = random.randint(1,5)
            unknownObs = observation
            for i in range(1, someRandomAmountOfTime):
                unknownR, unknownObs, terminal = mountaincar.sample(unknownObs, A)            
                if terminal:
                    w += alpha*delta*e
                    break

            # take action a and get reward R and new observation
            R, newObservation, terminal = mountaincar.sample(unknownObs, A)

            delta = R - Q[A]
            G += R

            # update the replacing traces
            for index in state:
                # index to the action space
                e[index+(A*12*9*9)] = 1

            # if newObservation is terminal is terminal
            if terminal:
                w += alpha*delta*e
                break

            # function approximation
            tilecode(newObservation[0], newObservation[1], state)

            # compute the Q values for the state and every action
            Q = Qs(state)

            # learning
            delta += gamma*Q[np.argmax(Q)]
            w += alpha*delta*e
            e = gamma*lmbda*e

            # update the observation
            observation = newObservation
            step += 1

        print "Episode: ", episodeNum, "Return: ", G

        # collect output for analysis
        bigReturn[run][episodeNum] = G
        bigSteps[run][episodeNum] = step
        returnSum = returnSum + G

    print "Average return:", returnSum/numEpisodes
    runSum += returnSum
print "Overall average return:", runSum/numRuns/numEpisodes
writeF()
np.savetxt('returns500run.out', bigReturn)
np.savetxt('steps500run.out', bigSteps)

t = range(numEpisodes)
fig = plt.figure()
ax1 = fig.add_subplot(111)
y = np.mean(bigReturn, axis=0)
e = np.std(bigReturn, axis=0)
ax1.errorbar(t, y, e)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
y = np.mean(bigSteps, axis=0)
e = np.std(bigSteps, axis=0)
ax2.errorbar(t, y, e)
plt.show()


# A sweep of parameters tested for alpha and epsilon are in the 3D plot in the part 2 folder. From this plot, it is
# apparent that a smaller epsilon and a moderate alpha combination is ideal. From this plot, we can conclude that the
# optimal value for alpha is between 0.3 and 0.6, and the optimal epsilon is very low (0.001-0.002). This makes sense
# we are looking at the first 200 episodes per run. This means that we want faster early learning, so a higher alpha is
# ideal. A smaller epsilon restricts exploration, so our learning is likely to be repeated for the same features each
# time.

# *** Note*** alpha values are stated as a number, but they really mean the value/numTilings. We were just a little lazy...

# Here are some results from some trials with 5 runs to start:
# alpha = 0.6, epsilon = 0.005: avg return = -297.91
# alpha = 0.4, epsilon = 0.001: avg return = -285.185, 50 runs: -294.35
# other values of alpha and epsilon in these ranges were not performing as well.

# Next set of tests used alpha = 0.4, epsilon = 0.001
# Try different tiling parameters. We thought that a better solution could be obtained with a coarser tiling.
# We first looked at using coarser tilings since it is advantageous for early learning. Results are as follows:
# Tiling parameters:
# 8 tilings of 5X5: -397 (way worse!)
# 4 tilings of 5X5: -395
# Also tried asymmetrical tilings (size = 9X5). These performed horribly: range of -400
# Try finer tilings:
# 4 tilings of 13X13: -323
# 2 tilings of 9X9: -266.53
# 8 tilings of 9X9: -269.481, after 50 runs: -275.713
# 10 tilings of 9X9: -268.182, after 50 runs: -272.4425, 5 runs of 500 episodes (for fun): -241.9588
# ****************************************
# 12 tilings of 9X9: after 50 runs: -266.5319, after 500 runs: -266.73222 (we chose this as our optimal parameters)
# More tilings covering the whole space means that there is a lot of overlap balanced with a good level of
# fineness. This is ideal for early learning.
# Just for fun: 50 runs of 500 episodes: -241.73208
# The runs with longer episodes perform better because the weights are retained and used for learning for a longer
# period of time. Less episodes per run means that the weight vectors are reset more often.

# Compared to the initial parameters:
# These were calculated by summing all returns for the first 200 episodes.
# Initial parameter mean performance (50runs)
#    9.9559e+04
#
# Standard Error
#   419.6630
#
# Tweaked parameter mean performance (500runs)
#    5.8974e+04
#
# Standard Error
#   136.2432
#
# Initial parameter mean performance (500runs)
#    9.9651e+04
#
# Standard Error
#   126.5347
#
# Success!
# Mean performance
# Difference between sum of all rewards received in first 200 episodes:
#    4.0677e+04
#
# 2.5 times the larger of the two standard errors:
#   340.6079

# We are clearly superior to the original parameters. This is also apparent in the learning curves generated.
# We achieve much faster learning than the initial parameters.

