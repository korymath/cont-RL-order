import mountaincar
from Tilecoder import numTilings, tilecode, numTiles
from Tilecoder import numTiles as n
import numpy as np
from pylab import *
import random
import sys

import matplotlib.pyplot as plt

numRuns = 100
numEpisodes = 100
alpha = 0.4/numTilings
gamma = 1
lmbda = 0.9
epsilon = 0.001
n = numTiles * 3
zerovec = np.zeros(n)
state = [-1]*numTilings
textActions = ['go backward', 'stop', 'go forward']
actions = [0, 1, 2]
minNumExtraSteps = 1
maxNumExtraSteps = 1
maxSteps = 1000
runSum = 0.0
flipped = False
filePrefix = ''
if len(sys.argv) == 2 and sys.argv[1] == 'flipped':
    flipped = True
    filePrefix = 'flipped-'

# output arrays
bigReturn = np.zeros(shape=(numRuns, numEpisodes))
bigSteps = np.zeros(shape=(numRuns, numEpisodes))

def Qs(state):
    Q = [0, 0, 0]
    for a in actions:
        for index in state:
            Q[a] += w[index + (a*numTiles)]
    return Q

def writeF():
    fout = open(filePrefix + 'value500ep.out', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps, 1, F)
            height = -max(Qs(F))
            tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps, 2, F)
            height += -max(Qs(F))
            fout.write(repr(height) + ' ')
        fout.write('\n')
    fout.close()

def chooseAction(Q):
    # choose action a (epsilon-greedy)
    if np.random.uniform(0, 1) < epsilon:
        A = np.random.choice(actions)
        e = np.zeros(n)
    else:
        # choose A from max, Q
        A = Q.index(max(Q))
    return A


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
        tilecode(observation[0], observation[1], observation[2], state)

        # compute the Q values for the state and every action
        Q = Qs(state)

        terminal = False
        A = chooseAction(Q)
        unknownObs = observation
        
        if flipped:
            R, observation, terminal = mountaincar.sample(observation, A, terminal, False)
            someRandomAmountOfTime = random.randint(minNumExtraSteps,maxNumExtraSteps)
            for i in range(1, someRandomAmountOfTime):
                unknownR, observation, terminal = mountaincar.sample(observation, A, terminal, True)
                G += unknownR
            step += someRandomAmountOfTime

        # repeat for each step of episode
        while True:

            if not flipped:
                # take action a and get reward R and new observation
                R, observation, terminal = mountaincar.sample(observation, A, terminal, False)

                # if newObservation is terminal
                if terminal:
                    w += alpha*delta*e
                    break

                delta = R - Q[A]
                G += R

                # update the replacing traces
                for index in state:
                    e[index+(A*numTiles)] = 1

                # function approximation
                tilecode(observation[0], observation[1], observation[2], state)

                # compute the Q values for the state and every action
                Q = Qs(state)

                A = chooseAction(Q)

                #if observation[2] == 2:
                #    print('There is a sheep and the classic algorithm acted with ' + textActions[A])

                # learning
                delta += gamma*Q[np.argmax(Q)]
                w += alpha*delta*e
                e = gamma*lmbda*e

            if flipped:

                delta = R - Q[A]
                G += R

                # update the replacing traces
                for index in state:
                    e[index+(A*numTiles)] = 1

                # function approximation
                tilecode(observation[0], observation[1], observation[2], state)

                # compute the Q values for the state and every action
                Q = Qs(state)
                A = chooseAction(Q)

                #if observation[2] == 2:
                #    print('There is a sheep and the flipped algorithm acted with ' + textActions[A])                

                R, observation, terminal = mountaincar.sample(observation, A, terminal, True)
                if terminal:
                    w += alpha*delta*e
                    break

                # learning
                delta += gamma*Q[np.argmax(Q)]
                w += alpha*delta*e
                e = gamma*lmbda*e

            # update the observation
            someRandomAmountOfTime = random.randint(minNumExtraSteps,maxNumExtraSteps)
            for i in range(1, someRandomAmountOfTime):
                if i == someRandomAmountOfTime:
                    unknownR, observation, terminal = mountaincar.sample(observation, A, terminal, False)    
                else:
                    unknownR, observation, terminal = mountaincar.sample(observation, A, terminal, True)
                G += unknownR      
            step += 1 + someRandomAmountOfTime
            if step > maxSteps:
                step = 1000
                terminal = True                

        # collect output for analysis
        bigReturn[run][episodeNum] = G
        bigSteps[run][episodeNum] = step
        returnSum = returnSum + G

    print "Average return:", returnSum/numEpisodes
    runSum += returnSum
print "Overall average return:", runSum/numRuns/numEpisodes
writeF()

print('Saving values')
np.savetxt(filePrefix + 'returns500run.out', bigReturn)
np.savetxt(filePrefix + 'steps500run.out', bigSteps)


r1 = loadtxt('returns500run.out')
s1 = loadtxt('steps500run.out')
r2 = loadtxt('flipped-returns500run.out')
s2 = loadtxt('flipped-steps500run.out')


t = range(numEpisodes)
fig = plt.figure()
ax1 = fig.add_subplot(121)
y = np.mean(r1, axis=0)
e = np.std(r1, axis=0)
l1 = ax1.errorbar(t, y, e)
y = np.mean(r2, axis=0)
e = np.std(r2, axis=0)
l2 = ax1.errorbar(t, y, e)

ax2 = fig.add_subplot(122)
y = np.mean(s1, axis=0)
e = np.std(s1, axis=0)
l3 = ax2.errorbar(t, y, e)
y = np.mean(s2, axis=0)
e = np.std(s2, axis=0)
l4 = ax2.errorbar(t, y, e)

fig.legend((l1, l2), ('Classic Return', 'Flipped Return'), 'upper left')
fig.legend((l3, l4), ('Classic Steps', 'Flipped Steps'), 'upper right')
plt.show()
