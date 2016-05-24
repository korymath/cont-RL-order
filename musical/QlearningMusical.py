import musicalMountain as mountaincar
from musicalTilecoder import simple_tiles
import numpy as np
from pylab import *
import random
import sys

import matplotlib.pyplot as plt
numTilings = 12
observationRanges =  [[-1.2,.5], [-0.07, 0.07], [1,2]]
observationAcc = [.01, .01, 1]
numstates = 1
for i in range(0,3):
    numstates *= (observationRanges[i][1] - observationRanges[i][0]) / observationAcc[i]
    numstates = int(math.floor(numstates))

print('numstates: ' + str(numstates))
numTiles = numstates * numTilings
print('numTiles: ' + str(numTiles))
numRuns = 10
numEpisodes = 50
alpha = 0.4/numTilings
gamma = 1
lmbda = 0.9
epsilon = 0.001
n = numTiles * 3
zerovec = np.zeros(n)
state = [-1]*numTilings
actions = [0, 1, 2]
minNumExtraSteps = 1
maxNumExtraSteps = 1
runSum = 0.0

previous_tiles = [0 for item in range(numTilings)]
flipped = False
if len(sys.argv) == 2 and sys.argv[1] == 'flipped':
    flipped = True

# output arrays
bigReturn = np.zeros(shape=(numRuns, numEpisodes))
bigSteps = np.zeros(shape=(numRuns, numEpisodes))

def Qs(state):
    Q = [0, 0, 0]
    
    for a in actions:
        Q[a] += w[state + (a*numstates)]
    return Q

def writeF():
    fout = open('value500ep.out', 'w')
    F = [0]*numTilings
    steps = 50
    for i in range(steps):
        for j in range(steps):
            tilecode(-1.2+i*1.7/steps, -0.07+j*0.14/steps, F)
            height = -max(Qs(F))
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
print('n:' + str(n))
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
        #tilecode(observation[0], observation[1], observation[2], state)

        state = simple_tiles(observation, observationRanges, observationAcc)

        # compute the Q values for the state and every action
        Q = Qs(state)

        terminal = False
        A = chooseAction(Q)
        unknownObs = observation

        if flipped:
            R, observation, terminal = mountaincar.sample(observation, A, terminal)
            someRandomAmountOfTime = random.randint(minNumExtraSteps,maxNumExtraSteps)
            for i in range(1, someRandomAmountOfTime):
                unknownR, unknownObs, terminal, surprise = mountaincar.sample(unknownObs, A, terminal)
                G += unknownR
            step += someRandomAmountOfTime

        # repeat for each step of episode
        while True:

            if not flipped:
                # take action a and get reward R and new observation
                R, observation, terminal = mountaincar.sample(unknownObs, A, terminal)
                print(observation)
                # if newObservation is terminal
                if terminal:
                    w += alpha*delta*e
                    break

                delta = R - Q[A]
                G += R

                # update the replacing traces
                e[state+(A*numstates)] = 1

                # function approximation
                #tilecode(observation[0], observation[1], observation[2], state)
                state = simple_tiles(observation, observationRanges, observationAcc)

                # compute the Q values for the state and every action
                Q = Qs(state)

                A = chooseAction(Q)

                # learning
                delta += gamma*Q[np.argmax(Q)]
                w += alpha*delta*e
                e = gamma*lmbda*e



            if flipped:
                observation = unknownObs

                delta = R - Q[A]
                G += R

                # update the replacing traces
                e[state+(A*numstates)] = 1

                # function approximation
                #tilecode(observation[0], observation[1], observation[2], state)

                state = simple_tiles(observation, observationRanges, observationAcc)

                # compute the Q values for the state and every action
                Q = Qs(state)

                A = chooseAction(Q)

                R, observation, terminal = mountaincar.sample(observation, A, terminal)
                if terminal:
                    w += alpha*delta*e
                    break

                # learning
                delta += gamma*Q[np.argmax(Q)]
                w += alpha*delta*e
                e = gamma*lmbda*e


            # take some time with the world changing
            unknownObs = observation
            someRandomAmountOfTime = random.randint(minNumExtraSteps,maxNumExtraSteps)
            for i in range(1, someRandomAmountOfTime):
                unknownR, unknownObs, terminal, unknownSurprise = mountaincar.sample(unknownObs, A, terminal)
                G += unknownR 
                if terminal:
                    print('In terminal')
                    w += alpha*delta*e
                    break

            # update the observation
            step += someRandomAmountOfTime

        # collect output for analysis
        bigReturn[run][episodeNum] = G
        bigSteps[run][episodeNum] = step
        returnSum = returnSum + G

    print "Average return:", returnSum/numEpisodes
    runSum += returnSum
print "Overall average return:", runSum/numRuns/numEpisodes
writeF()
print 'saving flipped'
np.savetxt('flipped-returns500run.out', bigReturn)
np.savetxt('flipped-steps500run.out', bigSteps)

