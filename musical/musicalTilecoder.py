import math

def simple_tiles(observations, observationRanges, observationAcc):

    features = [0]*len(observations)
    numstates = 1
    for i in range(0,len(observations)):
        numstates *= (observationRanges[i][1] - observationRanges[i][0]) / observationAcc[i]
        features[i] = int(math.floor((observations[i] - observationRanges[i][0]) /observationAcc[i]))

    numstates = int(math.floor(numstates))

    tiled_features = 0

    for j in range(len(features)):
        tiled_features += features[j]*numstates**j

    return int(tiled_features)

# from musicalTilecoder import *
# observations = [.3, -.4, 1]
# observationRanges = [[-.2,.2],[-1.5, 1.5], [0,1]]
# observationAcc = [.1, .1, 1]
# simple_tiles(observations, observationRanges, observationAcc)