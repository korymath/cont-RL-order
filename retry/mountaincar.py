'''
mountain car modified to have an extra penalty for non-zero action
'''

# from pylab import random, cos
import numpy as np
from random import randint

# chance of sheep
chanceOfSheep = 50


# returns position, velocity, and 1 for no sheep (2 is sheep)
def init():
    position = -0.6 + np.random.random_sample()*0.2
    return position, 0.0, 1

def sample(S,A, T, keepSheep):
    position,velocity, sheep = S
    
    #print('pos: ' + "{0:5.3f}".format(position) + '\tvel: ' + "{0:5.3f}".format(velocity) + '\tsheep: ' + str(sheep) + '\t A: ' + str(A))
    if not A in (0,1,2):
        print 'Invalid action:', A
        raise StandardError
    A = A - 1

    if T:
        return 1, (position, velocity, sheep), T
    
    # could be exactly based on book reward pg 214
    # R = -1 if A==0 else -2

    # no sheep
    R = 0
    if A == -1 or A == 1:
        R = .1
    hitSheep = False
    #sheep and didnt stop
    if sheep == 2 and ((velocity > 0 and A == 1) or (velocity < 0 and A == -1)):
        R = -50
        hitSheep = True
        velocity = 0


    #possible sheep appears on road
    if not keepSheep:
        sheep = 2 if randint(1,chanceOfSheep) == 1 else 1


    if not hitSheep:
        velocity += 0.001*A - 0.0025*np.cos(3*position)
        if velocity < -0.07:
            velocity = -0.07
        elif velocity >= 0.07:
            velocity = 0.06999999

    position += velocity

    if position >= 0.5 or hitSheep:
        return R, (position,velocity, sheep), True

    if position < -1.2:
        position = -1.2
        velocity = 0.0
    return R,(position,velocity, sheep), False
