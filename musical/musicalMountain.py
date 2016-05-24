'''
mountain car modified to have an extra penalty for non-zero action
'''

# from pylab import random, cos
import numpy as np
from random import randint

def init():
    position = -0.6 + np.random.random_sample()*0.2
    return position, 0.0, False

def sample(S,A,T):
    position,velocity, surprise = S
    if not A in (0,1,2):
        print 'Invalid action:', A
        raise StandardError
    A = A - 1


    R = -1


    if T:
        return R, (position, velocity), T, surprise


    # if surprise and not stop
    if surprise == 2 and A != 1:
        R = -10

    position += velocity

    velocity += 0.001*A - 0.0025*np.cos(3*position)
    if velocity < -0.07:
        velocity = -0.07
    elif velocity >= 0.07:
        velocity = 0.06999999

    if position >= 0.5:
        return R, (position,velocity), True, False

    if position < -1.2:
        position = -1.2
        velocity = 0.0

    if position > -.2 and position > .2:
        surprise = 2


    return R,(position,velocity, surprise), False
