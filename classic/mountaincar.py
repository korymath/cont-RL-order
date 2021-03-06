'''
mountain car modified to have an extra penalty for non-zero action
'''

# from pylab import random, cos
import numpy as np

def init():
    position = -0.6 + np.random.random_sample()*0.2
    return position, 0.0

def sample(S,A, T):
    position,velocity = S
    if not A in (0,1,2):
        print 'Invalid action:', A
        raise StandardError
    A = A - 1

    if T:
        return -1, (position, velocity), T
    
    # could be exactly based on book reward pg 214
   # R = -1 if A==0 else -2

    R = -1 

    velocity += 0.001*A - 0.0025*np.cos(3*position)
    if velocity < -0.07:
        velocity = -0.07
    elif velocity >= 0.07:
        velocity = 0.06999999



    position += velocity

    if position >= 0.5:
        return R, (position,velocity), True

    if position < -1.2:
        position = -1.2
        velocity = 0.0
    return R,(position,velocity), False
