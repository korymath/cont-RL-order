import math

numTilings = 12 # originally 4
numTiles = 9 * 9 * numTilings # originally 9*9

def tilecode(in1, in2, tileIndices):
    in1 += 1.2
    in2 += 0.07
    for n in range(numTilings):
        # in1 is the x, or position values, bounded by -1.2 and 0.5
        # in2 is the y, or velocity values, bounded by -0.07 and 0.07
        # need to get a positive value, so we shift up
        offsetX = n * (1.7/8) / numTilings
        offsetY = n * (0.14/8) / numTilings
        idxX = int(math.floor(8 * (in1 + offsetX)/1.7))
        idxY = int(math.floor(8 * (in2 + offsetY)/0.14))
        tileIndices[n] = int((81 * n) + (9 * idxY) + idxX) # originally 81*n + 9*...

def printTileCoderIndices(in1,in2):
    tileIndices = [-1]*numTilings
    tilecode(in1, in2, tileIndices)
    print 'Tile indices for input (', in1, ',', in2, ') are : ', tileIndices

# Check the bounding limits for tile coder sanity
# printTileCoderIndices(-1.2,-0.07)
# printTileCoderIndices(0.5, 0.07)
# Tile indices for input ( -1.2 , -0.07 ) are :  [0, 81, 162, 243]
# Tile indices for input ( 0.5 , 0.07 ) are :  [80, 161, 242, 323]