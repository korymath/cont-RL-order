import math

numTilings = 1 # originally 4

xTiling = 2
yTiling = 2
sheepTiling = 2

offsets = [xTiling * yTiling*sheepTiling, xTiling*sheepTiling, sheepTiling]


numTiles = xTiling * yTiling * numTilings * sheepTiling

def tilecode(in1, in2, in3, tileIndices):
    in1 += 1.2
    in2 += 0.07
    for n in range(numTilings):
        # in1 is the x, or position values, bounded by -1.2 and 0.5
        # in2 is the y, or velocity values, bounded by -0.07 and 0.07
        # need to get a positive value, so we shift up
        offsetX = n * (1.7/(xTiling-1)) / numTilings
        offsetY = n * (0.14/(yTiling-1)) / numTilings
        offsetSheep = n / numTilings

        idxX = int(math.floor((xTiling-1) * (in1 + offsetX)/1.7))

        idxY = int(math.floor((yTiling-1) * (in2 + offsetY)/0.14))
        idxSheep = int(math.floor((in3 + offsetSheep)/2))

        #print('idxX: ' + str(idxX) + '  idxY: ' + str(idxY) + '  idxSheep: ' + str(idxSheep))
        tileIndices[n] = int((offsets[0] * n) + (offsets[1] * idxY) + (offsets[2] * idxX) + (idxSheep))
def printTileCoderIndices(in1,in2, in3):
    tileIndices = [-1]*numTilings
    tilecode(in1, in2, in3, tileIndices)
    print 'Tile indices for input (', in1, ',', in2, ',', in3,') are : ', tileIndices

# Check the bounding limits for tile coder sanity
# printTileCoderIndices(-1.2,-0.07)
# printTileCoderIndices(0.5, 0.07)
# Tile indices for input ( -1.2 , -0.07 ) are :  [0, 81, 162, 243]
# Tile indices for input ( 0.5 , 0.07 ) are :  [80, 161, 242, 323]