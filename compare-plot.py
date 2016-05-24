import sys
import matplotlib.pyplot as plt
from pylab import *

folder = sys.argv[1]
print(folder)
r1 = loadtxt(folder + 'returns500run.out')
s1 = loadtxt(folder + 'steps500run.out')
r2 = loadtxt(folder + 'flipped-returns500run.out')
s2 = loadtxt(folder + 'flipped-steps500run.out')


t = range(numEpisodes)
fig = plt.figure()
ax1 = fig.add_subplot(111)
y = np.mean(r1, axis=0)
e = np.std(r1, axis=0)
ax1.errorbar(t, y, e)
y = np.mean(r2, axis=0)
e = np.std(r2, axis=0)
ax1.errorbar(t, y, e)

fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
y = np.mean(s1, axis=0)
e = np.std(s1, axis=0)
ax2.errorbar(t, y, e)
y = np.mean(s2, axis=0)
e = np.std(s2, axis=0)
ax2.errorbar(t, y, e)
plt.show()
