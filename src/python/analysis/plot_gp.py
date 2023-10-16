import glob
#from pylab import *
import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

mpl.use('agg')

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': [8, 4.5]
}
mpl.rcParams.update(params)

print('updated params')

gp = np.loadtxt('gp.dat')
data = np.loadtxt('data.dat')

fig, axs = plt.subplots(2,1, figsize=(7,8), gridspec_kw={'height_ratios':[1,3]}) 

ax = axs[1]

# GP/exp conf interval
ax.fill_between(gp[:,0], gp[:,1] - gp[:,2],  gp[:,1] + gp[:,2], alpha=0.25, linewidth=0, color=colors[0])

# GP/exp
ax.plot(gp[:,0], gp[:,1], linewidth=2, color=colors[0])

# sampled points
ax.plot(data[:,0], data[:, 1], 'o', color=colors[2])

legend = ax.legend(["GP-kernel mean", 'Data'], loc=8)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)

uniqueXvals = np.unique(np.concatenate((data[:,0], np.array([0.0, 0.25, 0.5, 0.75, 1.0]))))
ax.set_xticklabels(uniqueXvals)
ax.set_xticks(uniqueXvals)

ax.set_axisbelow(True)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
ax.set_title('Exploration Space')


# let's plot the acqusition function now
ax = axs[0]

acqui = np.loadtxt('acqui.dat')

ax.fill_between(acqui[:,0], 0,  acqui[:,1], alpha=0.25, linewidth=0, color=colors[1])
ax.plot(acqui[:,0], acqui[:,1], linewidth=2, color=colors[1])
ax.set_title('Acquisition Function')

fig.suptitle('BO Visualization')

# save our plot for viewing
print('saving figure...')
fig.savefig('gp.png')
