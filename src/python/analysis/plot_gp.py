import glob
#from pylab import *
import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import argparse

mpl.use('agg')

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}
mpl.rcParams.update(params)

def rescale(a, numPolicies, toInt=True):
    if toInt:
        return min(int(a*numPolicies), numPolicies-1)
    else:
        return min(a*numPolicies, numPolicies-1)

def loadDataFiles(gpFile='gp.dat', dataFile='data.dat', acquiFile='acqui.dat', numPolicies=0):
    print('Reading in data files...')
    gp = np.loadtxt(gpFile)
    print('Read GP file', gpFile)
    data = np.loadtxt(dataFile)
    print('Read raw sampled data file', dataFile)
    acqui = np.loadtxt(acquiFile)
    print('Acquisition Fnct file', acquiFile)
    print('File reading complete!')

    # re-scaling the data to use the provided policy space
    if numPolicies != 0:
        rescaler = np.vectorize(rescale)
        gp[:,0] = rescaler(gp[:,0], numPolicies, toInt=False)
        data[:,0] = rescaler(data[:,0], numPolicies)
        acqui[:,0] = rescaler(acqui[:,0], numPolicies, toInt=False)

    return gp, data, acqui

def plotGP(ax, gp, data, numPolicies):
    # GP/exp conf interval
    ax.fill_between(gp[:,0], gp[:,1] - gp[:,2],  gp[:,1] + gp[:,2], alpha=0.25, linewidth=0, color=colors[0], zorder=0)

    # GP/exp
    ax.plot(gp[:,0], gp[:,1], linewidth=2, color=colors[0], zorder=1, label='GP Kernel Mean')

    # sampled points
    ax.scatter(x=data[:,0], y=data[:, 1], s=5, color=colors[2], zorder=2, label='Data')

    # color the top10 points red
    top10Samples = np.argsort(data[:,1])[-10:][::-1]
    ax.scatter(x=data[top10Samples,0], y=data[top10Samples, 1], s=5, color='red', zorder=3, label='Best Points')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                        fancybox=True, shadow=True, ncol=5)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    if numPolicies == 0:
        #uniqueXvals = np.unique(np.concatenate((data[:,0], np.array([0.0, 1.0]))))
        uniqueXvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        ax.set_xticklabels(uniqueXvals)
        ax.set_xticks(uniqueXvals)


    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_title('Exploration Space')
    ax.set_xlabel('Policy')
    ax.set_ylabel('-1 * xtime')
    return


def plotAcqui(ax, acqui):
    ax.fill_between(acqui[:,0], 0,  acqui[:,1], alpha=0.25, linewidth=0, color=colors[1])
    ax.plot(acqui[:,0], acqui[:,1], linewidth=2, color=colors[1])
    ax.set_title('Acquisition Function')
    ax.set_xlabel('Policy')
    ax.set_ylabel('Score')
    return

def plotTimeseries(ax, data):
    xtime = np.maximum.accumulate(data[:,1])
    step = np.array(list(range(len(xtime))))
    earliestStep = np.argmax(xtime)

    top10Samples = np.argsort(data[:,1])[-10:][::-1]
    print('Top 10 BEST samples')
    for idx in top10Samples:
        print(f'[step {idx}] policy idx {data[idx,0]} --> {data[idx,1]}')

    ax.plot(step, xtime, linewidth=1, color=colors[3])
    ax.axvline(earliestStep, color=colors[4], ls='--')
    ax.set_title('Best Sample Timeseries')
    ax.set_xlabel('Sample Timestep')
    ax.set_ylabel('-1 * xtime')
    return





if __name__ == "__main__":
    # Let's handle user inputs to the data files
    parser = argparse.ArgumentParser(prog='GP Viz', 
                                     description='Visualizes the Gaussian Processes used with Apollo')

    parser.add_argument('--gp', default='./gp.dat', type=str)
    parser.add_argument('--data', default='./data.dat', type=str)
    parser.add_argument('--acqui', default='./acqui.dat', type=str)

    # Let's also handle user inputs for the number of policies in each dimension
    # we only support one dimension at the moment
    parser.add_argument('--numPolicies', help='Max policy index for each dimension plotted', 
                        default=0, type=int)

    args = parser.parse_args()

    print("Given the following input args: \n", args)

    gp, data, acqui = loadDataFiles(args.gp, args.data, args.acqui, args.numPolicies)

    fig, axs = plt.subplots(3,1, figsize=(7,10), gridspec_kw={'height_ratios':[2,1,1]}) 

    plotGP(axs[0], gp, data, args.numPolicies)
    plotAcqui(axs[1], acqui)
    plotTimeseries(axs[2], data)

    plt.tight_layout()

    # save our plot for viewing
    print('saving figure...')
    fig.savefig('gp.png')
    print('figure saved!')
