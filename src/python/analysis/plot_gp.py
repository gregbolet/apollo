import glob
#from pylab import *
import brewer2mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import argparse
import pandas as pd
import os

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

#def loadDataFiles(gpFile='gp.dat', dataFile='data.dat', acquiFile='acqui.dat', numPolicies=0):
#    print('Reading in data files...')
#    gp = np.loadtxt(gpFile)
#    print('Read GP file', gpFile)
#    data = np.loadtxt(dataFile)
#    print('Read raw sampled data file', dataFile)
#    acqui = np.loadtxt(acquiFile)
#    print('Acquisition Fnct file', acquiFile)
#    print('File reading complete!')
#
#    # re-scaling the data to use the provided policy space
#    if numPolicies != 0:
#        rescaler = np.vectorize(rescale)
#        gp[:,0] = rescaler(gp[:,0], numPolicies, toInt=False)
#        data[:,0] = rescaler(data[:,0], numPolicies)
#        acqui[:,0] = rescaler(acqui[:,0], numPolicies, toInt=False)
#
#    return gp, data, acqui

def plotGP(ax, gp, data):#, numPolicies):

    # assumes only one region's data is being plotted
    regionName = gp.loc[0,'region']

    # GP/exp conf interval
    ax.fill_between(gp.x, gp['mean'] - gp['std'],  gp['mean'] + gp['std'], alpha=0.25, linewidth=0, color=colors[0], zorder=0)

    # GP/exp
    ax.plot(gp.x, gp['mean'], linewidth=2, color=colors[0], zorder=1, label='GP Kernel Mean')

    # sampled points
    ax.scatter(x=data.x, y=data.y, s=5, color=colors[2], zorder=2, label='Data')

    # color the top10 points red
    top10 = data.nlargest(10, columns=['y'])
    ax.scatter(x=top10.x, y=top10.y, s=5, color='red', zorder=3, label='Best Points')

    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
                        fancybox=True, shadow=True, ncol=5)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', length=0)

    #if numPolicies == 0:
    #    #uniqueXvals = np.unique(np.concatenate((data[:,0], np.array([0.0, 1.0]))))
    #    uniqueXvals = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #    ax.set_xticklabels(uniqueXvals)
    #    ax.set_xticks(uniqueXvals)


    ax.set_axisbelow(True)
    ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
    ax.set_title(regionName)
    ax.set_xlabel('Policy Index')
    ax.set_ylabel('-1 * xtime')
    return


def plotAcqui(ax, gp):
    # assumes only one region's data is being plotted
    ax.fill_between(gp.x, 0,  gp.acqui, alpha=0.25, linewidth=0, color=colors[1])
    ax.plot(gp.x, gp.acqui, linewidth=2, color=colors[1])
    ax.set_title('Acquisition Function')
    ax.set_xlabel('Policy')
    ax.set_ylabel('Score')
    return

def plotTimeseries(ax, data):
    # assumes only one region's data is being plotted
    #xtime = np.maximum.accumulate(data[:,1])
    #step = np.array(list(range(len(xtime))))
    #earliestStep = np.argmax(xtime)

    xtime = data.y.cummax()
    step = np.array(list(range(len(xtime))))
    #noMoreImprovements = data.nlargest(1, columns=['y'])['x']

    #top10Samples = np.argsort(data[:,1])[-10:][::-1]
    #print('Top 10 BEST samples')
    #for idx in top10Samples:
    #    print(f'[step {idx}] policy idx {data[idx,0]} --> {data[idx,1]}')

    ax.plot(step, xtime, linewidth=1, color=colors[3])
    #ax.axvline(noMoreImprovements, color=colors[4], ls='--')
    ax.set_title('Best Sample Timeseries')
    ax.set_xlabel('Sample Timestep')
    ax.set_ylabel('-1 * xtime')
    return


# policy values are assumed to be in alphabetical order 
# of the region files
def loadFiles(filesDir, policies):
    boFiles = list(glob.glob(filesDir+"/*.bo"))
    datFiles = list(glob.glob(filesDir+"/*.dat"))

    #print(boFiles, '\n', datFiles)

    allBOData = pd.DataFrame(columns=['x', 'mean', 'std', 'acqui', 'region'])
    allDatData = pd.DataFrame(columns=['x', 'y', 'region'])

    for boFile in boFiles:
        regionName = os.path.basename(boFile)[:-3]
        boData = pd.read_csv(boFile, header=None, names=['x', 'mean', 'std', 'acqui'], sep=' ')

        boData['region'] = regionName
        #print(boData.shape)
        #print(boData.head())
        allBOData = pd.concat([allBOData, boData], ignore_index=True)

    for datFile in datFiles:
        regionName = os.path.basename(datFile)[:-4]
        datData = pd.read_csv(datFile, header=None, names=['x', 'y'], sep=' ')

        datData['region'] = regionName
        #print(datData.shape)
        #print(datData.head())
        allDatData = pd.concat([allDatData, datData], ignore_index=True)

    
    print('BO Data')
    print(allBOData.shape)
    #print(allBOData.head())
    #print(allBOData.tail())
    
    print('Time Data')
    print(allDatData.shape)
    #print(allDatData.head())
    #print(allDatData.tail())

    boRegionNames = sorted(list(allBOData['region'].unique()))
    datRegionNames = sorted(list(allDatData['region'].unique()))

    assert((len(boRegionNames) == len(datRegionNames)) and (boRegionNames == datRegionNames))

    if (len(policies) == 1) and (len(datRegionNames) > 1):
        policies = policies*len(datRegionNames)

    assert(len(policies) == len(datRegionNames))

    polMap = dict(zip(datRegionNames, policies))

    [print(k,' ',v) for k,v in polMap.items()]

    # now let's rescale the x-axis data to be the policies that were actually executed
    allDatData['x'] = allDatData.apply(lambda r: rescale(r['x'], polMap[r['region']], toInt=True), axis=1)
    allBOData['x'] = allBOData.apply(lambda r: rescale(r['x'], polMap[r['region']], toInt=False), axis=1)

    print('BO Data')
    print(allBOData.shape)
    #print(allBOData.head())
    #print(allBOData.tail())
    
    print('Time Data')
    print(allDatData.shape)
    #print(allDatData.head())
    #print(allDatData.tail())

    return allBOData, allDatData

def plotData(boData, timeData):

    # let's make a figure where we have 3 rows, and the number of columns
    # corresponds to the number of regions we have
    regions = list(boData['region'].unique())
    numRegions = len(regions)

    fig, axs = plt.subplots(3,numRegions, figsize=(6*numRegions,10), gridspec_kw={'height_ratios':[2,1,1]}) 

    for j in range(numRegions):
        regionName = regions[j]
        gp = boData[boData['region'] == regionName].reset_index()
        data = timeData[timeData['region'] == regionName].reset_index()
        print('plotting', regionName)
        for i in range(3):
            ax = axs[i,j]
            if i == 0:
                plotGP(ax, gp, data)
            elif i == 1:
                plotAcqui(ax, gp)
            else:
                plotTimeseries(ax, data)


    plt.tight_layout()

    ## save our plot for viewing
    print('saving figure...')
    fig.savefig('gp.png')
    print('figure saved!')

    return


if __name__ == "__main__":
    # Let's handle user inputs to the data files
    parser = argparse.ArgumentParser(prog='GP Viz', 
                                     description='Visualizes the Gaussian Processes used with Apollo')

    # we're just going to plot all the .dat and .bo files in the current dir
    # the .bo files contain 4 columns, x-value, mean-value (gp), std-value (gp), acqui-fnct value
    # the .dat files contain the samples fed to the BO method


    parser.add_argument('--filesDir', default='./', type=str)

    # Let's also handle user inputs for the number of policies in each dimension
    # we only support one dimension at the moment
    #parser.add_argument('--numPolicies', help='Max policy index for each dimension plotted', 
    #                    default=0, type=int)

    parser.add_argument('--policies', nargs='+', type=int, default=[100], help='Number of policies explored by each region', required=False)

    args = parser.parse_args()

    print("Given the following input args: \n", args)

    boData, timeData = loadFiles(args.filesDir, args.policies)
    plotData(boData, timeData)



    #gp, data, acqui = loadDataFiles(args.gp, args.data, args.acqui, args.numPolicies)

    #fig, axs = plt.subplots(3,1, figsize=(7,10), gridspec_kw={'height_ratios':[2,1,1]}) 

    #plotGP(axs[0], gp, data, args.numPolicies)
    #plotAcqui(axs[1], acqui)
    #plotTimeseries(axs[2], data)

    #plt.tight_layout()

    ## save our plot for viewing
    #print('saving figure...')
    #fig.savefig('gp.png')
    #print('figure saved!')
