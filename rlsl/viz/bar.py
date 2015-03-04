import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

import numpy as np

import rlsl.viz.style

def plot_bars(output_filename, df):
    rlsl.viz.style.reset_cycles()
    f, ax = plt.subplots(2, sharex=True)

    sample = np.unique(df['problem size'])
    num_items = len(sample)
    num_classifiers = len(np.unique(df['classifier']))
    ind = np.arange(num_items)

    margin = 0.10
    width = (1.-2.*margin)/num_classifiers

    for num,i in enumerate(np.unique(df['classifier'])):
        xdata = ind+margin+(num*width)
        ax[1].bar(xdata, df[df['classifier']==i].accuracy.values, width, label=i, 
                #hatch=next(rlsl.viz.style.HATCH),
                color=next(rlsl.viz.style.COLOR))
                #edgecolor='black')

    rlsl.viz.style.reset_cycles()

    for num,i in enumerate(np.unique(df['classifier'])):
        xdata = ind+margin+(num*width)
        ax[0].bar(xdata, df[df['classifier']==i].f1.values, width, label=i,
                #hatch=next(rlsl.viz.style.HATCH),
                color=next(rlsl.viz.style.COLOR))
                #edgecolor='black')

    #ax.set_xticks(ind+margin+num*width)
    ax[1].set_xticks(ind+margin+(num_classifiers/2)*width)
    ax[1].set_xticklabels(sample)

    ax[1].set_ylabel('Accuracy', fontsize=14)
    ax[0].set_ylabel('F1', fontsize=14)
    ax[1].set_xlabel('Problem Size', fontsize=14)

    ax[1].legend(bbox_to_anchor=(0.5, 1.10), loc='lower center', ncol=3)

    rlsl.viz.style.tuftestyle(ax[0])
    rlsl.viz.style.tuftestyle(ax[1])

    plt.savefig(output_filename)

def plot_importances(output_filename, df):
    rlsl.viz.style.reset_cycles()
    f, ax = plt.subplots(len(np.unique(df['classifier'])))

    for i,clf in enumerate(np.unique(df['classifier'])):
        data = df[df['classifier'] == clf][:10]

        ax[i].set_ylabel('Feature Importance')
        ax[i].set_xlabel('Feature Name')
        ax[i].set_title(clf)
        ax[i].bar(range(len(data['importance'])), data['importance'].values, color=next(rlsl.viz.style.COLOR),align='center')
        ax[i].set_xticks(range(len(data['importance'])))
        ax[i].set_xticklabels(map(lambda x:x[:10], data['feature_name'].values))
        ax[i].set_xlim([-1, len(data['importance'])])
        rlsl.viz.style.tuftestyle(ax[i])

    plt.savefig(output_filename)
