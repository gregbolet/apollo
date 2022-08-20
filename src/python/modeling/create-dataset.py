#!/usr/bin/env python3

import argparse
import pandas as pd
import glob

'''
Reads data from CSV traces and creates a loadable dataset for training models.
'''

colsToKeep=[]


def print_to_str(index, features, policy, xtime):
    features = [str(n) for n in features]
    output = '  ' + str(index) + ': { features: [ '
    for fi in features:
        output += fi + ', '
    output += ']'
    output += ', policy: ' + str(policy)
    output += ', xtime: ' + str(xtime)
    output += ' },\n'
    return output


def agg_by_none(data_per_region):
    output = '# agg by none\n'

    output += 'data: {\n'
    feature_cols = list(data_per_region.columns[5:-2])

    # Output all the data points.
    for i, row in data_per_region.iterrows():
        output += print_to_str(i, row[feature_cols],
                               row['policy'], row['xtime'])

    output += '}\n\n'
    return output


def agg_by_mean(data_per_region):
    output = '# agg by mean\n'

    output += 'data: {\n'
    feature_cols = list(data_per_region.columns[5:-2])

    # Group by features and policy to extract policies per feature with
    # their mean xtime.
    groups = data_per_region.groupby(by=feature_cols + ['policy'])
    index = 0
    for name, g in groups:
        features = name[:-1]
        policy = name[-1]
        xtime = g['xtime'].mean()
        output += print_to_str(index, features, policy, xtime)
        index += 1

    output += '}\n\n'
    return output


def agg_by_min(data_per_region):
    output = '# agg by min\n'

    output += 'data: {\n'
    feature_cols = list(data_per_region.columns[5:-2])

    groups = data_per_region.groupby(by=feature_cols)
    index = 0
    for _, g in groups:
        # Find the index of the minimum xtime.
        min_g = g.loc[g['xtime'].idxmin()]

        # Extract data from the minimum xtime data point.
        features = min_g[feature_cols]
        policy = min_g['policy']
        xtime = min_g['xtime']
        output += print_to_str(index, features, policy, xtime)
        index += 1

    output += '}\n\n'
    return output


def agg_by_mean_min(data_per_region):
    output = '# agg by mean_min\n'

    output += 'data: {\n'
    print('all cols:', data_per_region.columns)

    #feature_cols = list(data_per_region.columns[5:-2])
    # these features are sometimes out of order, we need to extract
    # the columns that start with 'f' for feature
    #feature_cols = list(data_per_region.columns[6:-2])
    feature_cols = []
    for col in list(data_per_region.columns):
        if (col[0] == 'f'):
            if (col in colsToKeep) or (len(colsToKeep) == 0):
                feature_cols.append(col)

    print('feature columns', feature_cols)

    groups = data_per_region.groupby(by=feature_cols)
    index = 0
    # let's add in a 0 vector to the dataset
    output += print_to_str(index, [0]*len(feature_cols), 0, 0)
    index += 1
    for name, g in groups:
        # Group by policy to compute the mean xtime.
        g2 = g.groupby(by=['policy'])
        mean_g = g2['xtime'].mean()

        # Find the index (equals the policy) with the minimum mean xtime.
        idxmin = mean_g.idxmin()

        # Create a list of feature values if there is a single feature.
        if len(feature_cols) > 1:
            features = name
        else:
            features = [name]
        policy = idxmin
        xtime = mean_g.loc[idxmin]
        output += print_to_str(index, features, policy, xtime)
        index += 1

    output += '}\n\n'
    return output


# Concatenate all the trace data into one CSV
def read_tracefiles(tracefiles):
    data = pd.DataFrame()

    for f in tracefiles:
        #print('Read', f)
        try:
            csv = pd.read_csv(f, sep=' ', header=0, index_col=False)
            # drop any rows with NaN values (they were malformed)
            csv = csv.dropna(axis='rows').reset_index()
        except pd.errors.EmptyDataError:
            print('Warning: no data in', f)

        data = pd.concat([data, csv], ignore_index=True, sort=False)

    return data

# get the trace data from each directory and concatenante it all into one csv
def read_tracedirs(tracedirs):
    data = pd.DataFrame()

    for dir in tracedirs:
        globlist = glob.glob('%s/trace*.csv' % (dir))
        data_dir = read_tracefiles(globlist)
        data = pd.concat([data, data_dir], ignore_index=True, sort=False)

    return data


def main():
    global colsToKeep

    parser = argparse.ArgumentParser(
        description='Create loadable training datasets from existing CSV measurements.')
    parser.add_argument('--tracedirs', nargs='+',
                        help='trace directories')
    parser.add_argument('--tracefiles', nargs='+',
                        help='trace filenames')
    parser.add_argument('--agg',
                        help='aggregate measures ', choices=['none', 'mean', 'min', 'mean-min'], required=True)
    parser.add_argument('--singlemodel', action='store_true',
                        help='Should we make a single dataset?')
    parser.add_argument('-k', '--colsToKeep', nargs='+', default=[], 
                        help='What feature columns to keep when creating a dataset?')
    args = parser.parse_args()

    if len(args.colsToKeep) != 0:
        colsToKeep = args.colsToKeep

    if not (args.tracedirs or args.tracefiles):
        raise RuntimeError('Either tracedirs or tracefiles must be set')

    if args.tracefiles:
        data = read_tracefiles(args.tracefiles)

    if args.tracedirs:
        data = read_tracedirs(args.tracedirs)



    print('Finished reading in all the datasets! Gathering unique regions...')
    #print('data\n', data)
    regions = data['region'].unique().tolist()


    print('All DATA COLUMNS:', data.columns)

    print('Creating region dataset files...')

    # this allows us to make a single datasetfile
    if args.singlemodel:
        print('Making single model...')
        
        if args.agg == 'none':
            output = agg_by_none(data)
        elif args.agg == 'mean':
            output = agg_by_mean(data)
        elif args.agg == 'min':
            output = agg_by_min(data)
        elif args.agg == 'mean-min':
            output = agg_by_mean_min(data)
        else:
            raise RuntimeError('Invalid aggregation args ' + str(args.agg))

        with open('Dataset-' + 'single-model' + '.yaml', 'w') as f:
            f.write(output)

        print('Wrote dataset file for single model')
        return


    # Create datasets per region.
    for r in regions:
        print('About to write dataset file for region:', r)
        # drops columns that contain missing values
        #data_per_region = data.loc[data['region'] == r].dropna(axis='columns').reset_index()
        data_per_region = data.loc[data['region'] == r].dropna(axis='columns').reset_index()
        #data_per_region = data_per_region.dropna(axis='rows').reset_index()
        #data_per_region = data.loc[data['region'] == r].reset_index()
        #print(data_per_region.head())
        #print(data_per_region.tail())
        print('region has %d rows of data'%(data_per_region.shape[0]))

        if args.agg == 'none':
            output = agg_by_none(data_per_region)
        elif args.agg == 'mean':
            output = agg_by_mean(data_per_region)
        elif args.agg == 'min':
            output = agg_by_min(data_per_region)
        elif args.agg == 'mean-min':
            output = agg_by_mean_min(data_per_region)
        else:
            raise RuntimeError('Invalid aggregation args ' + str(args.agg))
        with open('Dataset-' + str(r) + '.yaml', 'w') as f:
            f.write(output)

        print('Wrote dataset file for region:', r)

if __name__ == "__main__":
    main()
