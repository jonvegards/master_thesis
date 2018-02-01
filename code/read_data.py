#!/usr/bin/env python2

"""Functions for reading in data to BDT model
search in BDT.py

@ Jon Vegard Sparre 2018 <jonvsp@fys.uio.no>
"""

import pandas as pd
import numpy as np
import textwrap, sys

outpref = sys.argv[0] + ' : '

eps2 = .01

def ReadData(filenames, feature_list, target_list, drop_col=None, train_test=False, samples=10000, eps=1E-9, squark_mean=False):
    """ Function for reading in test data to BDT model.

    Args:

        filenames:      list with filenames as strings
        feature_list:   list with features to fetch from data set
        target_list:    list with name of target(s)
        drop_col:       name of column to be dropped
        train_test:     bool, if true data set is split into train and
                        test set, else it will return a given number of
                        samples
        samples:        number of samples to fetch
        eps:            small number to replace zero xsec
        squark_mean     bool, if true variables from partonic LO cross
                        sections are added to output

    Return:

        features_test:  array with test data for a BDT
        target_test:    array with test target values for a BDT
        features:       array with training data for a BDT
        targe:          array with target values for a BDT

    """
    print(outpref + "Small number eps = {}".format(eps))
    dataframes = []
    for files in filenames:
        data   = pd.read_csv(files, sep=' ', skipinitialspace=True)
        df     = pd.DataFrame(data)
        df_new = df.drop(drop_col, axis=1) # Drop BS column
        # Removing outliers (i.e. samples with too large masses)
        # based on what we saw in quality check plots
        for particle in feature_list:
            df_new = df_new[df_new[particle] < 4100]
        dataframes.append(df_new)
        print(outpref + "Data frames:")
        print(outpref + "len(data) = {}".format(len(df_new)))
        print(outpref + "data.shape = {}".format(df_new.shape))

    if len(filenames) > 1:
        # Concatenate data frames
        df = pd.concat(dataframes, ignore_index=True)
        print(outpref + "Merged data frame")
        print(outpref + "len(data) = {}".format(len(df)))
        print(outpref + "data.shape = {}".format(df.shape))
        print(outpref)
        print(df[0:5])
    elif len(filenames) == 1:
        df = dataframes[0]

    if squark_mean:
        # mean of squark masses
        squarks_list = feature_list[1:]
        # feature_list.append("mean_squark_mass")
        squarks = df[squarks_list].mean(axis=1)
        df = df.assign(mean_squark=squarks.values)                                                                                                                                 
        s = 8000.0 # GeV COM energy

        # m_minus^2                                                                                                                                                                
        m_minus2 = df["3.mGluino"]**2 - df["mean_squark"]**2
        df = df.assign(m_minus2=m_minus2.values)
        feature_list.append("m_minus2")

        # beta_gluino
        beta_gluino = np.sqrt(1. - (4*df["3.mGluino"]**2)/(s**2))
        df = df.assign(beta_gluino=beta_gluino.values)
        feature_list.append("beta_gluino")

        # L_2
        L_2 = np.log10((s**2 - 2*m_minus2**2 - s**2*beta_gluino)/(s**2 - 2*m_minus2**2 + s**2*beta_gluino))
        L_2[pd.isnull(L_2)] = eps2         # Replacing NaNs with a small number.
        df = df.assign(L_2=L_2.values)
        feature_list.append("L_2")
        
        print(df[0:5])

    # Split data for training using function from sklearn (can also use pandas functionality)
    if train_test:
        print(outpref + "Splitting data set in training and test set...")
        train = df.sample(frac=0.9, random_state=42, replace=False)
        test  = df.drop(train.index)
        print(outpref + "Traing and test data frames:")
        print(outpref + "len(data) = {} {}".format(len(train), len(test)))
        print(outpref + "data.shape = {} {}".format(train.shape,test.shape))
        # Define features to train on and target numbers (used below)
        # Convert to array for scikit using values
        # ravel is used to avoid [[n]], i.e. n x 1 arrays
        features = train[feature_list].values
        target   = train[target_list].values.ravel()
        features_test = test[feature_list].values
        target_test   = test[target_list].values.ravel()
        target_test[target_test == 0.0]   = eps # replacing zero xsec with small number
        target[target == 0.0]   = eps # replacing zero xsec with small number
        target        = np.log10(target) 
        target_test   = np.log10(target_test)
        return features, target, features_test, target_test

    
    else:
        print(outpref + "Fetching {} samples from dataset".format(samples))
        test = df.sample(n=samples, random_state=42, replace=False)
        print(outpref + "Output data frames:")
        print(outpref + "len(data) = {}".format(len(test)))
        print(outpref + "data.shape = {}".format(test.shape))
        # Define features to train on and target numbers (used below)
        # Convert to array for scikit using values
        # ravel is used to avoid [[n]], i.e. n x 1 arrays
        features_test = test[feature_list].values
        target_test   = test[target_list].values.ravel()
        target_test[target_test == 0.0]   = eps # replacing zero xsec with small number
        target_test   = np.log10(target_test)
        return features_test, target_test