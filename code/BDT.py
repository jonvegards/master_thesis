#!/usr/bin/env python2

"""Example script for training a BDT to predict NLO cross sections
for gluino pair production and plotting the resulting precision.

@ Jon Vegard Sparre 2018 <jonvsp@fys.uio.no>
"""

# print(__doc__)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(**params)
import pandas as pd
import sklearn as sk
from sklearn import ensemble
from sklearn.externals import joblib
from read_data import ReadData
import sys, os, time
from plotting_functions.plotting_functions import IntervalError, PlotVariableImportance, TrainingDeviance, ErrorDistribution

outpref = sys.argv[0] + ' : '

print(outpref + 'Python version ' + sys.version)
print(outpref + 'Pandas version ' + pd.__version__)
print(outpref + 'Sklearn version ' + sk.__version__ )
print(outpref + 'Matplotlib version ' + mpl.__version__)
print(outpref + 'Numpy version ' + np.__version__)

####################################################################
#              Reading of files with physical masses               #
####################################################################

# Read in data (last column is BS)
files = ["~/Jottacloud/data_for_bdt/MSSM_lin_MASS_allsquarks.dat","~/Jottacloud/data_for_bdt/MSSM_log_MASS_allsquarks.dat"]

# Define list with features for MASS dataset
feature_list = ["3.mGluino","4.mdL","5.mdR","6.muL","7.muR","8.msL","9.msR","10.mcL","11.mcR"]
target_list  = ["2.gg_NLO"]
# The data files *_MASS.txt contains a column with NaNs, this must be removed
drop_col = 'Unnamed: 15'

features, target, features_test_M, target_test_M = ReadData(files, feature_list, target_list, drop_col, eps=1E-9, squark_mean=False, train_test=True)

# Set file suffix:
suffix       = "LS_loss"
# Where to save plots
directory       = "plots/"

####################################################################

# Load saved model if it exist
# reg          = joblib.load('BDT_LS_loss.pkl')

####################################################################

if not os.path.exists(directory):
    os.makedirs(directory)

params = {'learning_rate': 0.01, 'loss': 'ls', 'max_depth': 13, 'n_estimators': 5000,
          'random_state': 42, 'subsample': 0.5, 'verbose':1, 'max_leaf_nodes': 100}

reg = ensemble.GradientBoostingRegressor(**params)
print(reg.get_params)

# Fit to data
reg.fit(features, target)

# Save model
joblib.dump(reg, directory + 'BDT_' + suffix + '.pkl')

start_time = time.time()
predicted_ls    = reg.predict(features_test)
print('It took {0:.2f}s to predict {1} samples with LS loss'.format(time.time()-start_time, len(target_test)))

####################################################################
#                       Inspection of model                        #
####################################################################

# Printing R^2 score
print("R^2-score {}: {}".format(suffix, reg.score(features_test,target_test)))

# Plot mean relative deviance for all loss functions in one figure
IntervalError(target_test, predicted_ls, directory, suffix)

# Plot test and training set deviance as function of boosting iterations
plt.clf()
TrainingDeviance(reg, target_test, features_test, suffix, directory)

# List with nicely formatted strings for variable importance plot
feature_list = [r"$m_{\tilde g}$", r"$m_{\tilde d_L}$", r"$m_{\tilde d_R}$", r"$m_{\tilde u_L}$", r"$m_{\tilde u_R}$", r"$m_{\tilde s_L}$", r"$m_{\tilde s_R}$", r"$m_{\tilde c_L}$", r"$m_{\tilde c_R}$", r"$m_-^2$", r"$\beta_{\tilde g}$", r"$L_2$"]
plt.clf()
PlotVariableImportance(reg, feature_list, suffix, directory)