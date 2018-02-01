#!/usr/bin/env python2

"""Plotting functions for plotting performance of BDT

2018, Jon Vegard Sparre <jonvsp@fys.uio.no>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
import matplotlib.ticker as ticker
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(**params)
# Setting plot style
plt.rcParams['font.family'] = 'sans-serif'

def IntervalError(target_test, predicted, directory, suffix, offset=0):
    """Plot distribution of error in decades of xsec
    target_test

    :param target_test: log 10 of true values 
    :type target_test: array
    :param predicted: log10 of predicted values
    :type predicted: array
    :param directory: name of plot directory 
    :type directory: str
    :param suffix: suffix for labelling plots 
    :type suffix: str
    :param offset: offset if several graphs in one figure 
    :type offset: float

    :returns: 0

    Usage:

    >>> IntervalError(target_test, predicted, directory, suffix, offset=0)

    """

    intervals   = []
    labelliste  = []
    start       = int(min(target_test))-1 # Ensure that smallest xsec is included
    stop        = int(max(target_test))+1 # Ensure that largest xsec is included

    for i in range(start,stop):    
        intervals.append([i,i+1])
    
    intervals  = np.array(intervals)
    # Creating lists for saving interesting numbers
    rel_deviance_list = []
    mean_list         = []
    std_list          = []
    no_points         = []
    # Loop over all intervals
    for i in range(0,len(intervals)):
        # Picking out samples with xsec inside the current interval and calculating the error
        index     = np.logical_and(target_test > intervals[i][0], target_test <= intervals[i][1])
        # Check if there's any samples in current interval
        if np.any(index):
            labelliste.append('({0} {1}]'.format(intervals[i][0],intervals[i][1]))
            x_true       = np.power(10,target_test[index])
            x_BDT        = np.power(10,predicted[index])
            rel_deviance = (x_BDT-x_true)/(x_true)
            # Adding values to list
            rel_dev_mean = np.mean(rel_deviance)
            rel_dev_std  = np.std(rel_deviance)
            rel_deviance = rel_deviance.tolist()
            rel_deviance_list.append(rel_deviance)
            mean_list.append(rel_dev_mean)
            std_list.append(rel_dev_std)
            no_points.append(len(rel_deviance))
    print('Interval: Mean rel. deviance')
    print('\n'.join('{0:9}: {1:.6f}'.format(str(i[1]),i[0]) for k, i in enumerate(zip(mean_list,labelliste))))
    print('Interval: Std rel. deviance')
    print('\n'.join('{0:9}: {1:.6f}'.format(str(i[1]),i[0]) for k, i in enumerate(zip(std_list,intervals))))
    print('Interval: No. of samples')
    print('\n'.join('{0:9}: {1}'.format(str(i[1]),i[0]) for k, i in enumerate(zip(no_points,labelliste))))
    
    # Changing first element of labellist to point out that these are the xsec=0 samples
    labelliste[0] = r'$\sigma$ = 0'
    # Plotting mean error scatter plot with std error as error bars as function of decades
    plt.axvline(7-0.2, ls='--', c='C4') # Vertical line at N=0.02 events.
    plt.errorbar(np.arange(len(mean_list))+offset, mean_list, yerr=std_list, fmt='o', label=suffix, markersize=4)
    plt.ylim(-.4,.401)
    plt.xticks(np.arange(len(mean_list)), labelliste, rotation='vertical')
    plt.ylabel(r'$\bar\epsilon$',size=20)
    plt.xlabel(r'$\log_{10}(\sigma/\sigma_0),\quad \sigma_0=1$ fb',size=15)
    plt.tight_layout(pad=2.)
    plt.legend(loc='upper right')
    plt.grid('on')
    plt.savefig(directory + 'mean_rel_dev_scatter.pdf')
    return 0

def TrainingDeviance(reg, target_test, features_test, suffix, directory, comp=False):
    '''Function for plotting deviance plot for LS, LAD, and Huber loss functions.

    :param reg: BDT-model 
    :type reg: predictor
    :param target_test: target test values
    :type target_test: array
    :param features_test: features test values
    :type features_test: array
    :param suffix: info about which model are plotted
    :type suffix: str
    :param directory: path to plot folder
    :type directory: str

    :returns: 0

    '''
    
    # Compute test set deviance
    params = reg.get_params()
    test_score = np.zeros( (params['n_estimators'],) )
    for i, y_pred in enumerate( reg.staged_predict(features_test) ):
        test_score[i] = reg.loss_(target_test, y_pred)

    if params['loss'] == 'ls':
        train_score = reg.train_score_
        color1 = '-C0'
        color2 = '--C0'
    elif params['loss'] == 'lad':
        # Squaring the linear loss before plotting it
        # as discussed in the thesis.
        train_score = reg.train_score_**2
        test_score  = test_score**2
        color1 = '-C1'
        color2 = '--C1'
    elif params['loss'] == 'huber':
        # Multiplying Huber loss by two
        # as discussed in thesis
        train_score = reg.train_score_*2
        test_score  = test_score*2
        color1 = '-C2'
        color2 = '--C2'
    # Plotting
    plt.plot(np.arange(params['n_estimators']) + 1, train_score, color1,
             label='{}'.format(suffix))
    plt.plot(np.arange(params['n_estimators']) + 1, test_score, color2,
             label='{}'.format(suffix))
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.yscale('log')
    plt.tight_layout(pad=2.)
    plt.savefig(directory + 'deviance.pdf')
    return 0

def PlotVariableImportance(reg, feature_list, suffix, directory):
    '''Function for plotting feature importance for BDT models.

    :param reg: BDT-model 
    :type reg: predictor
    :param target_test: true values
    :type target_test: array
    :param features_test: feature values
    :type features_test: array
    :param suffix: info about which model are plotted
    :type suffix: str
    :param directory:path to plot folder
    :type directory: str
    '''
    plt.clf()
    feature_importance = reg.feature_importances_
    # Scale importances relative to total
    feature_importance = 100.0 * (feature_importance / feature_importance.sum())
    sorted_importance = np.argsort(feature_importance)
    feature_list = np.array(feature_list)
    pos = np.arange(sorted_importance.shape[0]) + .5
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.2)
    ax.barh(pos, feature_importance[sorted_importance], align='center')
    plt.yticks(pos, feature_list[sorted_importance],size=18)
    plt.xlabel('Relative Importance')
    plt.savefig(directory + 'variable_importance'+suffix+'.pdf')
    return 0

def ErrorDistribution(target, predicted, directory, suffix):
    """Plot error distribution between a predictive model and
    Prospino results.

    """
    predicted = np.power(10, predicted)
    target    = np.power(10, target)
    rel_dev   = (predicted - target)/(target)

    plt.clf()
    y, x, _ = plt.hist(rel_dev,histtype='step', bins=60, range=(-1.,1.), normed=1,alpha=1., lw=2)
    plt.xticks(np.arange(-1,1.1,0.5), np.arange(-1,1.1,0.5),size=20)
    plt.yticks(np.arange(0,y.max(),4), np.arange(0,y.max(),4),size=20)
    plt.xlabel('$(x_{pred} - x_{true})/x_{true}$', size=20)
    plt.ylabel('Normed to 1', size=20)
    plt.tight_layout(pad=1.)
    plt.savefig(directory + 'error_dist'+ suffix + '.pdf')