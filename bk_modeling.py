import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sklearn.linear_model as linear
import scipy.stats as stats
from collections import defaultdict
import random

pd.options.display.float_format = '{:,.2f}'.format
sns.set(style="whitegrid")

def adjusted_r_squared(result):
    if (result['len'] - len(result['coeff']) - 1) > 1:
        other = (result['len'] - 1)/(result['len'] - len(result['coeff'])-1)
    else:
        other = (result['len'] - 1)
    return 1 - (1 - result["r_squared"]) * other

def get_result(X, y, model, X_labels):
    result = {}
    result['x_labels'] = X_labels
    result['len'] = len(y)
    result['y'] = y
    result['model'] = model
    result['y_hat'] = model.predict(X)
    result['residuals'] = y - result['y_hat']   
    
    result['intercept'] = model.intercept_
    result['coeff'] = model.coef_
    
    result['r_squared'] = model.score(X, y)
    result['adj_r_squared'] = adjusted_r_squared(result)
    sum_squared_error = sum([d**2 for d in result[ "residuals"]])
    if (result['len'] - len(result['coeff'])) > 1:
        result['sigma'] = np.sqrt( sum_squared_error / (result['len'] - len(result['coeff'])))
    else:
        result['sigma'] = np.sqrt( sum_squared_error / 1)
    
    return result

def linear_reg(data, X_labels, y_label):
    X = data[X_labels].values
    y = data[y_label].values
    model = linear.LinearRegression().fit(X, y)
    
    result = get_result(X, y, model, X_labels)   
    
    return result

def bci_df(df):
    result = []
    
    for column in df.columns:
        result.append(stats.mstats.mquantiles(df[column], [0.025, 0.975]))
        
    return result

def bootstrap_lr_values(data, X_labels, y_label, samples):
    result = {}
    
    r_sqs = []
    adj_r_sqs = []
    sigmas = []
    intercepts = []
    coeffs = []
    
    for i in range(samples):
        sampling = data.sample(len(data), replace=True)
        temp_result = linear_reg(sampling, X_labels, y_label)
        
        r_sqs.append(temp_result['r_squared'])
        adj_r_sqs.append(temp_result['adj_r_squared'])
        sigmas.append(temp_result['sigma'])
        intercepts.append(temp_result['intercept'])
        coeffs.append(temp_result['coeff'])
    
    result['bci_intercept'] = stats.mstats.mquantiles(intercepts, [0.025, 0.975])  
    result['bci_coeff'] = bci_df(pd.DataFrame(coeffs, columns=X_labels))
    result['bci_r_squared'] = stats.mstats.mquantiles(r_sqs, [0.025, 0.975])
    result['bci_adj_r_squared'] = stats.mstats.mquantiles(adj_r_sqs, [0.025, 0.975])
    result['bci_sigma'] = stats.mstats.mquantiles(sigmas, [0.025, 0.975])
    
    return result

def bootstrap_linear_reg(data, X_labels, y_label, samples=100):
    result = {}
    
    X = data[X_labels].values
    y = data[y_label].values
    model = linear.LinearRegression().fit(X, y)
    
    single_results = get_result(X, y, model, X_labels)
    
    for key in single_results:
        result[key] = single_results[key]
        
    bootstrap_results = bootstrap_lr_values(data, X_labels, y_label, samples)
    
    for key in bootstrap_results:
        result[key] = bootstrap_results[key]
    
    return result

def output_results(result):
    column1 = np.concatenate([['intercept'], result['x_labels'], ['$R^2$', '$\bar{R}^2$', '$\sigma$']])
    column2 = np.concatenate([[result['intercept']], result['coeff'],
                              [result['r_squared'], result['adj_r_squared'], result['sigma']]])
        
    return pd.DataFrame(data = {'Labels': column1, 'Values': column2})

def output_bootstrap_results(result):
    column1 = np.concatenate([['intercept'], result['x_labels'], ['$R^2$', '$Adj {R}^2$', '$\sigma$']])
    column2 = np.concatenate([[result['intercept']], result['coeff'],
                              [result['r_squared'], result['adj_r_squared'], result['sigma']]])
    
    coeff_low = [x[0] for x in result['bci_coeff']]
    coeff_high = [x[1] for x in result['bci_coeff']]
    
    column3 = np.concatenate([[result['bci_intercept'][0]], coeff_low,
                              [result['bci_r_squared'][0], result['bci_adj_r_squared'][0], result['bci_sigma'][0]]])
    column4 = np.concatenate([[result['bci_intercept'][1]], coeff_high,
                              [result['bci_r_squared'][1], result['bci_adj_r_squared'][1], result['bci_sigma'][1]]])
    
        
    return pd.DataFrame(data = {'Labels': column1, 'Values': column2,
                                '95% BCI Low': column3, '95% BCI High': column4})

def plot_residuals(data, variables, result, shape, size=(20,7)):
    figure = plt.figure(figsize=size)
    
    plots = len(variables)
    
    for i, variable in enumerate(variables):
        axes = figure.add_subplot(shape[0], shape[1], i+1)
        
        keyed_values = sorted(zip(data[variable].values, result['residuals']), key=lambda x: x[0])
        residuals = [x[1] for x in keyed_values]
    
        axes.plot(list(range(0, result['len'])), residuals, '.', color='dimgray', alpha=0.75)
        axes.axhline(y=0.0, xmin=0, xmax=result['len'], c='firebrick', alpha=0.5)
        axes.set_title( variable + ' v. residuals', fontsize=16)

    plt.tight_layout()
    plt.show()
    plt.close()

def folding(indexs, n):
    k, m = divmod(len(indexs), n)
    return [indexs[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def cross_validation(data, X_labels, y_label, evaluate, fold_count=10, repetitions=3, seed=False):
    if seed!=False:
        random.seed(seed)
        
    indices = list(range(len( data)))
    metrics = []
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = folding(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[fold]
            train_indices = [idx not in fold for idx in indices]
            train_data = data.iloc[train_indices]
            
            result = linear_reg(train_data, X_labels, y_label)
            model = result['model']
            
            X_test = test_data[X_labels].values
            y_test = test_data[y_label].values
            test_result = get_result(X_test, y_test, model, X_labels)
            metric = evaluate(test_result)
            metrics.append(metric)
            
    return metrics

def resample(data):
    n = len(data)
    return [data[ i] for i in [stats.randint.rvs(0, n - 1) for _ in range( 0, n)]]

def data_structures():
    result = dict()
    result['train'] = defaultdict( list)
    result['test'] = defaultdict( list)
    return result

def validation_curves_features(data, X_labels, y_label, features, evaluate, fold_count=10, repetitions=3, seed=False):
    if seed!=False:
        random.seed(seed)
        
    indices = list(range(len( data)))
    results = data_structures()
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = folding(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[ fold]
            train_indices = [idx for idx in indices if idx not in fold]
            train_data = data.iloc[train_indices]
            
            for k in features:
                holder = X_labels + k
                
                result = linear_reg(train_data, holder, y_label)
                model = result['model']

                metric = evaluate(result)
                results['train'][k[0]].append(metric)

                X_test = test_data[holder].values
                y_test = test_data[y_label].values
                test_result = get_result(X_test, y_test, model, holder)
                metric = evaluate(test_result)
                results['test'][k[0]].append(metric)

    statistics = {}
    for k, v in results["train"].items():
        statistics[ k] = (np.mean(v), np.std(v))
    results["train"] = statistics
    statistics = {}
    for k, v in results["test"].items():
        statistics[ k] = (np.mean(v), np.std(v))
    results["test"] = statistics
    return results

def results_to_curves(curve, results):
    all_statistics = results[curve]
    keys = list( all_statistics.keys())
    keys.sort()
    mean = []
    upper = []
    lower = []
    for k in keys:
        m, s = all_statistics[ k]
        mean.append( m)
        upper.append( m + 2 * s)
        lower.append( m - 2 * s)
    return keys, lower, mean, upper

def plot_validation_curves(results, metric, parameter, values, zoom=False):
    figure = plt.figure(figsize=(10,6))

    axes = figure.add_subplot(1, 1, 1)

    xs, train_lower, train_mean, train_upper = results_to_curves( "train", results)
    _, test_lower, test_mean, test_upper = results_to_curves( "test", results)

    axes.plot( values, train_mean, color="steelblue")
    axes.fill_between( values, train_upper, train_lower, color="steelblue", alpha=0.25, label="train")
    axes.plot( values, test_mean, color="firebrick")
    axes.fill_between( values, test_upper, test_lower, color="firebrick", alpha=0.25, label="test")
    axes.legend()
    axes.set_xticks(values)
    axes.set_xlabel( parameter)
    axes.set_ylabel( metric)
    axes.set_title("Validation Curves")

    if zoom==True:
        y_lower = ( 0.8 * test_lower[-1])
        y_upper = ( 1.2 * test_upper[-1])
        axes.set_ylim((y_lower, y_upper))
    elif type(zoom)==tuple:
        axes.set_ylim((zoom[0], zoom[1]))

    plt.show()
    plt.close()
  
def learning_curves(data, X_labels, y_label, evaluate, fold_count=10, repetitions=3, increment=5, seed=False):
    if seed!=False:
        random.seed(seed)
        
    indices = list(range(len( data)))
    results = data_structures()
    for _ in range(repetitions):
        random.shuffle(indices)
        folds = folding(indices, fold_count)
        for fold in folds:
            test_data = data.iloc[ fold]
            train_indices = [idx for idx in indices if idx not in fold]
            train_data = data.iloc[train_indices]
            
            for i in list(range(increment, 100, increment)) + [100]:
                train_chunk_size = int( np.ceil((i/100)*len( train_indices)))
                train_data_chunk = data.iloc[train_indices[0:train_chunk_size]]

                result = linear_reg(train_data_chunk, X_labels, y_label)
                model = result['model']

                metric = evaluate(result)
                results['train'][i].append(metric)

                X_test = test_data[X_labels].values
                y_test = test_data[y_label].values
                test_result = get_result(X_test, y_test, model, X_labels)
                metric = evaluate(test_result)
                results['test'][i].append(metric)

    statistics = {}
    for k, v in results["train"].items():
        statistics[ k] = (np.mean(v), np.std(v))
    results["train"] = statistics
    statistics = {}
    for k, v in results["test"].items():
        statistics[ k] = (np.mean(v), np.std(v))
    results["test"] = statistics
    return results

def plot_learning_curves( results, metric, desired=None, zoom=False, credible=True):
    figure = plt.figure(figsize=(10,6))

    axes = figure.add_subplot(1, 1, 1)

    xs, train_lower, train_mean, train_upper = results_to_curves( "train", results)
    _, test_lower, test_mean, test_upper = results_to_curves( "test", results)

    axes.plot( xs, train_mean, color="steelblue", label="train")
    axes.plot( xs, test_mean, color="firebrick", label="test")
    if credible:
        axes.fill_between( xs, train_upper, train_lower, color="steelblue", alpha=0.25)
        axes.fill_between( xs, test_upper, test_lower, color="firebrick", alpha=0.25)
    
    if desired:
        if type(desired) is tuple:
            axes.axhline((desired[0] + desired[1])/2.0, color="gold", label="desired")
            axes.fill_between( xs, desired[1], desired[0], color="gold", alpha=0.25)
        else:
            axes.axhline( desired, color="gold", label="desired")
    
    axes.legend()
    axes.set_xlabel( "training set (%)")
    axes.set_ylabel( metric)
    axes.set_title("Learning Curves")

    if zoom==True:
        y_lower = ( 0.8 * test_lower[-1])
        y_upper = ( 1.2 * test_upper[-1])
        axes.set_ylim((y_lower, y_upper))
    elif type(zoom)==tuple:
        axes.set_ylim((zoom[0], zoom[1]))

    plt.show()
    plt.close()

def predict(result, value):
    predictions = []
    
    for i in value:
        predictions.append(result['model'].predict(value))

    return predictions
