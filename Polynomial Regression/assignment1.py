import numpy as np
import pandas as pd
import scipy.stats as stats

def load_unicef_data():
    fname = 'SOWC_combined_simple.csv'
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    features = data.axes[1][1:]
    countries = data.values[:,0]
    values = data.values[:,1:]
    values = np.asmatrix(values, dtype='float64')
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return countries, features, values


def normalize_data(x):
    mean = x.mean(0)
    sd = x.std(axis=0)
    return (x - mean)/sd
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0):
    
    if reg_lambda == 0:
        phi = design_matrix(x,basis, degree)
        w = np.matmul(np.linalg.pinv(phi),t)
        y_p = np.transpose(w)*np.transpose(phi)
        t_e = t - np.transpose(y_p)
        train_test_error = np.sqrt(np.mean(np.square(t_e)))
    else:
        phi = design_matrix(x, basis, degree)
        w = np.linalg.inv(phi.T * phi + float(reg_lambda) * np.identity(phi.shape[1])) * phi.T * t
        y_p = np.transpose(w)*np.transpose(phi)
        t_e = t - np.transpose(y_p)
        train_test_error = np.sqrt(np.mean(np.square(t_e)))

    return (w, train_test_error)

def design_matrix(x, basis, degree=0):
    if basis == 'polynomial':
        phi = np.reshape(np.ones(x.shape[0]),(x.shape[0],1))
        for v in range(1,degree+1):
            ap = np.apply_along_axis(np.power,0,x,v)
            phi = np.concatenate((phi,ap),1)
    elif basis == 'one_polynomial':
        phi = np.reshape(np.ones(x.shape[0]),(x.shape[0],1))
        for v in range(1,degree+1):
            ap = np.apply_along_axis(np.power,0,x,v)
            phi = np.concatenate((phi,ap.T),1)
    elif basis == 'ReLU':
        phi = np.reshape(np.ones(x.shape[0]),(x.shape[0],1))
        relu = np.maximum(0,(np.negative(x)+5000))
        phi = np.concatenate((phi,relu),1)
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x, t, w, basis, degree=0):
    phi = design_matrix(x,basis,degree)
    test = np.transpose(w)*np.transpose(phi)
    test_error = np.sqrt(np.mean(np.square(t - np.transpose(test))))
    return test, test_error
