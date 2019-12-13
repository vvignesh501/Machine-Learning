import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()
N_TRAIN = 100;
x = values[:,7:]
targets = values[:,1]
training = x[0:N_TRAIN,:]
reg_test = x[N_TRAIN:,:]
testing = targets[0:N_TRAIN]
test = targets[N_TRAIN:]
train_err = {}
test_err = {}

for val in range(1,7):
    w, tr_err = a1.linear_regression(training, testing, basis='polynomial', degree=val)
    train_err[val] = tr_err
    test, test_error = a1.evaluate_regression(reg_test,test,w=w, basis='polynomial', degree=val)
    test_err[val] = test_error

plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('Root Mean Square')
plt.legend(['Test Error', 'Train Error'])
plt.title('Fit with polynomials, no Regularization')
plt.xlabel('Polynomial Degree')
plt.show()


x = a1.normalize_data(x)
training = x[0:N_TRAIN,:]
reg_test = x[N_TRAIN:,:]
testing = targets[0:N_TRAIN]
test = targets[N_TRAIN:]
train_err = {}
test_err = {}

for val in range(1, 7):
    (w, tr_err) = a1.linear_regression(training, testing, basis='polynomial', degree=val)
    train_err[val] = tr_err
    (test, test_error) = a1.evaluate_regression(reg_test,test, w=w, basis='polynomial', degree=val)
    test_err[val] = test_error
plt.plot(train_err.keys(), train_err.values())
plt.plot(test_err.keys(), test_err.values())
plt.ylabel('Root Mean Square')
plt.legend(['Test Error', 'Train Error'])
plt.title('Fit with polynomials, no Regularization')
plt.xlabel('Polynomial Degree')
plt.show()
