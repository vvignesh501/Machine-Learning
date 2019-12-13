import assignment1 as a1
import numpy as np
import scipy
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]
x = values[:, 7:]
x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN, :]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:, :]
t_test = targets[N_TRAIN:]
train_err = []
test_err = []
Lambda = (0, .01, .1, 1, 10, 100, 1000, 100000)

for lambda_1 in Lambda:
    error_test = []
    error_train = []
    start = [k for k in range(0,N_TRAIN,10)]
    end = [k for k in range(10,N_TRAIN+10,10)]

    for i in range(10):
        mod_strt = start[(i+1) % 10]
        mod_err = end[(i+1) % 10]
        x_train_val = np.concatenate((x_train[:mod_strt], x_train[mod_err:]), axis=0)
        t_train_val = np.concatenate((t_train[:mod_strt], t_train[mod_err:]), axis=0)
        x_test_val = x_train[mod_strt:mod_err]
        t_test_val = t_train[mod_strt:mod_err]
        w, train = a1.linear_regression(x_train_val, t_train_val, basis='polynomial', reg_lambda=lambda_1, degree=2)
        y_est ,test = a1.evaluate_regression(x_test_val, t_test_val,w =w, basis='polynomial', degree=2)
        error_test.append(test)
        error_train.append(train)
    test_err.append(np.average(error_test))
    train_err.append(np.average(error_train))

#Plot
plt.semilogx(Lambda, test_err)
plt.title('Graph for Validation Set Error vs Lambda')
plt.show()

