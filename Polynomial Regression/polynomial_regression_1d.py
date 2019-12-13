import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

# Initializing all the arrays and the required items
targets = values[:,1]
x = values[:,7:15]
N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]
train_err = {}
test_err = {}


for val in range(0,x.shape[1]):
    degree=3
    x_train_1 = x_train[:,val]
    x_test_1 = x_test[:,val]
    w, tr_err = a1.linear_regression(x_train_1, t_train, basis='one_polynomial', degree=degree)
    train_err[val] = tr_err
    t_est, te_err = a1.evaluate_regression(x_test_1,t_test, w=w, basis='one_polynomial', degree=degree)
    test_err[val] = te_err


#Plot
plt.bar(np.arange(x.shape[1]), [val for val in train_err.values()],0.1, color='blue')
plt.bar(np.arange(x.shape[1])+0.1, [val for val in test_err.values()],0.1, color='green')
plt.xticks(np.arange(x.shape[1])+0.1,[('F'+str(k)) for k in train_err.keys()])
plt.ylabel('Root Mean Square')
plt.legend(['Train Error','Test Error'])
plt.title('Single Feature Polynomial')
plt.xlabel('Feature')
plt.show()
