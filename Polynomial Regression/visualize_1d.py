import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)
N_TRAIN = 100;
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

for i in range(10,13):
    x_1 = values[:,i]
    x_train = x_1[0:N_TRAIN,:]
    x_test = x_1[N_TRAIN:,:]
    x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    x_ev = np.reshape(x_ev,(x_ev.shape[0],1))
    phi_ev = a1.design_matrix(x_ev, basis='polynomial',degree=3)
    w, _ = a1.linear_regression(x_train, t_train, basis = 'one_polynomial', degree=3)
    y_ev = phi_ev * w
    plt.plot(x_ev,y_ev,'r.-',color='pink')
    plt.plot(x_train,t_train,'ro',color='yellow')
    plt.plot(x_test,t_test,'go',color = 'blue')
    plt.show()
