import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,10]

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

(w, train_error) = a1.linear_regression(x_train, t_train, basis = 'ReLU')
(t_est, test_error) = a1.evaluate_regression(x_test,t_test, w = w, basis = 'ReLU')
x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.reshape(x_ev,(x_ev.shape[0],1))
phi_ev = a1.design_matrix(x_ev, basis='ReLU')
y_ev = phi_ev * w

print("Training Error:", train_error)
print("Testing Error:", test_error)

plt.plot(x_ev,y_ev,'r.-',color='blue')
plt.plot(x_train,t_train,'go',color='yellow')
plt.plot(x_test,t_test,'bo',color = 'red')
plt.xlabel('Features')
plt.ylabel('Target')
plt.title('ReLU Regression')
plt.show()
