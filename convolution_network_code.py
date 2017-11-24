# As usual, a bit of setup
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.cnn import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient
from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.solver import Solver

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython

def rel_error(x, y):
	""" returns relative error """
	return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

data = get_CIFAR10_data()
for k, v in data.items():
	print('%s: ' % k, v.shape)

num_train = 100
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}



# model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)

# solver = Solver(model, data,
#                 num_epochs=1, batch_size=50,
#                 update_rule='adam',
#                 optim_config={
#                   'learning_rate': 1e-3,
#                 },
#                 verbose=True, print_every=20)
# solver.train()



# model = ThreeLayerConvNet(use_batchnorm = True)

# N = 50
# X = np.random.randn(N, 3, 32, 32)
# y = np.random.randint(10, size=N)

# loss, grads = model.loss(X, y)
# print('Initial loss (no regularization): ', loss)

# model.reg = 0.5
# loss, grads = model.loss(X, y)
# print('Initial loss (with regularization): ', loss)

# num_inputs = 2
# input_dim = (3, 10, 10)
# reg = 0.0
# num_classes = 10
# X = np.random.randn(num_inputs, *input_dim)
# y = np.random.randint(num_classes, size=num_inputs)

# model = ThreeLayerConvNet(num_filters=3, filter_size=3,
#                           input_dim=input_dim, hidden_dim=7,
#                           dtype=np.float64,use_batchnorm = True)
# loss, grads = model.loss(X, y)




input_dim = [3, 32, 32]

model = FirstConvNet(num_filters=[16, 32, 64, 128], filter_size=3,
                          input_dim=input_dim, hidden_dims=[256, 256],
                          dtype=np.float64,use_batchnorm = True, reg=0.05, weight_scale=0.05)

solver = Solver(model, data,
                num_epochs=20, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()

y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())


model2 = FirstConvNet(num_filters=[16, 32, 64, 128], filter_size=3,
                          input_dim=input_dim, hidden_dims=[500, 500],
                          dtype=np.float64,use_batchnorm = True, reg=0.05, weight_scale=0.05)

solver2 = Solver(model2, data,
                num_epochs=5, batch_size=50,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver2.train()


y_test_pred = np.argmax(model2.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(model2.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

scores = model.loss(data['X_test'])
scores2 = model2.loss(data['X_test'])

avg_score = (scores+scores2)/2

y_test_pred = np.argmax(avg_score, axis=1)
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
