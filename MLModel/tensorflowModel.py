# here is a simple model for predicting house prices depending on size in squre feet and number of bedrooms
# data.txt contains data for training the model with first colums as size and second column as no. bedroom and the
# last column to be the price
# the model is represented by the equation h(x) = theta_1 +  theta_2.x_2 + theta_3.x_3
# the goal is to get the thetas hence the model has learned and can be used for predicting new values
import os
import shutil

import tensorflow as tf
# Scientific and vector computation for python
import numpy as np

# Plotting library
# from matplotlib import pyplot

# Add intercept term to X
def  featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def computeCostMulti(X, y, theta):
    m = y.shape[0]  # number of training examples
    h = np.dot(X, theta)
    J = (1/(2 * m)) * np.sum(np.square(h - y))
    return J

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.shape[0]  # number of training examples
    theta = theta.copy()
    J_history = []
    # h can not be used here as theta is changing
    # h = np.dot(X, theta)
    for i in range(num_iters):
        theta = theta - (alpha / m) * (np.dot(X, theta) - y).dot(X)
        J_history.append(computeCostMulti(X, y, theta))
    return theta, J_history

# test example is -> size = 1650 and rooms = 3 price should be $293081
def createModel(theta):
    graph = tf.Graph()
    if os.path.exists('./model'):
        shutil.rmtree('./model')
    builder = tf.saved_model.builder.SavedModelBuilder('./model')
    with graph.as_default():
        theta_1 = tf.constant(theta[0], name='theta_1')
        theta_2 = tf.constant(theta[1], name='theta_2')
        theta_3 = tf.constant(theta[2], name='theta_3')
        x_1 = tf.placeholder(tf.float64, name='x_1')
        x_2 = tf.placeholder(tf.float64, name='x_2')

        h_x = tf.math.add((theta_1 + theta_2 * x_1), theta_3 * x_2, name='h_x')
        sess = tf.Session()
        # feed the input to the equation
        result = sess.run(h_x, feed_dict={x_1: 1650, x_2: 3})
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        builder.save()
        return result
        # tf.saved_model.save(graph,'./model')

data = np.loadtxt(os.path.join('/Users/Hesham/dev/ml-coursera-python-assignments/MLModel', 'data.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
# print out some data points
print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
print('-'*26)
for i in range(10):
    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))

X_norm, mu, sigma = featureNormalize(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)

alpha = 0.1
num_iters = 400
theta = np.zeros(3)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
# print(theta)

print(createModel(theta))
