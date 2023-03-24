import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random

alpha = 0.05
tau1 = alpha/2
tau2 = 1 - alpha/2

u = np.linspace(-0.5,1,100)

def pinball(u,tau):
    y = np.empty(u.shape)
    u_lower = u < 0
    u_upper = u >= 0
    y[u_lower] = u[u_lower]*(tau - 1)
    y[u_upper] = u[u_upper]*tau
    return y

def multiple_pinball(alpha, y_true, y_pred):
    # hyperparameters
    alpha_lower = alpha / 2  # for pinball
    alpha_upper = 1 - alpha_lower

    y_true = y_true[:, 0]
    y_l = y_pred[:, 0]
    y_u = y_pred[:, 1]
    y_m = y_pred[:, 2]

    error_l = tf.subtract(y_true, y_l)
    error_m = tf.subtract(y_true, y_m)
    error_u = tf.subtract(y_true, y_u)
    lower_sum = tf.maximum(alpha_lower * error_l, (alpha_lower - 1) * error_l)
    middle_sum = tf.maximum(0.5 * error_m, -0.5 * error_m)
    upper_sum = tf.maximum(alpha_upper * error_u, (alpha_upper - 1) * error_u)
    return tf.reduce_mean((lower_sum + middle_sum + upper_sum) / 3, axis=-1)

def func (x):
    return x*2 +3

y = func(u)
y_ = y + np.random.normal(0,0.1,len(y))

res = multiple_pinball(alpha,y,y_)
print(res)
print(res.shape)

plt.figure(figsize=(24,12))
plt.subplot(2,2,1)
plt.plot(u,y_)
plt.subplot(2,2,2)
plt.plot(u,res)

plt.show()
