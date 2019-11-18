#chapter 14
import tensorflow as tf

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

# notice here for the batch size dimension, DO NOT specify as None, just leave it as blank!
X = tf.keras.Input(shape=(n_steps, n_inputs,), dtype='float32')
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
rnn_outputs_and_states = tf.keras.layers.SimpleRNN(units=n_neurons, activation='relu', return_state=True)(X)
stacked_rnn_outputs = tf.reshape(rnn_outputs_and_states[0], [-1, n_neurons])
stacked_outputs = tf.keras.layers.Dense(n_outputs)(stacked_rnn_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# generate data
import numpy as np
import matplotlib.pylab as plt

x_cors = np.linspace(0.0, 30.0, num=300)
x_vals = x_cors * np.sin(x_cors) / 3.0 + 2.0 * np.sin(5.0 * x_cors)
#plt.plot(x_cors, x_vals)
#len(x_vals)


# training
n_iterations = 1500
batch_size = 50

def GetNextBatch(x_vals, batch_size, n_steps, n_inputs):
  X_batch = []
  y_batch = []
  for batch in range(batch_size):
    start_pos = np.random.randint(0, len(x_vals) - n_steps - 1)
    x_instance = []
    y_instance = []
    for i in range(start_pos, start_pos + n_steps):
      x_instance.append([x_vals[i]])
      y_instance.append([x_vals[i + 1]])
    X_batch.append(x_instance)
    y_batch.append(y_instance)
  return np.array(X_batch), np.array(y_batch)


with tf.Session() as sess:
  init.run()
  for iteration in range(n_iterations):
    X_batch, y_batch = GetNextBatch(x_vals, batch_size, n_steps, n_inputs)
    #print(y_batch.shape)
    sess.run(training_op, feed_dict={X:X_batch, y:y_batch})
    if iteration % 100 == 0:
      mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
      print(iteration, "\tMSE:", mse)

  X_new, y_new = GetNextBatch(x_vals, 1, n_steps, n_inputs)
  y_pred = sess.run(outputs, feed_dict={X: X_new})
  y_new = np.reshape(y_new, n_steps)
  X_new = np.reshape(X_new, n_steps)
  y_pred = np.reshape(y_pred, n_steps)
  plt.plot(range(len(X_new)), y_new, range(len(X_new)), y_pred)
