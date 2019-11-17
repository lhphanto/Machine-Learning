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

# training
