import gym
import numpy as np
import tensorflow as tf
#import keras

env = gym.make("CartPole-v0")
obs = env.reset()


# define neural network
n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)

#X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#hidden = keras.layers.Dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
#logits = keras.layers.Dense(hidden, n_outputs, kernel_initializer=initializer)
#outputs = tf.nn.sigmoid(logits)


#keras model
X = tf.keras.layers.Input(shape=(n_inputs,), dtype='float32')
hidden = tf.keras.layers.Dense(n_hidden, activation='elu', kernel_initializer=initializer)(X)
predictions = tf.keras.layers.Dense(n_outputs, activation='sigmoid', kernel_initializer=initializer)(hidden)

model = tf.keras.models.Model(inputs=X, outputs=predictions)

p_left_and_right = tf.concat(axis=1, values=[predictions, 1 - predictions])
action = tf.random.categorical(tf.math.log(p_left_and_right), num_samples=1)
init = tf.global_variables_initializer()

y = tf.constant(1.0, dtype='float32') - tf.cast(action, tf.float32)

cross_entropy = tf.keras.losses.binary_crossentropy #y, predictions

lrate = 0.01

optimizer = tf.keras.optimizers.Adam(lr=lrate)

model.compile(loss=cross_entropy, target_tensors=y, optimizer=optimizer)

grads = optimizer.get_gradients(model.total_loss, model.trainable_weights)
grads_and_vars = zip(grads, model.trainable_weights)

gradient_placeholders = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars:
  gradient_placeholder = tf.placeholder(tf.float32, shape = grad.get_shape())
  gradient_placeholders.append(gradient_placeholder)
  grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = model.optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()



def discount_rewards(rewards, discount_rate):
  discounted_rewards = np.zeros(len(rewards))
  cumulative_rewards = 0
  for step in reversed(range(len(rewards))):
    cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
    discounted_rewards[step] = cumulative_rewards
  return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
  all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
  flat_rewards = np.concatenate(all_discounted_rewards)
  reward_mean = flat_rewards.mean()
  reward_std = flat_rewards.std()
  return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
  
  
n_iterations = 1 #500
n_max_steps = 1000
n_games_per_update = 10
save_iterations = 10
discount_rate = 0.95

with tf.Session() as sess:
  init.run()
  for iteration in range(n_iterations):
    all_rewards = []
    all_gradients = []
    for game in range(n_games_per_update):
      current_rewards = [] # list of reward for a single epside, one reward value for each step
      current_gradients = []
      obs = env.reset()
      for step in range(n_max_steps):
        action_val, gradient_val = sess.run([action, grads], feed_dict={X:obs.reshape(1, n_inputs)})
        obs, reward, done, info = env.step(action_val[0][0])
        current_rewards.append(reward)
        current_gradients.append(gradient_val)
        if done:
          break
      all_rewards.append(current_rewards) # rewards per step for 10 games
      all_gradients.append(current_gradients)

    all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
    feed_dict = {}
    for var_ind, gradient_placeholder in enumerate(gradient_placeholders):
      mean_gradients = np.mean(
          [reward * all_gradients[game_ind][step][var_ind]
              for game_ind, rewards in enumerate(all_rewards)
              for step, reward in enumerate(rewards)
           ],
           axis=0, # averge over the number of games
      )
      feed_dict[gradient_placeholder] = mean_gradients

    sess.run(training_op, feed_dict=feed_dict)

  # now let us try out our NN policy!!
  totals = []

  for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):
      action_val = sess.run(action, feed_dict={X:obs.reshape(1, n_inputs)})
      obs, reward, done, info = env.step(action_val[0][0])
      episode_rewards += reward
      if done:
        break
    totals.append(episode_rewards)

  print('max step:' + str(np.max(totals)) + ',mean:' + str(np.mean(totals)))
