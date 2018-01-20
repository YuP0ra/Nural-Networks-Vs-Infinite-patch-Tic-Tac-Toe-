from libs import tic_judge, subsidiaries
import tensorflow as tf
import numpy as np

# Hyper parameters
RANDOMIZATION_RATE = 0.1
DISCOUNTED_RATE = 0.95
LEARNING_RATE = 0.01
EPISODES_NUM = 2000
HIDDEN_NUM = 81
GAME_SIZE = 9
GAME_COLS = 3

# We start by creating a judge for the game, the jude will give us the reward and till us when the game ends.
rl_judge = tic_judge.Judge(GAME_COLS, GAME_COLS)

# The actions wallet is so important because we'll use it to make sure the network output is valid.
# If not. the wallet will bring us random valid action.
wallet = tic_judge.ActionsWallet(GAME_SIZE)

# Here we define the Network. It's a 1 hidden layer neural network.

# Place holder for the states and actions
x_holder = tf.placeholder(tf.float32, [None, GAME_SIZE])
y_holder = tf.placeholder(tf.float32, [None, GAME_SIZE])

# Network weights and biases.
layer1_weights = tf.Variable(tf.random_normal([GAME_SIZE, HIDDEN_NUM]))
layer1_biases = tf.Variable(tf.random_normal([HIDDEN_NUM]))
layer2_weights = tf.Variable(tf.random_normal([HIDDEN_NUM, GAME_SIZE]))
layer2_biases = tf.Variable(tf.random_normal([GAME_SIZE]))


# The FeedForward part of the network.
def feed_forward(input_data):
    layer1_output = tf.nn.relu(tf.matmul(input_data, layer1_weights) + layer1_biases)
    layer2_output = tf.matmul(layer1_output, layer2_weights) + layer2_biases
    return layer2_output


# The actions space of the network, Losses and the Optimizer.
actions = feed_forward(x_holder)
arg_max_actions = tf.argmax(actions, 1)
loss = tf.losses.mean_squared_error(predictions=actions, labels=y_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
update = optimizer.minimize(loss)

print('Going Into the Session')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    wins = 0

    # The main loop where we:
    # 1: Play a game till the end
    # 2: Sum the rewards and discounted reward
    # 3: back prop the network to update the new weights

    for i in range(EPISODES_NUM):
        # new board and wallet to keep a clean record
        rl_judge.initialize_new_board()
        wallet.initialize_new_wallet()

        # Buffer to hold all the network output to be updated when the game is finished
        actions_buffer = np.zeros(shape=(GAME_SIZE + 1, GAME_SIZE), dtype=np.float32)
        arg_max_buffer = np.zeros([GAME_SIZE], dtype=np.int32)

        # Same for the action buffer
        states_buffer = np.zeros(shape=(GAME_SIZE + 1, GAME_SIZE), dtype=np.float32)

        # Temp game board, we'll keep updating it and feed it to the network
        temp_board = np.zeros(shape=(1, GAME_SIZE), dtype=np.float32)

        for j in range(GAME_SIZE):
            # Check if the game reached the end
            if j >= GAME_SIZE // 2:
                break

            # Run a feedForward to get the actions space
            all_actions, max_index = sess.run([actions, arg_max_actions], feed_dict={x_holder: temp_board})

            # TESTING
            if i % 100 == 0:
                print(all_actions)

            # Check is the action is valid. If not pick random action.
            if wallet.valid_action(max_index[0]) is False:
                chosen_action = wallet.get_random_action()
            else:
                # Make sure the network chose some random actions from a time to time.
                # That's must be done so we won't stuck on some fixed path.
                if np.random.ranf() < RANDOMIZATION_RATE:
                    chosen_action = wallet.get_random_action()
                else:
                    chosen_action = max_index[0]
                    wallet.remove_action(max_index[0])

            # Add the state and result actions to the buffers
            states_buffer[j] = temp_board[0]
            actions_buffer[j] = all_actions
            arg_max_buffer[j] = chosen_action

            # Keep tracking of the temp board
            temp_board[0, chosen_action] = 1

            # Feed the action to the jude and wait for reward
            # The jude take and index in which we play. 1 means X
            reward = rl_judge.step(max_index, 1)

            # Game ended. Do the next:
            # 1. Calculate the discounted reward
            # 2. Train with the new data
            # 3. Break the loop

            if reward == 1:
                wins += 1

                one_hot = np.zeros(GAME_SIZE)
                one_hot_reward = one_hot[arg_max_buffer[j]] = 1
                actions_buffer[j + 1] = one_hot_reward

                # Calculate the discounted reward. Q(t) = Q(t) + Y * Max:Q(t + 1).
                for k in reversed(range(j)):
                    r_ = actions_buffer[k + 1, arg_max_buffer[k] + 1]
                    actions_buffer[k, arg_max_buffer[k]] += DISCOUNTED_RATE * r_

                # BackProp
                _ = sess.run([update], feed_dict={x_holder: states_buffer[:j], y_holder: actions_buffer[:j]})
                break

            # Random bot turn
            bot_action = wallet.get_random_action()
            reward = rl_judge.step(bot_action, -1)

            # Keep tracking of the temp board
            temp_board[0, bot_action] = -1

            if reward == -1:
                one_hot = np.zeros(GAME_SIZE)
                one_hot_reward = one_hot[arg_max_buffer[j]] = -1
                actions_buffer[j + 1] = one_hot_reward

                # Calculate the discounted reward. Q(t) = Q(t) + Y * Max:Q(t + 1).
                for k in reversed(range(j)):
                    r_ = actions_buffer[k + 1, arg_max_buffer[k + 1]]
                    actions_buffer[k, arg_max_buffer[k]] += DISCOUNTED_RATE * r_

                # BackProp
                _ = sess.run([update], feed_dict={x_holder: states_buffer[:j], y_holder: actions_buffer[:j]})
                break

        if i % 100 == 0:
            print('At Ep:\t', i, 'Wins Are:\t', wins)
            wins = 0
