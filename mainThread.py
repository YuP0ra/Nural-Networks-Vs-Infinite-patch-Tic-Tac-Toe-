from libs import tic_judge, RL_Policies
import tensorflow as tf
import numpy as np

# Hyper parameters
RANDOMIZATION_RATE = 1.
DISCOUNTED_RATE = 0.95
LEARNING_RATE = 0.001
EPISODES_NUM = 1000000
HIDDEN_NUM = 81
GAME_SIZE = 9
GAME_COLS = 3

# We start by creating a judge for the game, the jude will give us the reward and till us when the game ends.
rl_judge = tic_judge.Judge(GAME_COLS, GAME_COLS)

# The actions wallet is so important because we'll use it to make sure the network output is valid.
# If not. the wallet will bring us random valid action.
wallet = tic_judge.ActionsWallet(GAME_SIZE)

# Here we define the Network. It's a 1 hidden layer neural network.

# Place holder for the states, wights and rewards
x_holder = tf.placeholder(tf.float32, [None, GAME_SIZE])
y_holder = tf.placeholder(tf.float32, [None, GAME_SIZE])


# Network weights and biases.
layer1_weights = tf.Variable(tf.random_normal([GAME_SIZE, HIDDEN_NUM]))
layer1_biases = tf.Variable(tf.random_normal([HIDDEN_NUM]))
layer2_weights = tf.Variable(tf.random_normal([HIDDEN_NUM, HIDDEN_NUM]))
layer2_biases = tf.Variable(tf.random_normal([HIDDEN_NUM]))
layer3_weights = tf.Variable(tf.random_normal([HIDDEN_NUM, GAME_SIZE]))
layer3_biases = tf.Variable(tf.random_normal([GAME_SIZE]))


# The FeedForward part of the network.
def feed_forward(input_data):
    layer1_output = tf.nn.relu(tf.matmul(input_data, layer1_weights) + layer1_biases)
    layer2_output = tf.nn.relu(tf.matmul(layer1_output, layer2_weights) + layer2_biases)
    layer3_output = tf.matmul(layer2_output, layer3_weights) + layer3_biases
    return layer3_output


# The actions space of the network, Losses and the Optimizer.
prediction = feed_forward(x_holder)
loss = tf.losses.mean_squared_error(predictions=prediction, labels=y_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
update = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    index, wins = 0, 1

    # The main loop where we:
    # 1: Play a game till the end
    # 2: Sum the rewards and discounted reward
    # 3: back prop the network to update the new weights

    long_term_states = np.zeros(shape=(1000 * GAME_SIZE, GAME_SIZE))
    long_term_values = np.zeros(shape=(1000 * GAME_SIZE, GAME_SIZE))

    for i in range(EPISODES_NUM):
        # new board and wallet to keep a clean record
        rl_judge.initialize_new_board()
        wallet.initialize_new_wallet()

        # Same for the action buffer
        states_buffer = np.zeros(shape=(GAME_SIZE, GAME_SIZE), dtype=np.float32)
        action_buffer = np.zeros(shape=(GAME_SIZE, GAME_SIZE), dtype=np.float32)
        actions = np.zeros(GAME_SIZE, dtype=int)
        rewards = np.zeros(GAME_SIZE, dtype=int)
        # Temp game board, we'll keep updating it and feed it to the network
        temp_board = np.zeros(shape=(1, GAME_SIZE), dtype=np.float32)

        for j in range(GAME_SIZE):
            # Run a feedForward to get the actions space
            network_q_output = sess.run([prediction], feed_dict={x_holder: temp_board})
            chosen_action = np.argmax(network_q_output[0])

            # Check is the action is valid. If not pick random action.
            if wallet.valid_action(chosen_action) is False:
                chosen_action = wallet.get_random_action()
            else:
                # Make sure the network chose some random actions from a time to time.
                # That's must be done so we won't stuck on some fixed path.
                if np.random.ranf() < RANDOMIZATION_RATE:
                    chosen_action = wallet.get_random_action()
                else:
                    wallet.remove_action(chosen_action)

            # Add the state to the buffers
            rewards[j] = -0.1
            actions[j] = chosen_action
            states_buffer[j] = temp_board[0]
            action_buffer[j] = network_q_output[0]

            """ -------------------------------------------------------------------------------------------------------
                                                        Network Turn 
             ------------------------------------------------------------------------------------------------------- """
            temp_board[0, chosen_action] = 1
            reward = rl_judge.step(chosen_action, 1)
            if reward == 1:
                action_buffer, states_buffer, actions, rewards = \
                    action_buffer[:j + 1], states_buffer[:j + 1], actions[:j + 1], rewards[:j + 1]
                wins, rewards[j] = wins + 1, 2

                RL_Policies.update_action_value(action_buffer, actions, rewards)
                long_term_states[index: index + j + 1] = states_buffer
                long_term_values[index: index + j + 1] = action_buffer
                index = index + j + 1
                break

            """ -------------------------------------------------------------------------------------------------------
                                                              Draw! 
             ------------------------------------------------------------------------------------------------------- """
            if j >= GAME_SIZE // 2:
                rewards[j] = -.5
                action_buffer, states_buffer, actions, rewards = \
                    action_buffer[:j + 1], states_buffer[:j + 1], actions[:j + 1], rewards[:j + 1]
                RL_Policies.update_action_value(action_buffer, actions, rewards)
                long_term_states[index: index + j + 1] = states_buffer
                long_term_values[index: index + j + 1] = action_buffer
                index = index + j + 1
                break

            """ -------------------------------------------------------------------------------------------------------
                                                         Random Bot Turn 
             ------------------------------------------------------------------------------------------------------- """
            bot_action = wallet.get_random_action()
            reward = rl_judge.step(bot_action, -1)
            temp_board[0, bot_action] = -1
            if reward == -1:
                action_buffer, states_buffer, actions, rewards = \
                    action_buffer[:j + 1], states_buffer[:j + 1], actions[:j + 1], rewards[:j + 1]
                rewards[j] = -2

                RL_Policies.update_action_value(action_buffer, actions, rewards)
                long_term_states[index: index + j + 1] = states_buffer
                long_term_values[index: index + j + 1] = action_buffer
                index = index + j + 1
                break

        if i % 1000 == 0:
            RANDOMIZATION_RATE *= 0.98
            wrt = int(100 * (wins / 1000.))
            print('EP: {}\tWinRate: {}'.format(i // 1000, wrt))
            _ = sess.run([update], feed_dict={x_holder: long_term_states[:index], y_holder: long_term_values[:index]})
            index, wins = 0, 1
