import numpy as np


def update_action_value(Q, actions, rewards, learning_rate=0.1, alpha=0.99):
    """
    Calculate the actions value vector to train the function approximation.

    :param Q: The Quality matrix that will be updated
    :param actions: 1D vector of the action related to evey state
    :param rewards: 1D vector of evey state reward
    :param learning_rate: the learning rate
    :param alpha: the discount factor
    :return: None
    """

    # The last Q. We've to do it here to avoid the main loop index error.
    Q[-1, actions[-1]] = Q[-1, actions[-1]] + learning_rate * rewards[-1]

    # The main loop that calculate the discounted reward update the quality matrix.
    for t in reversed(range(len(rewards) - 1)):
        rewards[t] = rewards[t] + alpha * rewards[t + 1]

        s, s_, a, a_ = t, t + 1, actions[t], actions[t + 1]

        Q[s, a] = Q[s, a] + learning_rate * ((rewards[t] + alpha * Q[s_, a_]) - Q[s, a])