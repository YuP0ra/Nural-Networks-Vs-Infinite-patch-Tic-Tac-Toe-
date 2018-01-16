from libs import games_generator, subsidiaries, neural_network
import numpy as np


def training_data(samples_count, game_size):

    gg = games_generator.TicGameGenerator(game_size, game_size)
    x, y = gg.get_n_random_games_samples(samples_count)
    return subsidiaries.first_player_sequence_into_network_perspective(x[np.where(y == 1)], game_size)


my_network = neural_network.NeuralNetworkFourHiddenLayers([9, 80, 80, 80, 9], name='alpha')
print('Neural Network got initialized')


for i in range(100):
    print('Working on loop:', i)
    data, labels = training_data(100, 3)
    my_network.train_network(data, labels)

    res = my_network.predict_optimized(np.zeros(9, dtype=np.float32).reshape(1, 9)) * 100
    print(res.astype(int))
