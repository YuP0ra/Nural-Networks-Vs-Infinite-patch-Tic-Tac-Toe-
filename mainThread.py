from libs import games_generator, subsidiaries
import numpy as np

for _ in range(10):
    samples_count, game_size, indexer = 10000, 9, 0

    all_data = np.zeros((samples_count * game_size * game_size, game_size * game_size))
    all_labels = np.zeros(samples_count * game_size * game_size)

    gg = games_generator.TicGameGenerator(game_size, 5)
    x, y = gg.get_n_random_games_samples(samples_count)

    for i in range(len(x)):
        dx, dy = x[i], y[i]
        q, w = subsidiaries.win_lose_probability_from_gs(dx, dy, game_size)
        prev_pos, indexer = indexer, indexer + len(q)
        all_data[prev_pos:indexer], all_labels[prev_pos:indexer] = q, w

    all_data, all_labels = all_data[:indexer], all_labels[:indexer]

    all_data.astype(np.int8).tofile('data_sets\\ready_to_train_data' + str(_))
    all_labels.astype(np.int8).tofile('data_sets\\ready_to_train_labels' + str(_))
