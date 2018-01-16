from libs import games_generator, subsidiaries
import numpy as np


def old_job():
    for _ in range(1000):
        samples_count, game_size, indexer = 1000, 3, 0

        all_data = np.zeros((samples_count * game_size * game_size, game_size * game_size))
        all_labels = np.zeros(samples_count * game_size * game_size)

        gg = games_generator.TicGameGenerator(game_size, 3)
        x, y = gg.get_n_random_games_samples(samples_count)

        for i in range(len(x)):
            dx, dy = x[i], y[i]
            q, w = subsidiaries.win_lose_probability_from_gs(dx, dy, game_size)
            prev_pos, indexer = indexer, indexer + len(q)
            all_data[prev_pos:indexer], all_labels[prev_pos:indexer] = q, w

        all_data, all_labels = all_data[:indexer], all_labels[:indexer]
        final_labels = np.zeros(2 * len(all_labels)).reshape(len(all_labels), 2)
        final_labels[np.where(all_labels == 1), 0] = 1
        final_labels[np.where(all_labels == -1), 1] = 1

        np.savez_compressed(file='data_sets\\compressed_' + str(_),
                            data=all_data.astype(np.int8),
                            reward=all_labels.astype(np.int8),
                            labels=final_labels.astype(np.int8))

        print('Finished: ', _, '\tOut of: ', 1000)
