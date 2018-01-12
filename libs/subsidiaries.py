import numpy as np


def first_player_sequence_into_network_perspective(sequence, n):
    hn = n * n // 2 + 1
    data = np.zeros(hn * n * n).reshape(hn, n * n)
    labels = np.zeros(hn * n * n).reshape(hn, n * n)

    labels[0, sequence[0]] = 1
    for i in range(1, hn):
        if sequence[i * 2] == -1:
            hn = i
            break
        labels[i, sequence[i * 2]] = 1
        data[i] = labels[i - 1]

    return data[:hn], labels[:hn]
