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


def win_lose_probability_from_gs(sequence, label, n):
    length, index, flipper = len(sequence) + 1, 1, 1

    data = np.zeros((length, n * n), dtype=np.float32)
    labels = np.ones(length, dtype=np.float32) * label

    while index < length:
        if sequence[index - 1] == -1:
            break
        else:
            data[index] = data[index - 1]
            data[index, sequence[index - 1]] = flipper
            index, flipper = index + 1, flipper * - 1

    return data[:index], labels[:index]
