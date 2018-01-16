import numpy as np


def first_player_sequence_into_network_perspective(sequence, n):
    total_array_size, index, sample_size = n * n * len(sequence), 0, n * n

    present_state = np.zeros((total_array_size,sample_size), dtype=np.float32)
    next_state = np.zeros((total_array_size, sample_size), dtype=np.float32)

    for sec in sequence:
        for i in range(sample_size // 2):
            if sec[i] == -1:
                break
            else:
                index = index + 1
                next_state[index - 1, sec[i]] = 1
                present_state[index] = next_state[index - 1]
                i += 2

    return present_state[:index], next_state[:index]


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
