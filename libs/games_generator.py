import numpy as np


class TicGameGenerator:
    """ docstring for TicGameGenerator
    this class can be used to generate TicTacToe games
    """

    def __init__(self, board_diameter, win_diameter):
        self.__board_diameter = board_diameter
        self.__win_diameter = win_diameter
        self.__total_grid_size = board_diameter * board_diameter
        self.__valid_diagonals = 1 + 2 * (board_diameter - win_diameter)

        # Two times the grid. one for x other for o.
        self.__horizontal_evaluation_matrix = np.zeros(self.__board_diameter * 2)
        self.__vertical_evaluation_matrix = np.zeros(self.__board_diameter * 2)

        self.__diagonal_a_evaluation_matrix = np.zeros(self.__valid_diagonals * 2)
        self.__diagonal_b_evaluation_matrix = np.zeros(self.__valid_diagonals * 2)

    def __clear_evaluations(self):
        self.__horizontal_evaluation_matrix *= 0
        self.__vertical_evaluation_matrix *= 0
        self.__diagonal_a_evaluation_matrix *= 0
        self.__diagonal_b_evaluation_matrix *= 0

    def get_n_random_games_samples(self, n_samples):
        all_possible_positions = np.arange(self.__total_grid_size)
        n_games = np.zeros((n_samples, self.__total_grid_size), dtype=np.int16)
        n_label = np.zeros(n_samples, dtype=np.int16)

        for _ in range(n_samples):
            temp_board = np.zeros(self.__total_grid_size, dtype=int)
            seq_array = np.ones(self.__total_grid_size, dtype=int) * -1
            np.random.shuffle(all_possible_positions)
            self.__clear_evaluations()

            for i in range(self.__total_grid_size):
                pos, move = all_possible_positions[i], 1 - ((i % 2) * 2)  # 1, -1, 1, -1, ...
                seq_array[i], temp_board[pos] = pos, move
                current_state = self.__check_board_state(temp_board, pos, move)
                if current_state in (1, -1):
                    n_label[_] = current_state
                    break

            n_games[_] = seq_array

        return n_games, n_label

    def __update_evaluations(self, last_play_pos, move_color):
        vh_offset = 0 if move_color is -1 else self.__board_diameter
        row_index = last_play_pos // self.__board_diameter
        col_index = last_play_pos % self.__board_diameter

        vrt_ev_index = vh_offset + row_index
        hrz_ev_index = vh_offset + col_index

        self.__vertical_evaluation_matrix[vrt_ev_index] += 1
        self.__horizontal_evaluation_matrix[hrz_ev_index] += 1

        diagonals_offset = 0 if move_color == 1 else self.__valid_diagonals

        diagonal_a_index = abs(row_index - col_index)
        diagonal_b_index = abs(self.__board_diameter - row_index - col_index - 1)

        if diagonal_a_index <= self.__valid_diagonals // 2:
            diagonal_a_index = diagonals_offset + self.__valid_diagonals // 2 + row_index - col_index
            self.__diagonal_a_evaluation_matrix[diagonal_a_index] += 1
        else:
            diagonal_a_index = -1

        if diagonal_b_index <= self.__valid_diagonals // 2:
            diagonal_b_index = diagonals_offset + self.__valid_diagonals // 2 + self.__board_diameter - 1 - row_index - col_index
            self.__diagonal_b_evaluation_matrix[diagonal_b_index] += 1
        else:
            diagonal_b_index = -1

        return vh_offset, row_index, col_index, diagonal_a_index, diagonal_b_index

    def __check_board_state(self, board, last_play_pos, move_color):
        o, a, b, c, d = self.__update_evaluations(last_play_pos, move_color)

        if self.__vertical_evaluation_matrix[o + a] >= self.__win_diameter:
            self.__vertical_evaluation_matrix[o + a] -= 1

            for bi in range(self.__board_diameter - self.__win_diameter + 1):
                s0 = 0
                for i in range(self.__win_diameter):
                    s0 += board[(self.__board_diameter * a) + (bi + i)]

                if s0 == int(move_color * self.__win_diameter):
                    return move_color

        if self.__horizontal_evaluation_matrix[o + b] >= self.__win_diameter:
            self.__horizontal_evaluation_matrix[o + b] -= 1

            for bi in range(self.__board_diameter - self.__win_diameter + 1):
                s0 = 0
                for i in range(self.__win_diameter):
                    s0 += board[(self.__board_diameter * (bi + i)) + b]

                if s0 == int(move_color * self.__win_diameter):
                    return move_color

        if self.__diagonal_a_evaluation_matrix[c] >= self.__win_diameter:
            self.__diagonal_a_evaluation_matrix[c] -= 1

            for bi in range(self.__board_diameter - self.__win_diameter + 1):
                s0 = 0
                for i in range(self.__win_diameter):
                    s0 += board[self.__board_diameter * (bi + i) + i]

                if s0 == int(move_color * self.__win_diameter):
                    return move_color

        if self.__diagonal_b_evaluation_matrix[d] >= self.__win_diameter:
            self.__diagonal_b_evaluation_matrix[d] -= 1

            for bi in range(self.__board_diameter - self.__win_diameter + 1):
                s0 = 0
                for i in range(self.__win_diameter):
                    s0 += board[self.__board_diameter * (bi + i) + self.__board_diameter - bi - i - 1]

                if s0 == int(move_color * self.__win_diameter):
                    return move_color

        return 0