import numpy as np


class Judge:
    def __init__(self, board_diameter, win_diameter):
        self.__board_diameter = board_diameter
        self.__win_diameter = win_diameter
        self.__total_grid_size = board_diameter * board_diameter
        self.__valid_diagonals = 1 + 2 * (board_diameter - win_diameter)

        # the temp board on which the game will process
        self.__board = np.zeros(board_diameter * board_diameter)

        # Two times the grid. one for x other for o.
        self.__horizontal_evaluation_matrix = np.zeros(self.__board_diameter * 2)
        self.__vertical_evaluation_matrix = np.zeros(self.__board_diameter * 2)

        self.__diagonal_a_evaluation_matrix = np.zeros(self.__valid_diagonals * 2)
        self.__diagonal_b_evaluation_matrix = np.zeros(self.__valid_diagonals * 2)

    def initialize_new_board(self):
        self.__board *= 0

        self.__horizontal_evaluation_matrix *= 0
        self.__vertical_evaluation_matrix *= 0
        self.__diagonal_a_evaluation_matrix *= 0
        self.__diagonal_b_evaluation_matrix *= 0

    def set_next_move(self, pos, signature):
        self.__board[pos] = signature
        result = self.__check_board_state(self.__board, pos, signature)
        return result

    def __update_evaluations(self, last_play_pos, move_color):
        vh_offset = 0 if move_color is -1 else self.__board_diameter
        row_index = last_play_pos // self.__board_diameter
        col_index = last_play_pos % self.__board_diameter

        vrt_ev_index = vh_offset + row_index
        hrz_ev_index = vh_offset + col_index

        self.__vertical_evaluation_matrix[vrt_ev_index] += 1
        self.__horizontal_evaluation_matrix[hrz_ev_index] += 1

        digonals_offset = 0 if move_color is -1 else self.__valid_diagonals
        winning_offset = self.__board_diameter - self.__win_diameter

        digonal_a_index = digonals_offset + winning_offset + row_index - col_index
        digonal_b_index = digonals_offset + winning_offset + self.__board_diameter - 1 - row_index - col_index

        if digonal_a_index - digonals_offset < self.__valid_diagonals:
            self.__diagonal_a_evaluation_matrix[digonal_a_index] += 1
        else:
            digonal_a_index = -1

        if digonal_b_index - digonals_offset < self.__valid_diagonals:
            self.__diagonal_b_evaluation_matrix[digonal_b_index] += 1
        else:
            digonal_b_index = -1

        return vh_offset, row_index, col_index, digonal_a_index, digonal_b_index

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


class MovesWallet:
    def __init__(self, size):
        self.size, self.hi_index = size, size
        self.wallet = np.arange(size)

    def initialize_new_wallet(self):
        self.hi_index = self.size
        self.wallet = np.arange(self.size)

    def remove_move(self, index):
        self.hi_index -= 1
        self.wallet[self.hi_index] = self.wallet[index]

    def get_random_move(self):
        rnd = np.random.random_integers(0, self.hi_index)
        val = self.wallet[rnd]
        self.remove_move(rnd)
        return val
