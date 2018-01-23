import numpy as np
import random


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

    def step(self, pos, signature):
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
                    # Bug here, Fix later, index is not right
                    s0 += board[self.__board_diameter * (bi + i) + i]

                if s0 == int(move_color * self.__win_diameter):
                    return move_color

        if self.__diagonal_b_evaluation_matrix[d] >= self.__win_diameter:
            self.__diagonal_b_evaluation_matrix[d] -= 1

            for bi in range(self.__board_diameter - self.__win_diameter + 1):
                s0 = 0
                for i in range(self.__win_diameter):
                    # Bug here, Fix later, index is not right
                    s0 += board[self.__board_diameter * (bi + i) + self.__board_diameter - bi - i - 1]

                if s0 == int(move_color * self.__win_diameter):
                    return move_color

        return 0


class ActionsWallet:
    def __init__(self, size):
        self.size = size
        self.actions = set(np.arange(size))
        self.actions_lef = size

    def initialize_new_wallet(self):
        self.actions = set(np.arange(self.size))
        self.actions_lef = self.size

    def remove_action(self, action):
        if self.valid_action(action):
            self.actions.remove(action)
            self.actions_lef -= 1

    def valid_action(self, action):
        if action in self.actions:
            return True
        return False

    def get_random_action(self):
        if self.actions_lef > 0:
            action = random.sample(self.actions, 1)[0]
            self.remove_action(action)
            return action
