from libs import tic_judge
import numpy as np

rl_judge = tic_judge.Judge(3, 2)
wallet = tic_judge.MovesWallet(9)

rl_judge.initialize_new_board()


print(rl_judge.set_next_move(1, 1))
print(rl_judge.set_next_move(5, 1))
