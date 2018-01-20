from libs import tic_judge, subsidiaries
import numpy as np

GAME_SIZE = 10
wallet = tic_judge.ActionsWallet(GAME_SIZE)


wallet.initialize_new_wallet()

for i in range(GAME_SIZE):
    print(np.random.ranf())
