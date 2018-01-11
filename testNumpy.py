from libs import games_generator as gg
from libs import subsidiaries as sub


game_generator = gg.TicGameGenerator(3, 3)
x, y = game_generator.get_n_random_games_samples(1)

print(x[0])
xd, dx = sub.first_player_sequence_into_network_perspective(x[0], 3)
print(xd , '\n\n', dx)
