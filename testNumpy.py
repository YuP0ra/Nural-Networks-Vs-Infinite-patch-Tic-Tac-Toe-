from libs import games_generator as gg

game_generator = gg.TicGameGenerator(9, 5)
x, y = game_generator.get_n_random_games_samples(100000)

x.tofile('data_sets/data.kd')
y.tofile('data_sets/labels.kd')
