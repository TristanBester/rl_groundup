from .plot import create_line_plot, create_surface_plot, create_value_func_plot,\
		  plot_blackjack_value_functions, plot_mountain_car_value_function, create_bar_plot
from .printing import print_grid_world_actions, print_episode
from .policies import eps_greedy_policy, create_greedy_policy, test_policy, eps_greedy_func_policy,\
		      test_linear_policy, eps_greedy_policy_bin_features
from .tile_coding import TileCoding
from .encoding import encode_state, encode_sa_pair
