import numpy as np


# Game parameters

# number of players
player_num = 30

# number of iteration
K = 100

# strategy vector
x_plain = np.zeros((player_num, K))

# upper bound and lower bound for the strtegies
up_bound = 2; low_bound = 0

# rate in the gradient descent method
alpha = 0.01
