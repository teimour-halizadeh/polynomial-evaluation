import numpy as np




# resolution for quantization for states and controller
resolution = 10 ** -40
r1 = pow(resolution, 1/2)
r2 = pow(resolution, 1/4)





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





a_par = np.array([[1, 2, 3, 4]])

b_par = np.array([[1, 0, 0, 0],
                  [0, 2, 0, 0],
                  [0, 0, 3, 0],
                  [0, 0, 0, 4]]) @ np.ones((4,4))

c_par = np.ones((4,4))
c_par[0, 0] = 0.001


d_par = np.zeros((4,4))
d_par[0,:] = np.array([[0.001, 1, 1, 1]])




# all other players' parameters
a_other = 0.1; b_other = 0.1


# the communication graph for the player
# notice this communication graph is different from the inference graph.
# here we assume a star graph for the players
# it could be anything else as long as it has the condition in the main paper


all_one = np.ones((player_num, 1))

e1 = np.zeros((player_num, 1))
e1[0, 0] = np.array(1)

e_1 = all_one - e1

# adjacency matrix for the star graph
Ad = np.matmul(e1, np.transpose(e_1)) + np.matmul(e_1, np.transpose(e1))

D = np.sum(Ad,  axis=0)
Dw = np.diag(D)

# this is the Laplacian matrix. We use this matrix to implement the code in the Pavel's paper
L = Dw - Ad
