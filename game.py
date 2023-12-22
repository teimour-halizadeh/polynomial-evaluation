import numpy as np



from constants import player_num, K, x_plain, up_bound, low_bound, alpha





# plyer 1 parameters
a1 = 1; b11, b12, b13, b14 = (1, 1, 1, 1);  c11, c12, c13, c14 = (0.001, 1, 1, 1);  d11, d12, d13, d14 = (0.001, 1, 1, 1)
# d11, d12, d13, d14 = (0.001, 1, 1, 1)

# plyer 2 parameter
a2 = 2; b21, b22, b23, b24 = (2, 2, 2, 2); c21, c22, c23, c24 = (1, 1, 1, 1);   d21, d22, d23, d24 = (0, 0, 0, 0)

# plyer 3 parameter
a3 = 3; b31, b32, b33, b34 = (3, 3, 3, 3); c31, c32, c33, c34 = (1, 1, 1, 1);   d31, d32, d33, d34 = (0, 0, 0, 0)

# plyer 4 parameter
a4 = 4; b41, b42, b43, b44 = (4, 4, 4, 4); c41, c42, c43, c44 = (1, 1, 1, 1);   d41, d42, d43, d44 = (0, 0, 0, 0)

# all other players' parameters
a_other = 0.1; b_other = 0.1


# initial conditions of players
x_plain[:, 0] = 2 * np.ones((player_num,))



def grad(xx):
    x1 = xx[0]; x2 = xx[1]; x3 = xx[2]; x4 = xx[3]


    W11 = c11 * x1 + d11 * x1 ** 2 
    W12 = c12 * x2 + d12 * x2 ** 2
    W13 = c13 * x3 + d13 * x3 ** 2
    W14 = c14 * x4 + d14 * x4 ** 2


    W21 = c21 * x1 + d21 * x1 ** 2
    W22 = c22 * x2 + d22 * x2 ** 2
    W23 = c23 * x3 + d23 * x3 ** 2
    W24 = c24 * x4 + d24 * x4 ** 2

    W31 = c31 * x1 + d31 * x1 ** 2
    W32 = c32 * x2 + d32 * x2 ** 2
    W33 = c33 * x3 + d33 * x3 ** 2
    W34 = c34 * x4 + d34 * x4 ** 2


    W41 = c41 * x1 + d41 * x1 ** 2
    W42 = c42 * x2 + d42 * x2 ** 2
    W43 = c43 * x3 + d43 * x3 ** 2
    W44 = c44 * x4 + d44 * x4 ** 2


    F1 = 2 *a1 *(x1) + (2 * b11 * x1) + (b12 * x2 + b13 * x3 + b14 * x4) + (c11 + 2 * d11 * x1) * (W12 * W13 * W14)

    F2 = 2 *a2 *(x2) + (2 * b22 * x2) + (b21 * x1 + b23 * x3 + b24 * x4) + (c22 + 2 * d22 * x2) * (W21 * W23 * W24)

    F3 = 2 *a3 *(x3) + (2 * b33 * x3) + (b31 * x1 + b32 * x2 + b34 * x4) + (c33 + 2 * d33 * x3) * (W31 * W32 * W34)

    F4 = 2 *a4 *(x4) + (2 * b44 * x4) + (b41 * x1 + b42 * x2 + b43 * x3) + (c44 + 2 * d44 * x4) * (W41 * W42 * W43)

    grad_four = np.array([F1, F2, F3, F4])

    agg_term = (a_other * np.sum(xx)) * np.ones((player_num - 4, 1))
    grad_other = b_other * xx[4:,]
    
    grad_player = np.append(grad_four, grad_other)


    return grad_player


def Pro(xx_plus, up_bound, low_bound, player_num):
    # here we are doing the projection

    for ind in range(player_num):
        beta = xx_plus[ind]
        if beta > up_bound:

            beta = up_bound


        elif beta <0:

            beta = 0

        xx_plus[ind] = beta
   

    return xx_plus    