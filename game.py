import numpy as np



from constants import player_num, K, x_plain, up_bound, low_bound, alpha

from constants import a_par, b_par, c_par, d_par, a_other, b_other






def grad(xx):
    x1 = xx[0]; x2 = xx[1]; x3 = xx[2]; x4 = xx[3]


    W11 = c_par[0,0] * x1 + d_par[0,0] * x1 ** 2 
    W12 = c_par[0,1] * x2 + d_par[0,1] * x2 ** 2
    W13 = c_par[0,2] * x3 + d_par[0,2] * x3 ** 2
    W14 = c_par[0,3] * x4 + d_par[0,3] * x4 ** 2


    W21 = c_par[1, 0] * x1 + d_par[1,0] * x1 ** 2
    W22 = c_par[1, 1] * x2 + d_par[1,1] * x2 ** 2
    W23 = c_par[1, 2] * x3 + d_par[1,2] * x3 ** 2
    W24 = c_par[1, 3] * x4 + d_par[1,3] * x4 ** 2

    W31 = c_par[2,0] * x1 + d_par[2,0] * x1 ** 2
    W32 = c_par[2,1] * x2 + d_par[2,1] * x2 ** 2
    W33 = c_par[2,2] * x3 + d_par[2,2] * x3 ** 2
    W34 = c_par[2,3] * x4 + d_par[2,3] * x4 ** 2


    W41 = c_par[3,0] * x1 + d_par[3,0] * x1 ** 2
    W42 = c_par[3,1] * x2 + d_par[3,1] * x2 ** 2
    W43 = c_par[3,2] * x3 + d_par[3,2] * x3 ** 2
    W44 = c_par[3,3] * x4 + d_par[3,3] * x4 ** 2


    F1 = 2 *a_par[0,0] *(x1) + (2 * b_par[0,0] * x1) + (b_par[0,1] * x2 + b_par[0,2] * x3 + b_par[0,3] * x4) + (c_par[0,0] + 2 * d_par[0,0] * x1) * (W12 * W13 * W14)

    F2 = 2 *a_par[0,1] *(x2) + (2 * b_par[1,1] * x2) + (b_par[1,0] * x1 + b_par[1,2] * x3 + b_par[1,3] * x4) + (c_par[1, 1] + 2 * d_par[1,1] * x2) * (W21 * W23 * W24)

    F3 = 2 *a_par[0,2] *(x3) + (2 * b_par[2,2] * x3) + (b_par[2,0] * x1 + b_par[2,1] * x2 + b_par[2,3] * x4) + (c_par[2,2] + 2 * d_par[2,2] * x3) * (W31 * W32 * W34)

    F4 = 2 *a_par[0,3] *(x4) + (2 * b_par[3,3] * x4) + (b_par[3,0] * x1 + b_par[3,1] * x2 + b_par[3,2] * x3) + (c_par[3,3] + 2 * d_par[3,3] * x4) * (W41 * W42 * W43)

    grad_four = np.array([F1, F2, F3, F4])

    agg_term = (a_other * np.sum(xx)) * np.ones((player_num - 4, 1))
    grad_other = b_other * xx[4:,]
    
    grad_player = np.append(grad_four, grad_other)


    return grad_player


def projection_to_set(xx_plus, up_bound, low_bound, player_num):
    # here we are doing the projection

    for ind in range(player_num):
        beta = xx_plus[ind]
        if beta > up_bound:

            beta = up_bound


        elif beta <0:

            beta = 0

        xx_plus[ind] = beta
   

    return xx_plus    