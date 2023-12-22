import numpy as np
import matplotlib.pyplot as plt
from phe import paillier

import os
import glob

from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import math
# import secrets
import time
from decimal import Decimal
from phe.util import invert, powmod, getprimeover, isqrt, is_prime, miller_rabin
import matplotlib
from numpy.linalg import inv

from polynoimal_base_functions import integrization
from polynoimal_base_functions import nonnegtive_to_quantized
from polynoimal_base_functions import dis_share
from polynoimal_base_functions import dis_share_aggragation
from game import Pro
from game import grad

from grad_encrypt import grad_encrypt

from constants import player_num, K, x_plain, up_bound, low_bound, alpha
from constants import L, resolution, r1, r2





# Paillier scheme parameters
public_key, private_key, N, P, Q = paillier.generate_paillier_keypair(private_keyring=None, n_length=1024)
N_square = N * N

# Omega value in the paper.
# Notice it is 200 bits long, around the order of 10^60.
omega = getprimeover(200)





########################################################################################




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






# global shared constraint:
# this translates itself to sum(x_i) > 1
A = np.ones((player_num))
B = (1/player_num) * np.ones((player_num))



# lagrange multiplier
lamda = np.zeros((player_num, K))

# z vector defined in the main paper
Z = np.zeros((player_num, K))






# this part is when the game is done over plaintext
# we need to run this because we have to make sure that the
# proposed method does not introduce any error
for k in range(K-1):

    xx = x_plain[:, k]

    F = grad(xx)

    xx_plus = xx - alpha * (F) + alpha *((A) * lamda[:, k])


    x_proj = Pro(xx_plus, up_bound, low_bound, player_num)

    x_plain[:, k + 1] = x_proj


    Z[:, k + 1] = Z[:, k] + alpha * (np.matmul(L, lamda[:, k])) 

    t1 = ((A) * (2 * x_plain[:, k+1] - x_plain[:, k])) - (B)


    t2 = np.matmul(2 * L, Z[:, k+1])
    t3 = np.matmul(-L, Z[:, k])
    t4 = np.matmul(L, lamda[:, k])

    ll = lamda[:, k] - alpha *(t1 + t2 + t3 + t4)

    ll_proj = Pro(ll, 10000, low_bound, player_num)
    lamda[:, k + 1] = ll_proj


###############################################################################
# THIS PART IS WHEN THE GAME IS DONE OVER CIPHERTEXT
##############################################################################


x = np.zeros((player_num, K))
x[:, 0] = 2 * np.ones((player_num,))
# A matrix for measuring time
Measure_time = np.zeros(K)

for k in range(K-1):

    # it is possible that each agent store their aggregated random shares in  a file before the start of the algortihm
    # Here we just generate the shares in real time. 
    S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m = dis_share_aggragation(omega)
    

    xx = x[:, k]

    # we also call the plain text gradient because we need the trajectory of other players during the game
    F = grad(xx)


   # measuing time for possible usage in the future
    start = time.time()

    # calling the function which evaluate the gradient over the cipher text
    Enc_grad =  grad_encrypt(xx, S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m,
                  r1, r2, N_square, resolution, omega, public_key, private_key)
    end = time.time()
    Measure_time[k] = (end - start)


    # we replace the first element in the gradient of plain text with what we got from the gradient of evaluated over ciphertext
    F[0] = Enc_grad


    xx_plus = xx - alpha * (F) + alpha *((A) * lamda[:, k])

    # xx_plus = xx - alpha * (F) 

    x_proj = Pro(xx_plus, up_bound, low_bound, player_num)

    x[:, k + 1] = x_proj


    Z[:, k + 1] = Z[:, k] + alpha * (np.matmul(L, lamda[:, k])) 

    t1 = ((A) * (2 * x[:, k+1] - x[:, k])) - (B)


    t2 = np.matmul(2 * L, Z[:, k+1])
    t3 = np.matmul(-L, Z[:, k])
    t4 = np.matmul(L, lamda[:, k])

    ll = lamda[:, k] - alpha *(t1 + t2 + t3 + t4)

    ll_proj = Pro(ll, 10000, low_bound, player_num)
    lamda[:, k + 1] = ll_proj









################################################################
# PLOT
################################################################

# this is just a vector for plotting the results
Time_steps = np.arange(0, K, 1)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('seaborn-paper')


plt.step(Time_steps, x_plain[0, :],where = 'post')
plt.step(Time_steps, x[0, :], '--' ,where = 'post')

plt.legend(['$x_i$ using the Algorithm 1', '$x_i$ using plain data'])
plt.title('Player i decision trajectory')
plt.grid(axis='y', color='0.95')
plt.xlabel('# steps')


# f.set_size_inches(7,9.5,(10,10))

plt.savefig('game.pdf', bbox_inches='tight',pad_inches=0, dpi=1600, transparent=True)
plt.show()

