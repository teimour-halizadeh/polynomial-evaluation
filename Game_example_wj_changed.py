import numpy as np
import matplotlib.pyplot as plt
from phe import paillier

import os
import glob

from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import math
import secrets
import time
from decimal import Decimal
from phe.util import invert, powmod, getprimeover, isqrt, is_prime, miller_rabin
import matplotlib





def Integrization(x,resolution,omega):
    # a function for turning a real number into a positive integer
    x = x * (1 / resolution)
    x = int(x)
    x = x % omega
    return int(x)

def nonnegtive_to_quantized(x,resolution,omega):
    # a function to turn a positive integer to real(quantized number)
    if x>=omega/2:
        x=x-omega
    else:
        x=x
    x=x*resolution
    return x



def grad_encrypt(xx, S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m):

    x1 = xx[0]; x2 = xx[1]; x3 = xx[2]; x4 = xx[3]

    # Node 2 and 3 and 4 need to encrypt S2a and S3a, so let's do them now:
    S2_encrypted = public_key.raw_encrypt(S2a)
    S3_encrypted = public_key.raw_encrypt(S3a)
    S4_encrypted = public_key.raw_encrypt(S4a)



    # computation for agent 2 by agent 1
    x_1 = b12
    x_1_hat = Integrization(x_1, r1, omega)
    Ex1 = public_key.raw_encrypt(x_1_hat)
    

    # step 2: After receiving above shares from node 1, node 2 computes P12
    # function.

    k1 = x2
    k1_hat = Integrization(k1, r1, omega)
    P12_encrypted = ((pow(Ex1, k1_hat, N_square))) % N_square


    # send the following to node 1
    node_2_share_encrypted = ((pow(P12_encrypted, 1, N_square))  * (pow(S2_encrypted, 1, N_square))) % N_square


    ########################################################
    # we have to do the same thing for node 3 to get the value of P13


    # Computations for node 3 and in node 3: First node 1 encrypts those parts of its

    # step 1 : node 1 encrypts its shares in P13

    x_1 = b13
    x_1_hat = Integrization(x_1, r1, omega)
    Ex1 = public_key.raw_encrypt(x_1_hat)
    

    # step 2: After receiving above shares from node 1, node 3 computes P13
    # function.

    k1 = x3
    k1_hat = Integrization(k1, r1, omega)
    P13_encrypted = ((pow(Ex1, k1_hat, N_square))) % N_square

    # node 3 sends the following to node 1
    node_3_share_encrypted  = ((pow(P13_encrypted, 1, N_square))  * (pow(S3_encrypted, 1, N_square))) % N_square



    #################################################3
    # Now, it is time for computing (W1 * W2 * W3 * W4) This is the part of the protocol that needs to
    # determine who is the distinguish node because the duty of the distinguish node differs from the others.
    # here we assume that node 4 is the distinguish node 
    # W = W1 * W2 * W3 * W4

    
    # we need to change the resolution for the gains because, otherwise node 1 has to use
    # resolution ^4 in the final stage. This differs from what it has to do for P12 and P13.

    # node 1 computation
    W11 = c11 + 2 * d11 * x1 
    x_1_hat = Integrization(W11, r2, omega)
    x_1_hat = (x_1_hat * S1m) % omega
    # keep it for now


    # Node 2 and 3 send their shares S2m * W2 and S3m * W3 to node1 1.
    # notice they do this at the same time they do the above steps. so they need not to
    # wait for something
    # First let's take care of the node 2

    # this part is done by node 1 becuse it includes the private data
    # W2_coeffieicent = c12
    # x_1_hat = Integrization(x_1, r1, omega)
    Ec12 = public_key.raw_encrypt(c12)
    Ed12 = public_key.raw_encrypt(d12)
    # send the coefficients to node 2


    # now it's time for node 2 to perform its tasks for the multiplication part
    k2_1 =  x2
    k2_1_hat = Integrization(k2_1, r2, omega)
    k2_1_hat = (k2_1_hat * S2m) % omega
    W121_encrypted = ((pow(Ec12, k2_1_hat, N_square))) % N_square

    k2_2 =  x2 ** 2
    k2_2_hat = Integrization(k2_2, r2, omega)
    k2_2_hat = (k2_2_hat * S2m) % omega
    W122_encrypted = ((pow(Ed12, k2_2_hat, N_square))) % N_square

    # it is ready to be sent to node 1 over the following ciphertext 
    W12_encrypted  = ((pow(W121_encrypted, 1, N_square))  * (pow(W122_encrypted, 1, N_square))) % N_square






    # now we are in node 3, obviously node 3 does this at the same time with node 2,
    # so there is no priority.

    # this part is done by node 1 becuse it includes the private data
    # W3_coeffieicent = c13
    # x_1_hat = Integrization(x_1, r1, omega)
    Ec13 = public_key.raw_encrypt(c13)
    Ed13 = public_key.raw_encrypt(d13)
    # send the coefficients to node 2


    # now it's time for node 2 to perform its tasks for the multiplication part
    k3_1 =  x3
    k3_1_hat = Integrization(k3_1, r2, omega)
    k3_1_hat = (k3_1_hat * S3m) % omega
    W131_encrypted = ((pow(Ec13, k3_1_hat, N_square))) % N_square

    k3_2 =  x3 ** 2
    k3_2_hat = Integrization(k3_2, r2, omega)
    k3_2_hat = (k3_2_hat * S3m) % omega
    W132_encrypted = ((pow(Ed13, k3_2_hat, N_square))) % N_square

    # it is ready to be sent to node 1 over the following ciphertext 
    W13_encrypted  = ((pow(W131_encrypted, 1, N_square))  * (pow(W132_encrypted, 1, N_square))) % N_square



    # node 1 decrypts what it has received from node 2 for the multiplication part
    W2_share_plain = private_key.raw_decrypt(W12_encrypted)

    # node 1 computes the following to get S2 * S1 * W2 * W1.
    Y1_2 = (W2_share_plain * x_1_hat) % omega


    # node 1 decrypts what it has received from node 3 for the multiplication part
    W3_share_plain = private_key.raw_decrypt(W13_encrypted)


    # again node 1, uses W3_share_plain and does the following over the ciphertext.
    Y1_3 = (W3_share_plain * Y1_2) % omega



    # now node 1 has to send the final result to node 4, but it has to encrypt it
    Y1_3 = public_key.raw_encrypt(Y1_3)
    # nod1 sends this result to node 4 as the last node that has to cooperate in the
    # computation of multiplication.

    # node 2 and 3 have done their part in the protocol, and the only node that remains
    # is node 4






    ##########################
    # node 4 computations:
    # part of the computations for this node is different than other nodes
    ############################

    # first P14
    # node 1 encrypts its share in this polynomial and sends it to this node
    x_1 = b14
    x_1_hat = Integrization(x_1, r1, omega)
    Ex1 = public_key.raw_encrypt(x_1_hat)
    


    # step 2: After receiving above share from node 1, node 4 computes the P14
    # function.

    k1 = x4
    k1_hat = Integrization(k1, r1, omega)
    P14_encrypted = ((pow(Ex1, k1_hat, N_square))) % N_square



    # then we take care of the multiplicative part
    W14 = c14 * x4 + d14 * x4 ** 2
    k1_4 = W14
    k1_4_hat = Integrization(k1_4, r2, omega)
    k1_4_hat = (k1_4_hat * S4m) % omega

    # now node 4 uses what it received from node 1 to compute W1 * W2 * W3 * W4
    Y1_4 = ((pow(Y1_3, k1_4_hat, N_square))) % N_square

   
    # this is the final step that node 4 has to do as the distinguished node
    node_4_share_encrypted  = (pow(P14_encrypted, 1, N_square))* ((pow(Y1_4, 1, N_square))  * (pow(S4_encrypted, 1, N_square))) % N_square





    # node 1 receives the above shares form 2 and 3, 4 and decrypt them
    node_2_share = private_key.raw_decrypt(node_2_share_encrypted)
    node_3_share = private_key.raw_decrypt(node_3_share_encrypted)
    node_4_share = private_key.raw_decrypt(node_4_share_encrypted)

    # node 1 then adds its own share of zeros S1 to the sum of two previous values

    P = (node_2_share + node_3_share + node_4_share+ S1a) %omega

    P = nonnegtive_to_quantized(P , resolution, omega)
    return (P + 2 *a1 *(x1) + 2 * b11 *x1)



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


    return np.array([F1, F2, F3, F4])



def Pro(xx_plus, up_bound, low_bound):
    # here we are doing the projection

    c1 = xx_plus[0]; c2 = xx_plus[1]; c3 = xx_plus[2]; c4 = xx_plus[3]

    if c1 > up_bound:
        c1 = up_bound

    elif c1 <0:
        c1 = 0

    if c2 > up_bound:
        c2= up_bound

    elif c2 < 0:
        c2 = 0
    
    if c3 > up_bound:
        c3 = up_bound

    elif c3 < 0:
        c3 = 0

    if c4 > up_bound:
        c4 = up_bound

    elif c4 <0:
        c4 = 0

    return np.array([c1, c2, c3, c4])       


def dis_share():

    s2a = secrets.randbelow(omega)
    s3a = secrets.randbelow(omega)
    s4a = secrets.randbelow(omega)
    s1a = (-(s2a + s3a + s4a)) % omega

    s2m = secrets.randbelow(omega)
    s3m = secrets.randbelow(omega)
    s4m = secrets.randbelow(omega)
    mul = (s2m * s3m * s4m) % omega
    s1m = pow((mul), omega - 2, omega)


    return s1a, s2a, s3a, s4a, s1m, s2m, s3m, s4m   

def dis_share_aggragation():


    S11a, S12a, S13a, S14a, S11m, S12m, S13m, S14m = dis_share()
    S21a, S22a, S23a, S24a, S21m, S22m, S23m, S24m = dis_share()
    S31a, S32a, S33a, S34a, S31m, S32m, S33m, S34m = dis_share()
    S41a, S42a, S43a, S44a, S41m, S42m, S43m, S44m = dis_share()


    S1a = (S11a + S21a + S31a + S41a) % omega 
    S2a = (S12a + S22a + S32a + S42a) % omega 
    S3a = (S13a + S23a + S33a + S43a) % omega 
    S4a = (S14a + S24a + S34a + S44a) % omega 


    S1m = (S11m * S21m * S31m * S41m) % omega
    S2m = (S12m * S22m * S32m * S42m) % omega
    S3m = (S13m * S23m * S33m * S43m) % omega
    S4m = (S14m * S24m * S34m * S44m) % omega
    return S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m



# Paillier scheme parameters
public_key, private_key, N, P, Q = paillier.generate_paillier_keypair(private_keyring=None, n_length=1024)
N_square = N * N


# resolution for quantization for states and controller
resolution = 10 ** -40
r1 = pow(resolution, 1/2)
r2 = pow(resolution, 1/4)



# Omega value in the paper.
# Notice it is 200 bits long, around the order of 10^60.
omega = getprimeover(200)






# Game parameters

# number of iteration
K = 100

# strategy vector
x_plain = np.zeros((4, K))

# upper bound and lower bound for the strtegies
up_bound = 2; low_bound = 0

# rate in the gradient descent method
alpha = 0.01



# plyer 1 parameters
a1 = 1; b11, b12, b13, b14 = (1, 1, 1, 1);  c11, c12, c13, c14 = (0.001, 1, 1, 1);  d11, d12, d13, d14 = (0.001, 1, 1, 1)

# plyer 2 parameter
a2 = 2; b21, b22, b23, b24 = (2, 2, 2, 2); c21, c22, c23, c24 = (1, 1, 1, 1);   d21, d22, d23, d24 = (0, 0, 0, 0)

# plyer 3 parameter
a3 = 3; b31, b32, b33, b34 = (3, 3, 3, 3); c31, c32, c33, c34 = (1, 1, 1, 1);   d31, d32, d33, d34 = (0, 0, 0, 0)

# plyer 3 parameter
a4 = 4; b41, b42, b43, b44 = (4, 4, 4, 4); c41, c42, c43, c44 = (1, 1, 1, 1);   d41, d42, d43, d44 = (0, 0, 0, 0)

# initial conditions of players
x_plain[:, 0] = np.array([2, 2, 2, 2])


# A matrix for measuring time
Measure_time = np.zeros(K)


# this part is when the game is done over plaintext
# we need to run this because we have to make sure that the
# proposed method does not introduce any error
for k in range(K-1):
    xx = x_plain[:, k]
    F = grad(xx)
    xx_plus = xx - alpha * (F)
    x_proj = Pro(xx_plus, up_bound, low_bound)
    x_plain[:, k + 1] = x_proj




###############################################################################
# THIS PART IS WHEN THE GAME IS DONE OVER CIPHERTEXT
##############################################################################


x = np.zeros((4, K))
x[:, 0] = np.array([2, 2, 2, 2])


for k in range(K-1):

    # it is possible that each agent store their aggregated random shares in  a file before the start of the algortihm
    # Here we just generate the shares in real time. 
    S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m = dis_share_aggragation()
    

    xx = x[:, k]

    # we also call the plain text gradient because we need the trajectory of other players during the game
    F = grad(xx)


   # measuing time for possible usage in the future
    start = time.time()

    # calling the function which evaluate the gradient over the cipher text
    Enc_grad =  grad_encrypt(xx, S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m)
    end = time.time()
    Measure_time[k] = (end - start)


    # we replace the first element in the gradient of plain text with what we got from the gradient of evaluated over ciphertext
    F[0] = Enc_grad


    xx_plus = xx - alpha * (F)
    x_proj = Pro(xx_plus, up_bound, low_bound)
    x[:, k + 1] = x_proj







# this is just a vector for plotting the results
Time_steps = np.arange(0, K, 1)



################################################################
# PLOT
################################################################


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

