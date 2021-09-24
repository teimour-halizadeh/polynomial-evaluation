import numpy as np
import matplotlib.pyplot as plt
from phe import paillier
# import scipy.io as spio
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





def Integrization(x,resolution,phi):
    x = x * (1 / resolution)
    x = int(x)
    x = x % phi
    return int(x)

def nonnegtive_to_quantized(x,resolution,phi):
    if x>=phi/2:
        x=x-phi
    else:
        x=x
    x=x*resolution
    return x
def neighbours(X):

    K = np.size(X)
    A = X

    start = time.time()

    for k in range(K):
        
        

        

        x_1_hat = Integrization(A[k, 0], r1, phi)
            

        Ex1 = public_key.raw_encrypt(x_1_hat)

        node_2_share = private_key.raw_decrypt(Ex1)

        y = nonnegtive_to_quantized(node_2_share , r1, phi)
      
    end = time.time()
    return (end - start)
    


R = 10

key_dist = np.array([900, 1000, 1100, 1200, 1300 ,1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000])

Measure_time_1 = np.zeros((R, np.size(key_dist)))
Measure_time_2 = np.zeros((R, np.size(key_dist)))
Measure_time_3 = np.zeros((R, np.size(key_dist)))

cipher_size = np.zeros((1, np.size(key_dist)))


for RR in range(R):

    for kk in range(np.size(key_dist)):

        # key for paillier
        public_key, private_key, N, P, Q = paillier.generate_paillier_keypair(private_keyring=None, n_length=int(key_dist[kk]))

        N_square = N * N

        # resolution for quantization for states and controller
        resolution = 10 ** -40
        r1 = pow(resolution, 1/2)
        r2 = pow(resolution, 1/4)
        # gain_resolution = 10 ** -12


        phi = getprimeover(200)


        x1 = np.random.rand(3,1)

        x2 = np.random.rand(9,1)

        x3 = np.random.rand(27,1)

        Measure_time_1[RR, kk] = neighbours(x1)
        Measure_time_2[RR, kk] = neighbours(x2)
        Measure_time_3[RR, kk] = neighbours(x3)

       
        

        # Ex1 = public_key.raw_encrypt(10)
       
        # cipher_size[0, kk] = len(bin(Ex1)[2:])




Avg_Measure_time_1 = np.average(Measure_time_1, axis=0)
Avg_Measure_time_2 = np.average(Measure_time_2, axis=0)
Avg_Measure_time_3 = np.average(Measure_time_3, axis=0)


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('seaborn-paper')


plt.plot(key_dist, Avg_Measure_time_1)
plt.plot(key_dist, Avg_Measure_time_2,'--')
plt.plot(key_dist, Avg_Measure_time_3,'-.')


plt.legend(['$|\\mathcal{N}_i| = 3$', '$|\\mathcal{N}_i| = 9$', '$|\\mathcal{N}_i| = 27$'])
plt.title('Computation time vs. key length and $|\\mathcal{N}_i|$')
plt.grid(axis='y', color='0.95')
plt.xlabel('Key length (bits)')
plt.ylabel('Time(s)')

# f.set_size_inches(7,9.5,(10,10))

plt.savefig('compare_game_1.pdf', bbox_inches='tight',pad_inches=0, dpi=1600, transparent=True)
plt.show()






Avg_Measure_time_1 = np.log10(Avg_Measure_time_1)
Avg_Measure_time_2 = np.log10(Avg_Measure_time_2)
Avg_Measure_time_3 = np.log10(Avg_Measure_time_3)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.style.use('seaborn-paper')


plt.plot(key_dist, Avg_Measure_time_1)
plt.plot(key_dist, Avg_Measure_time_2,'--')
plt.plot(key_dist, Avg_Measure_time_3,'-.')


plt.legend(['$|\\mathcal{N}_i| = 3$', '$|\\mathcal{N}_i| = 9$', '$|\\mathcal{N}_i| = 27$'])
plt.title('Computation time vs. key length and $|\\mathcal{N}_i|$')
plt.grid(axis='y', color='0.95')
plt.xlabel('Key length (bits)')
plt.ylabel('log (time)(s)')

# f.set_size_inches(7,9.5,(10,10))

plt.savefig('compare_game.pdf', bbox_inches='tight',pad_inches=0, dpi=1600, transparent=True)
plt.show()


print(cipher_size)