
from polynoimal_base_functions import integrization
from polynoimal_base_functions import nonnegtive_to_quantized

from constants import a_par, b_par, c_par, d_par, a_other, b_other






def grad_encrypt(xx, S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m,
                  r1, r2, N_square, resolution, omega, public_key, private_key):

    x1 = xx[0]; x2 = xx[1]; x3 = xx[2]; x4 = xx[3]

    # Node 2 and 3 and 4 need to encrypt S2a and S3a, so let's do them now:
    S2_encrypted = public_key.raw_encrypt(S2a)
    S3_encrypted = public_key.raw_encrypt(S3a)
    S4_encrypted = public_key.raw_encrypt(S4a)



    # computation for agent 2 by agent 1
    x_1 = b_par[0,1]
    x_1_hat = integrization(x_1, r1, omega)
    Ex1 = public_key.raw_encrypt(x_1_hat)
    

    # step 2: After receiving above shares from node 1, node 2 computes P12
    # function.

    k1 = x2
    k1_hat = integrization(k1, r1, omega)
    P12_encrypted = ((pow(Ex1, k1_hat, N_square))) % N_square


    # send the following to node 1
    node_2_share_encrypted = ((pow(P12_encrypted, 1, N_square))  * (pow(S2_encrypted, 1, N_square))) % N_square


    ########################################################
    # we have to do the same thing for node 3 to get the value of P13


    # Computations for node 3 and in node 3: First node 1 encrypts those parts of its

    # step 1 : node 1 encrypts its shares in P13

    x_1 = b_par[0,2]
    x_1_hat = integrization(x_1, r1, omega)
    Ex1 = public_key.raw_encrypt(x_1_hat)
    

    # step 2: After receiving above shares from node 1, node 3 computes P13
    # function.

    k1 = x3
    k1_hat = integrization(k1, r1, omega)
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
    W11 = c_par[0,0] + 2 * d_par[0,0] * x1 
    x_1_hat = integrization(W11, r2, omega)
    x_1_hat = (x_1_hat * S1m) % omega
    # keep it for now


    # Node 2 and 3 send their shares S2m * W2 and S3m * W3 to node1 1.
    # notice they do this at the same time they do the above steps. so they need not to
    # wait for something
    # First let's take care of the node 2

    # this part is done by node 1 becuse it includes the private data
    # W2_coeffieicent = c_par[0,1]
    # x_1_hat = Integrization(x_1, r1, omega)
    Ec_par01 = public_key.raw_encrypt(int(c_par[0,1]))
    Ed12 = public_key.raw_encrypt(int(d_par[0,1]))
    # send the coefficients to node 2


    # now it's time for node 2 to perform its tasks for the multiplication part
    k2_1 =  x2
    k2_1_hat = integrization(k2_1, r2, omega)
    k2_1_hat = (k2_1_hat * S2m) % omega
    W121_encrypted = ((pow(Ec_par01, k2_1_hat, N_square))) % N_square

    k2_2 =  x2 ** 2
    k2_2_hat = integrization(k2_2, r2, omega)
    k2_2_hat = (k2_2_hat * S2m) % omega
    W122_encrypted = ((pow(Ed12, k2_2_hat, N_square))) % N_square

    # it is ready to be sent to node 1 over the following ciphertext 
    W12_encrypted  = ((pow(W121_encrypted, 1, N_square))  * (pow(W122_encrypted, 1, N_square))) % N_square






    # now we are in node 3, obviously node 3 does this at the same time with node 2,
    # so there is no priority.

    # this part is done by node 1 becuse it includes the private data
    # W3_coeffieicent = c_par[0,2]
    # x_1_hat = Integrization(x_1, r1, omega)
    Ec_par02 = public_key.raw_encrypt(int(c_par[0,2]))
    Ed13 = public_key.raw_encrypt(int(d_par[0,2]))
    # send the coefficients to node 2


    # now it's time for node 2 to perform its tasks for the multiplication part
    k3_1 =  x3
    k3_1_hat = integrization(k3_1, r2, omega)
    k3_1_hat = (k3_1_hat * S3m) % omega
    W131_encrypted = ((pow(Ec_par02, k3_1_hat, N_square))) % N_square

    k3_2 =  x3 ** 2
    k3_2_hat = integrization(k3_2, r2, omega)
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
    x_1 = b_par[0,3]
    x_1_hat = integrization(x_1, r1, omega)
    Ex1 = public_key.raw_encrypt(x_1_hat)
    


    # step 2: After receiving above share from node 1, node 4 computes the P14
    # function.

    k1 = x4
    k1_hat = integrization(k1, r1, omega)
    P14_encrypted = ((pow(Ex1, k1_hat, N_square))) % N_square



    # then we take care of the multiplicative part
    W14 = c_par[0,3] * x4 + d_par[0,3] * x4 ** 2
    k1_4 = W14
    k1_4_hat = integrization(k1_4, r2, omega)
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
    return (P + 2 *a_par[0,0] *(x1) + 2 * b_par[0,0] *x1)