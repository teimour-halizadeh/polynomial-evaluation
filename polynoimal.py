
import secrets


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



def dis_share(omega):

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


def dis_share_aggragation(omega):


    S11a, S12a, S13a, S14a, S11m, S12m, S13m, S14m = dis_share(omega)
    S21a, S22a, S23a, S24a, S21m, S22m, S23m, S24m = dis_share(omega)
    S31a, S32a, S33a, S34a, S31m, S32m, S33m, S34m = dis_share(omega)
    S41a, S42a, S43a, S44a, S41m, S42m, S43m, S44m = dis_share(omega)


    S1a = (S11a + S21a + S31a + S41a) % omega 
    S2a = (S12a + S22a + S32a + S42a) % omega 
    S3a = (S13a + S23a + S33a + S43a) % omega 
    S4a = (S14a + S24a + S34a + S44a) % omega 


    S1m = (S11m * S21m * S31m * S41m) % omega
    S2m = (S12m * S22m * S32m * S42m) % omega
    S3m = (S13m * S23m * S33m * S43m) % omega
    S4m = (S14m * S24m * S34m * S44m) % omega
    return S1a, S2a, S3a, S4a, S1m, S2m, S3m, S4m






