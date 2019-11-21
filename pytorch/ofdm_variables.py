#load coding matrices
from parity import *

#generate DFT matrix
def DFT(N):  
    W = np.zeros((N, N), dtype=np.complex)
    
    for x in range(0,N):
        for y in range(0,N): 
            W[x,y] = np.exp(-1j*2*np.pi*x*y / N) / np.sqrt(N)
            
    return W

def DFTreal(N):
    W = DFT(N)
    
    Wr = np.zeros((2*N, 2*N), dtype=np.float)
    
    for x in range(0,N):
        for y in range(0,N):
            Wr[2*x, 2*y] = W[x,y].real
            Wr[2*x, 2*y+1] = -W[x,y].imag
            Wr[2*x+1, 2*y] = W[x,y].imag
            Wr[2*x+1, 2*y+1] = W[x,y].real
            
    return Wr
    

#QPSK mapping
qpsk_mapping = {
        (0,0) : 1/np.sqrt(2) + 1j/np.sqrt(2),
        (1,0) : - 1/np.sqrt(2) + 1j/np.sqrt(2),
        (1,1) : - 1/np.sqrt(2) - 1j/np.sqrt(2),
        (0,1) : 1/np.sqrt(2) - 1j/np.sqrt(2)
        }

qpsk_demapping = {v : k for k, v in qpsk_mapping.items()}