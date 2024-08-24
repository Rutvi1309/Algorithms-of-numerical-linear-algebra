import numpy as np
from numpy.testing import assert_allclose


def givens_qr(H):
    H = H + 0.0
    dtype = np.complex128 if np.iscomplexobj(H) else np.float64
    m = np.shape(H)[1]
    i = 0
    R = np.copy(H)
    Q = np.zeros((m, 2)).astype(H.dtype)
    G = np.zeros((m, 2)).astype(H.dtype)
    
    def givens_rotation(a, b):
        r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        c =  np.conj(a)/ r
        s = np.conj(b)/ r
        return c, s

   

    for k in range(m):
        a = R[k, k]
        b = R[k + 1, k]
       
        c, s = givens_rotation(a, b)
    
        if b != 0:
            G = np.identity(2).astype(c.dtype)
            G[0] = c, s
            G[1, 0] = -s.conjugate()
            G[1, 1] = c.conjugate()
            
            
            R[k:k + 2, k:] = G @ R[k:k + 2, k:]
            
            Q[i] = c, s
        else:
            Q[i] = 1, 0

        i += 1

    return Q, R

# Example usage with complex numbers:
H = np.array([[1,2,3],[4,5,6],[0,7,8],[0,0,9]])


print("Original Upper Hessenberg Matrix:")
print(H)
# Compute QR factorization using Givens rotations
R = givens_qr(H)
print("\nUpper Triangular Matrix R:")
print(R)

def form_q(G):
    m = G.shape[0]
    Q = np.identity(m + 1).astype(G.dtype)

    for j in range(m):
        
        K = np.identity(2).astype(G.dtype)
        K[:, 0] = G[j].conjugate()
        K[0, 1] = -G[j, 1]
        K[1, 1] = G[j, 0]

       
        Q[:, j:j + 2] = Q[:, j:j + 2] @ K

    return Q

