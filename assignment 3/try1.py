import numpy as np
from numpy.testing import assert_allclose

def givens_qr(H):
    H = H + 0.0
    m = np.shape(H)[1]
    R = np.copy(H)
    Q = np.zeros((m, 2)).astype(H.dtype)
    G = np.zeros((m, 2)).astype(H.dtype)
    i = 0
    
    def givens_rotation(a, b):
        if a == 0:
            return 0, 1
        else:
            r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
            c = np.conj(a) / r
            s = np.conj(b) / r
            return c, s
    
    for k in range(m):
        # Compute Givens rotation parameters
        c, s = givens_rotation(R[k, k], R[k+1, k])

        if s != 0:  # Avoid division by zero
            # Construct Givens rotation matrix
            G = np.identity(2).astype(c.dtype)
            G[0] = c, s
            G[1, 0] = -np.conj(s)
            G[1, 1] = np.conj(c)

            # Update R with Givens rotation
            R[k:k+2, k:] = G @ R[k:k+2, k:]

            # Store Givens rotation parameters in Q
            Q[i] = c, s
        else:
            # If s is zero, set Q to the identity matrix
            Q[i] = 1, 0

        i += 1

    return Q, R

def form_q(G):
    m = G.shape[0]
    Q = np.eye(m + 1).astype(G.dtype)

    for j in range(m):
        c, s = G[j]

        # Construct Givens rotation matrix from parameters
        K = np.identity(2).astype(G.dtype)
        K[:, 0] = np.conj(c)
        K[0, 1] = -np.conj(s)
        K[1, 1] = c

        # Update Q with Givens rotation
        Q[:, j:j + 2] = Q[:, j:j + 2] @ K

    return Q


   