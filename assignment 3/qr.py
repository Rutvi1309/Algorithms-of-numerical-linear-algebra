import numpy as np


def givens_rotation(a, b):
    r = np.sqrt(a**2 + b**2)
    c = a / r
    s = -b / r
    return c, s

def givens_qr(H):
    m, n = H.shape
    R = H.copy()
    G = np.zeros((m, 2), dtype=complex)

    for k in range(min(m+1, m)):  # Ensure we don't go beyond the matrix dimensions
        # Compute Givens rotation parameters
        ck, sk = givens_rotation(R[k, k], R[k+1, k])

        # Apply Givens rotation to the submatrix R[k:k+2, k:n]
        R[k:k+2, k:n] = np.dot(np.array([[ck, -sk], [sk, ck]]), R[k:k+2, k:n])

        # Store Givens rotation parameters in matrix G
        G[k, :] = [ck, sk]

    return  R , G
    print("R:\n", np.round(R))






def form_q(G):
    Q = None
    #TODO
    return Q
