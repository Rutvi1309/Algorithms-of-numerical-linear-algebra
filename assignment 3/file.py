import numpy as np

def givens_qr(H):
    """
    Decompose matrix H into QR form using Given's rotations.
    """
    # get size of the matrix
    m, n = H.shape

    # create empty matrix for R
    R = np.copy(H).astype(complex)

    # create empty matrix for G
    G = np.zeros((m, 2))

    # apply Givens rotations to H
    for j in range(n):
        for i in range(j, m):
            if R[i, j] != 0:
                c, s = givens_rotation(R[j, j], R[i, j])
                G[i, :] = [c, s]

                # apply Givens rotation to R
                R[j:j+2, :] = np.dot(G[j:j+2, :].reshape(2, 2), R[j:j+2, :])

    return G, R

def givens_rotation(a, b):
    """
    Compute the Givens rotation matrix.
    """
    if b == 0:
        c = np.sign(a)
        s = 0
    elif a == 0:
        c = 0
        s = np.sign(b)
    else:
        if np.abs(b) > np.abs(a):
            tau = -a / b
            s = 1 / np.sqrt(1 + tau ** 2)
            c = s * tau
        else:
            tau = -b / a
            c = 1 / np.sqrt(1 + tau ** 2)
            s = c * tau

    return c, s

def form_q(G):
    """
    Retrieve the matrix Q from the Given's rotations.
    """
    # get size of the matrix
    m, n