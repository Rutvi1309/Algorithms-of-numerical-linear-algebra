import numpy as np

def givens_qr(H: np.ndarray):
    """
    Perform QR decomposition of a matrix using Givens rotations.

    Parameters:
    - H (numpy.ndarray): Input matrix.

    Returns:
    - Q (numpy.ndarray): Orthogonal matrix Q from QR decomposition.
    - R (numpy.ndarray): Upper triangular matrix R from QR decomposition.
    """
    # Convert H into float if necessary
    H = H + 0.0
    m = np.shape(H)[1]

    Q = np.zeros((m, 2)).astype(H.dtype)
    k = 0
    R = np.copy(H)

    for i in range(m):
        x = R[i, i]
        y = R[i + 1, i]

        # Compute matrix entries for Givens rotation
        r = np.sqrt(abs(x) ** 2 + abs(y) ** 2)
        c = x.conjugate() / r
        s = y.conjugate() / r

        if y != 0:
            # Construct Givens rotation matrix
            G = np.identity(2).astype(c.dtype)
            G[0] = c, s
            G[1, 0] = -s.conjugate()
            G[1, 1] = c.conjugate()

            # Update R with Givens rotation
            R[i:i + 2, i:] = G @ R[i:i + 2, i:]

            # Store Givens rotation parameters in Q
            Q[k] = c, s
        else:
            # If y is zero, set Q to the identity matrix
            Q[k] = 1, 0

        k += 1

    return Q, R

def form_q(G):
    """
    Form the orthogonal matrix Q from Givens rotation parameters.

    Parameters:
    - G (numpy.ndarray): Givens rotation parameters.

    Returns:
    - Q (numpy.ndarray): Orthogonal matrix Q.
    """
    m = G.shape[0]
    Q = np.identity(m + 1).astype(G.dtype)

    for i in range(m):
        # Construct Givens rotation matrix from parameters
        K = np.identity(2).astype(G.dtype)
        K[:, 0] = G[i].conjugate()
        K[0, 1] = -G[i, 1]
        K[1, 1] = G[i, 0]

        # Update Q with Givens rotation
        Q[:, i:i + 2] = Q[:, i:i + 2] @ K

    return Q
