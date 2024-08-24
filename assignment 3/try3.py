import numpy as np

def givens_qr(H):
    H = H + 0.0
    dtype = np.complex128 if np.iscomplexobj(H) else np.float64
    m = np.shape(H)[1]
    k = 0
    R = np.copy(H)
    Q = np.zeros((m, 2)).astype(dtype)
    G = np.zeros((m, 2)).astype(dtype)

    def givens_rotation(a, b):
        r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        c = np.conj(a) / r
        s = np.conj(b) / r
        return c, s

    for i in range(m):
        a = R[i, i]
        b = R[i + 1, i]

        c, s = givens_rotation(a, b)

        if b != 0:
            G = np.identity(2).astype(dtype)
            G[0] = c, s
            G[1, 0] = -s.conjugate()
            G[1, 1] = c.conjugate()

            R[i:i + 2, i:] = G @ R[i:i + 2, i:]

            Q[k] = c, s
        else:
            Q[k] = 1, 0

        k += 1

    return Q, R

def form_q(G):
    m = G.shape[0]
    Q = np.identity(m + 1).astype(G.dtype)

    for j in range(m):
        c, s = G[j]

        K = np.identity(2).astype(G.dtype)
        K[:, 0] = np.conj(c)
        K[0, 1] = -np.conj(s)
        K[1, 1] = c

        Q[:, j:j + 2] = Q[:, j:j + 2] @ K

    return Q
