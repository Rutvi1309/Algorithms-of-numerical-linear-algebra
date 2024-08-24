import numpy as np
from scipy.linalg import solve_triangular
from numpy.linalg import qr

def cg(A, b, tol=1e-12):
    m = A.shape[0]
    x = np.zeros(m, dtype=A.dtype)
    r = b
    p = r.copy()
    r_b = []
    for k in range(m):
        alpha = np.dot(r, r) / np.dot(p, np.dot(A, p))
        x += alpha * p
        r_new = r - alpha * np.dot(A, p)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        r_b.append(np.linalg.norm(r) / np.linalg.norm(b))
        if np.linalg.norm(r) / np.linalg.norm(b) < tol:
            break

    return x, r_b


def arnoldi_n(A, Q, P):
    m, n = Q.shape
    q = np.zeros(m, dtype=Q.dtype)
    h = np.zeros(n + 1, dtype=A.dtype)
    q = Q[:, n - 1]
    v = np.dot(A, q)
    for j in range(n):
        h[j] = np.dot(Q[:, j], v)
        v -= h[j] * Q[:, j]
    h[n] = np.linalg.norm(v, 2)
    if h[n] == 0:
        raise ValueError("Dividing by 0 is not possible! Check input.")
    else: 
        q = v / h[n]
    return h, q


def gmres(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)

    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    n = 100  
    H = np.zeros((n + 1, n))
    V = np.zeros((n + 1, m))
    r0 = b - solve_triangular(P, A @ x, lower=False)
    beta = np.linalg.norm(r0)
    V[0] = r0 / beta

    for j in range(n):
        w = solve_triangular(P, A @ V[j], lower=False)

        # Gram-Schmidt orthogonalization
        for i in range(j + 1):
            H[i, j] = np.dot(w, V[i])
            w -= H[i, j] * V[i]
        H[j + 1, j] = np.linalg.norm(w)

        # Add new vector to basis
        V[j + 1] = w / H[j + 1, j]

        # Least squares fit
        e1 = np.zeros(j + 2)
        e1[0] = beta
        Q, R = qr(H[:j + 2, :j + 1], mode='reduced')
        y = solve_triangular(R, Q.T @ e1)[:j + 1]

        # Update solution
        x_new = x + V[:j + 1].T @ y

        # Check convergence
        residual = np.linalg.norm(solve_triangular(P, A @ x_new, lower=False) - b)
        r_b.append(residual / np.linalg.norm(b))
        if residual < tol:
            return x_new, r_b

    return x, r_b


def gmres_givens(A, b, P=np.eye(0), tol=1e-12):
    m = A.shape[0]
    if P.shape != A.shape:
        P = np.eye(m)
    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]
    Q = np.zeros([m, m], dtype=A.dtype)
    H = np.zeros([m + 1, m], dtype=A.dtype)
    Q[:, 0] = b / np.linalg.norm(b, 2)
    r = b
    r_b.append(np.linalg.norm(r, 2) / np.linalg.norm(b, 2))
    
    def givens_rotation(a, b):
        r = np.hypot(a, b)
        c = a / r
        s = -b / r
        return c, s

    def apply_givens_rotation(H, cs, sn, k):
        temp = cs * H[k] - sn * H[k + 1]
        H[k + 1] = sn * H[k] + cs * H[k + 1]
        H[k] = temp
        return H
    
    m = A.shape[0]
    if P.shape != A.shape:
        # default preconditioner P = I
        P = np.eye(m)

    x = np.zeros(m, dtype=b.dtype)
    r_b = [1]

    # Initialize variables
    n_iter = 100  # You can adjust this based on the problem
    H = np.zeros((n_iter + 1, n_iter))
    V = np.zeros((n_iter + 1, m))
    r0 = b - solve_triangular(P, A @ x, lower=False)
    beta = np.linalg.norm(r0)
    V[0] = r0 / beta

    for j in range(n_iter):

        # Givens rotation
        denom = np.sqrt(H[j, j] ** 2 + H[j + 1, j] ** 2)

        if np.abs(denom) < 1e-12:
            cs = 1.0
            sn = 0.0
        else:
            cs = H[j, j] / denom
            sn = H[j + 1, j] / denom

        # Apply Givens rotation to H
        H = apply_givens_rotation(H, cs, sn, j)

        # Apply Givens rotation to V
        temp = cs * V[j, j] - sn * V[j + 1, j]
        V[j + 1, j] = sn * V[j, j] + cs * V[j + 1, j]
        V[j, j] = temp

        # Least squares fit
        e1 = np.zeros(j + 2)
        e1[0] = beta

        try:
            # Attempt to solve triangular system
            y = solve_triangular(H[:j + 1, :j + 1], e1[:j + 1])
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            print(f"Skipping iteration at {j} due to Singularity")
            continue

        x_new = x + V[:j + 1].T @ y
        residual = np.linalg.norm(solve_triangular(P, A @ x_new, lower=False) - b)
        r_b.append(residual / np.linalg.norm(b))
        if residual < tol:
            return x_new, r_b

    return x, r_b