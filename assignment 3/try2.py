# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:40:08 2023

@author: rutvishah
"""

import numpy as np
from numpy.testing import assert_allclose

def givens_rotation(a, b):
    if a == 0:
        return 0 ,1 
    else:
      r = np.sqrt(np.abs(a)**2 + np.abs(b)**2)
      c = a /abs(a)*np.conj(a)/ r
      s = -b /abs(b)*np.conj(b)/ r
      return c, s

def givens_qr(H):
    dtype = np.complex128 if np.iscomplexobj(H) else np.float64
    m_plus_1, m = H.shape

    R = H.astype(dtype)
    G = np.zeros((m, 2), dtype=dtype)

    for k in range(m - 1):
        # Compute Givens rotation parameters
        ck, sk = givens_rotation(R[k, k], R[k+1, k])

        # Apply Givens rotation to the submatrix R[k:k+2, k:m]
        Givens_matrix = np.array([[ck, -sk], [sk, ck]], dtype=dtype)
        R[k:k+2, k:m] = np.dot(Givens_matrix, R[k:k+2, k:m])

        # Store Givens rotation parameters in matrix G
        G[k, :] = [ck, sk]

    return R, G

# Example usage with complex numbers:
H = np.array([[1,2,3],[4,5,6],[0,7,8],[0,0,9]])


print("Original Upper Hessenberg Matrix:")
print(H)

# Compute QR factorization using Givens rotations
R, G = givens_qr(H)

print("\nUpper Triangular Matrix R:")
print(R)

print("\nMatrix G (Givens Rotation Parameters):")
print(G)

def form_q(G, dtype=np.float64):
    m, _ = G.shape
    Q = np.eye(m + 1, dtype=dtype)

    for i in range(m):
        c, s = G[i, 0], G[i, 1]

        # Construct the 2x2 Givens rotation matrix
        G_i = np.array([[c, -s], [s, c]], dtype=dtype)

        # Apply the Givens rotation to the corresponding columns in Q
        Q[i:i+2, :] = np.dot(G_i.T.conj(), Q[i:i+2, :])

    return Q







Q = form_q(G, dtype=np.float64)  # Use np.float64 or the desired data type


print("Unitary Matrix Q:")
print(Q)


def check_form_q(self, m, n):
    A = np.random.randint(1, 10, size=(m, n))
    H = np.triu(A)
    R, G = givens_qr(H)
    Q = form_q(G, dtype=np.float64)

    # Print matrices for debugging
    print("Original Matrix A:")
    print(A)
    print("\nUpper Triangular Matrix R:")
    print(R)
    print("\nMatrix G (Givens Rotation Parameters):")
    print(G)
    print("\nUnitary Matrix Q:")
    print(Q)

    # Check unitarity
    assert_allclose(np.dot(Q.T.conj(), Q), np.eye(m + 1), rtol=1e-7, atol=1e-14, err_msg="form_q: Q is not unitary")
