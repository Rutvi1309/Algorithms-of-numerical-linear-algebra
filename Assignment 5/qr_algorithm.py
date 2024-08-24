import numpy as np
from scipy.linalg import hilbert
import matplotlib.pyplot as pl

def sign_func(x1):
    "Sign function: Returns +1 if input is greater than zero, -1 otherwise"
    return (1 if x1>0 else -1)
def tridiag(A):
    m,n = np.shape(A)
    if n!= m:
        return ("Error: Input matrix is not symmetric") 
    
    for k in range(0,m-2):
        x_k = np.copy(A[k+1:,k]) 
        x_k_norm = np.linalg.norm(x_k,ord=2) 
        sign = sign_func(x_k[0]) 
        e_k = np.zeros((m-k-1,1)) 
        e_k[0] = 1 
        v_k = e_k*sign*x_k_norm + x_k.reshape(-1,1) 
        v_k = v_k/(np.linalg.norm(v_k,ord=2)) 
        A_temp1 = np.dot(v_k.T,A[k+1:,k:])
        A[k+1:,k:] = A[k+1:,k:] - 2*np.dot(v_k,A_temp1)
        A_temp2 = np.dot(A[:,k+1:],v_k)
        A[:,k+1:] = A[:,k+1:] - 2*np.dot(A_temp2,v_k.T) 
        A = (A + A.T)/2 
        flag = np.absolute(A) < 1e-12 
        A[flag] = 0
    return A


def QR_alg(T):
    t = []
    m,_ = np.shape(T)
    t.append(np.absolute(T[m-1,m-2]))
    while np.absolute(T[m-1,m-2]) >= 1e-12: 
        Q, R = np.linalg.qr(T)
        T = np.dot(R,Q)
        t.append(np.absolute(T[m-1,m-2]))
        flag = np.absolute(T) < 1e-12 
        T[flag] = 0
        T = (T+T.T)/2 
    return (T, t)


def wilkinson_shift(T):
    Λ = 0
    m,_ = np.shape(T)
    B = np.copy(T[m-2:,m-2:])
    delta = (B[0,0] - B[1,1])*0.5
    sign = sign_func(delta)
    denom = np.absolute(delta) + np.sqrt(delta**2+B[0,1]**2)
    Λ = B[1,1] - (sign*B[0,1]**2)/denom
    return Λ


def QR_alg_shifted(T):
    t = []
    m,_ = np.shape(T)
    t.append(np.absolute(T[m-1,m-2]))
    while np.absolute(T[m-1,m-2]) >= 1e-12:
        Λ = wilkinson_shift(T)
        TΛ = T - Λ*np.identity(m)
        Q,R = np.linalg.qr(TΛ)
        T = np.dot(R,Q) + Λ*np.identity(m)
        t.append(np.absolute(T[m-1,m-2]))
        flag = np.absolute(T) < 1e-12
        T[flag] = 0 
        T = (T+T.T)/2 
    return (T, t)


def QR_alg_driver(A, shift):
    all_t = []
    Λ = []
    T = tridiag(A)
    m_rows, n_cols = T.shape
    
    if shift == True:
         while m_rows > 0:
            if m_rows == 1:
                Λ.append(T[0, 0])
                break
            else:
                T, t = QR_alg_shifted(T)
                Λ.append(T[-1, -1])
                all_t.extend(t)
                m_rows -= 1
                T = T[0:m_rows, 0:m_rows]
                
    else:
        while m_rows > 0:
            if m_rows == 1:
                Λ.append(T[0, 0])
                break
            else:
                T, t = QR_alg(T)
                Λ.append(T[-1, -1])
                all_t.extend(t)
                m_rows -= 1
                T = T[0:m_rows, 0:m_rows]
    return (Λ, all_t)


if __name__ == "__main__":

    matrices = {
        "hilbert": hilbert(5),
        "diag(1,2,3,4)+ones": np.diag([1, 2, 3, 4]) + np.ones((4, 4)),
        "diag(5,6,7,8)+ones": np.diag([5, 6, 7, 8]) + np.ones((4, 4)),
    }

    fig, ax = pl.subplots(len(matrices.keys()), 2, figsize=(10, 10))

    for i, (mat, A) in enumerate(matrices.items()):
        for j, shift in enumerate([True, False]):

            Λ, conv = QR_alg_driver(A, shift)

            ax[i, j].semilogy(range(len(conv)), conv, ".-")
            ax[i, j].set_title(f"A = {mat}, shift = {shift}")

    pl.show()