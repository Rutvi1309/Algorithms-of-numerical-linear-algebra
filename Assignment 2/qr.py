import numpy as np

def implicit_qr(A):
    m,n = np.shape(A)
    W = np.zeros((m, n), dtype=complex)
    R = A.copy().astype(complex)

   
    if n > m: #checking whether m>=n is False
        return("Matrix with more columns than rows. Try again")
    W = np.zeros((m, n)).astype(complex) #initialize W as a zero matrix
    R = A.copy().astype(complex) #initialize R as the copy of input matrix A

    for k in range(0,n): #starting iteration from 0
        X = np.copy(R[k:,k]) #create Xk as a submatrix of R
        ek = np.zeros((m-k,1)) 
        ek[0]=1 #Unit vector created
        V = (1 if X[0]>0 else -1)*np.linalg.norm(X,ord=2)*ek + X.reshape(-1,1)
        V = V/np.linalg.norm(V,ord=2) #Vector Vk created
        R[k:m, k:n] = R[k:m, k:n] - 2 * np.dot(V, (V.T @ R[k:m, k:n]))
        #Substituting for the subspace in R with Vk
        W[k:,[k]] = V 
    return (W, R)


def form_q(W):
    m, n = np.shape(W)
    Q = np.eye(m, dtype=complex)
    
    m,n = np.shape(W)
    if n > m: #checking whether m>=n is False
        return("The matrix is invalid!")
    Q = np.eye(m).astype(complex) #initialize Q as an Identity matrix
    for i in range(0,m): 
        for j in range(n-1,-1,-1): #Since the process starts from the final col to first col
            Q[j:, i] = Q[j:, i] - 2 * W[j:, j] * np.dot(W[j:, j].T, Q[j:, i])
    return Q