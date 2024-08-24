import numpy as np

def gershgorin(A):
    λ_min, λ_max = 0, 0

    r = []
    c_max = A[0][0]
    c_min = A[0][0]
    m = A.shape[0]
    for i in range(m):
        r.append(0)
        for j in range(m):
            if i != j:
                r[i] += np.absolute(A[i][j])

            if i == j:
                if A[i][i] > c_max:
                    c_max = A[i][i]
                if A[i][i] == c_max:
                    λ_max = c_max + r[i]

            if i == j:
                if A[i][i] < c_min:
                    c_min = A[i][i]
                if A[i][i] == c_min:
                    λ_min = c_min - r[i]

    return λ_min, λ_max


def power(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    k = 1
    for k in iter(int, 1):
        w = np.matmul(A, np.transpose(v))
        v = w / np.linalg.norm(w)
        λ = np.matmul(np.matmul(np.transpose(v), A), v)
        err.append(np.linalg.norm((np.matmul(A, np.transpose(v))) - (λ * np.transpose(v)), np.inf))

        if err[k - 1] <= 10 ** -13:
            break

    return v, λ, err


def inverse(A, v0, μ):
    v = v0.copy()
    λ = 0
    err = []

    k = 1
    m = A.shape[0]
    for k in iter(int, 1):
        w = np.matmul(np.linalg.inv(A - μ * np.identity(m)), np.transpose(v))
        v = w / np.linalg.norm(w)
        λ = np.matmul(np.matmul(np.transpose(v), A), v)
        err.append(np.linalg.norm(np.matmul(A, np.transpose(v)) - (λ * np.transpose(v)), np.inf))

        if err[k - 1] <= 10 ** -13:
            break

    return v, λ, err


def rayleigh(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    m = A.shape[0]
    λ = np.matmul(np.matmul(np.transpose(v), A), v)
    for k in iter(int, 1):
        w = np.matmul(np.linalg.inv(A - λ * np.identity(m)), np.transpose(v))
        v = w / np.linalg.norm(w)
        λ = np.matmul(np.matmul(np.transpose(v), A), v)
        err.append(np.linalg.norm(np.matmul(A, np.transpose(v)) - (λ * np.transpose(v)), np.inf))

        if err[k - 1] <= 10 ** -13:
            break

    return v, λ, err


def randomInput(m):
    #! DO NOT CHANGE THIS FUNCTION !#
    A = np.random.rand(m, m) - 0.5
    A += A.T  # make matrix symmetric
    v0 = np.random.rand(m) - 0.5
    v0 = v0 / np.linalg.norm(v0)  # normalize vector
    return A, v0


if __name__ == '__main__':
    A, v0 = randomInput(5)
    A = np.array([[14, 0, 1],
                  [-3, 2, -2],
                  [5, -3, 3]])
    v0 = np.array([1, 1, 1])
    μ = 10
    print("A = ", A)
    print("V =", v0)
    print("µ =", μ)

    λ_min, λ_max = gershgorin(A)
    print("\nGershgorin circle :\nλ_min = {}, \tλ_max = {}".format(λ_min, λ_max))

    print("\nPower Iteration Method :")
    v, λ, err = power(A, v0)
    print("λ = {}, \terr = {}".format(λ, err[-1]))

    print("\nInverse Iteration Method :")
    v, λ, err = inverse(A, v0, μ)
    print("λ = {}, \terr = {}".format(λ, err[-1]))

    print("\nRayleigh Quotient Iteration Method :")
    v, λ, err = rayleigh(A, v0)
    print("λ = {}, \terr = {}".format(λ, err[-1]))
