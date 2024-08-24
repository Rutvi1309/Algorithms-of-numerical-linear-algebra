import numpy as np


def gershgorin(A):
    λ_min, λ_max = float('inf'),float('-inf')
    m = A.shape[0]
    λ_min_dict = {}  
    λ_max_dict = {}  

    for i in range(m):
        C = A[i, i]
        R = np.sum(np.abs(A[i, :])) - np.abs(C)  
        λ_min_dict[i] = C - R
        λ_max_dict[i] = C + R

    λ_min = min(λ_min_dict.values())
    λ_max = max(λ_max_dict.values())

    return λ_min, λ_max


def power(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    while True:
        Av = A @ v
        λ_new = np.dot(v, Av)
        err.append(np.max(np.abs(Av - λ_new * v)))

        if np.linalg.norm(Av - λ_new * v, np.inf) <= 1e-13:
            break

        v = Av / np.linalg.norm(Av)
        λ = λ_new

    return v, λ, err


def inverse(A, v0, μ):
    v = v0.copy()
    λ = 0
    err = []

    while True:
        v_new = np.linalg.solve(A - µ * np.eye(A.shape[0]), v)
        v_new /= np.linalg.norm(v_new)
        λ_new = v_new.T @ A @ v_new # Update λ
        error = np.linalg.norm(A @ v_new - λ_new * v_new, np.inf) # Compute error
        err.append(error)

        if error <= 1e-13:
            break

        v = v_new
        λ = λ_new

    return v, λ, err


def rayleigh(A, v0):
    v = v0.copy()
    λ = 0
    err = []

    λ = v.T @ A @ v
    while True:
        v_new = np.linalg.solve(A - λ * np.eye(A.shape[0]), v)
        v_new /= np.linalg.norm(v_new)

        λ_new = v_new.T @ A @ v_new
        error = np.linalg.norm(A @ v_new - λ_new * v_new, np.inf)
        err.append(error)

        if error <= 1e-13:
            break

        v = v_new
        λ = λ_new

    return v, λ, err


def randomInput(m):
    #! DO NOT CHANGE THIS FUNCTION !#
    A = np.random.rand(m, m) - 0.5
    A += A.T  # make matrix symmetric
    v0 = np.random.rand(m) - 0.5
    v0 = v0 / np.linalg.norm(v0) # normalize vector
    return A, v0


if __name__ == '__main__':
    pass
    #n = 5  # Size of the matrix
    A = np.array([[14, 0, 1],
                  [-3, 2,-2],
                  [ 5,-3, 3]])  # Generate a random symmetric matrix
    v0 = np.array([0, 0, 1])  # Initial eigenvector guess
    µ = 10  # Eigenvalue estimate

    # Test the gershgorin function
    λ_min, λ_max = gershgorin(A)
    print(f"Gershgorin: λ_min = {λ_min}, λ_max = {λ_max}")

    # Test the power function
    v, λ, err = power(A, v0)
    print(f"Power: v = {v}, λ = {λ}, err = {err}")

    # Test the inverse function
    v, λ, err = inverse(A, v0, µ)
    print(f"Inverse: v = {v}, λ = {λ}, err = {err}")

    # Test the rayleigh function
    v, λ, err = rayleigh(A, v0)
    print(f"Rayleigh: v = {v}, λ = {λ}, err = {err}")
