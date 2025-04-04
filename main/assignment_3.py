import numpy as np

A = np.array([
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
], dtype=float)

def gaussian_elimination(matrix):
    n = len(matrix)

    # Forward elimination
    for i in range(n):
        # Make the pivot element 1
        matrix[i] = matrix[i] / matrix[i][i]
        for j in range(i+1, n):
            matrix[j] = matrix[j] - matrix[j][i] * matrix[i]
    return matrix

def back_substitution(matrix):
    n = len(matrix)
    x = np.zeros(n)

    for i in range(n-1, -1, -1):
        x[i] = matrix[i][-1] - np.dot(matrix[i][i+1:n], x[i+1:n])
    return x

# Apply Gaussian elimination

echelon_matrix = gaussian_elimination(A.copy())

# Solve with back substitution

solution = back_substitution(echelon_matrix)

print(solution)


import numpy as np


A = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
], dtype=float)

# Manual LU decomposition using Doolittle's method (no pivoting)

def doolittle_lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i+1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

    return L, U

# Run LU decomposition

L, U = doolittle_lu_decomposition(A)

# Compute determinant: product of U's diagonal

det_A = np.prod(np.diag(U))

# Output
print("Determinant of A:", det_A)
print("\nL matrix:")
print(L)
print("\nU matrix:")
print(U)



import numpy as np


A = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
])

def is_diagonally_dominant(matrix):
    n = matrix.shape[0]
    for i in range(n):
        diagonal = abs(matrix[i, i])
        off_diagonal_sum = sum(abs(matrix[i, j]) for j in range(n) if j != i)
        if diagonal < off_diagonal_sum:
            return False
    return True


if is_diagonally_dominant(A):
    print("The matrix is diagonally dominant, True.")
else:
    print("The matrix is NOT diagonally dominant, False.")
    
    
import numpy as np


A = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
])

def is_positive_definite(matrix):
    try:
     
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False


if is_positive_definite(A):
    print("The matrix is positive definite, True.")
else:
    print("The matrix is NOT positive definite, False.")

