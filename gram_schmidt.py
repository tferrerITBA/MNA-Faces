import numpy as np

EPSILON = 1e-1


def gram_schmidt(A):
    (m, n) = A.shape
    R = np.zeros(shape=(n, n))
    Q = np.zeros(shape=(m, n))

    for k in range(0, n):
        R[k, k] = np.linalg.norm(A[0:m, k])
        Q[0:m, k] = A[0:m, k] / R[k, k]

        for j in range(k + 1, n):
            R[k, j] = np.dot(np.transpose(Q[0:m, k]), A[0:m, j])
            A[0:m, j] = A[0:m, j] - np.dot(Q[0:m, k], R[k, j])

    return Q, R


# A = np.array([[8., -2., 1.], [-1., 4., 0.], [1., -1., 2.]])
# for i in range(100):
#     Q,R = gram_schmidt(A)
#     A = np.dot(R, Q)
#
# print(Q)


def cmp_eigen(old_R, new_R):
    for i in range(0, old_R.shape[0]):
        if abs(old_R[i][i] - new_R[i][i]) > EPSILON:
            print(abs(old_R[i][i] - new_R[i][i]))
            return False

    return True
