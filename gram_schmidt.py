import numpy as np

def gram_schmidt(A):
    (m, n) = A.shape
    R = np.zeros(shape=(n, n))
    Q = np.zeros(shape=(n, n))

    for k in range(0, n):
        print("m: {}; k: {}; A[0:m,k] {}\n\n".format(m, k, A[0:m,k]))
        R[k,k] = np.linalg.norm(A[0:m,k])
        Q[0:m,k] = A[0:m,k] / R[k,k]

        for j in range(k+1, n):
            print("Q[0:m,k]: {}; A[0:m,j]: {}\n".format(Q[0:m,k], A[0:m,j]))
            print("Result: {}\n".format(np.dot(np.transpose(Q[0:m,k]), A[0:m,j])))
            R[k,j] = np.dot(np.transpose(Q[0:m,k]), A[0:m,j])
            A[0:m,j] = A[0:m,j] - Q[0:m,k] * R[k,j]

    return (Q,R)

    print("{}".format(R))


A = np.ones(shape=(4,4))
gram_schmidt(A)
