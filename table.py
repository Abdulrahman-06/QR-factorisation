import numpy as np

def classical_gram_schmidt(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

if __name__ == "__main__":
    eps_values = [10.0**(-k) for k in range(6, 17)]

    print(f"{'eps':>8}  {'error1':>12}  {'error2':>12}  {'error3':>12}")

    for eps in eps_values:
        A = np.array([[1.0,      1.0 + eps],
                      [1.0 + eps, 1.0      ]])

        Q, R = classical_gram_schmidt(A)

        error1 = np.linalg.norm(A - Q @ R, 2)
        error2 = np.linalg.norm(Q.T @ Q - np.eye(2), 2)
        error3 = np.linalg.norm(R - np.triu(R), 2)

        print(f"{eps:8.0e}  {error1:12.4e}  {error2:12.4e}  {error3:12.4e}")
