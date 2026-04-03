import numpy as np


def solve_explicit(S_max, K, T, r, sigma, M, N):
    """
    Explicit finite difference solver for Black-Scholes PDE.
    Returns asset grid S and option price at t=0.
    """
    dt = T / N

    grid = np.zeros((M + 1, N + 1))
    S = np.linspace(0, S_max, M + 1)

    # Terminal condition
    grid[:, -1] = np.maximum(S - K, 0)

    # Boundary conditions
    time = np.linspace(0, T, N + 1)
    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * (T - time))

    # Backward time stepping
    for j in reversed(range(N)):
        for i in range(1, M):
            a = 0.5 * dt * (sigma**2 * i**2 - r * i)
            b = 1 - dt * (sigma**2 * i**2 + r)
            c = 0.5 * dt * (sigma**2 * i**2 + r * i)

            grid[i, j] = (
                a * grid[i - 1, j + 1] + b * grid[i, j + 1] + c * grid[i + 1, j + 1]
            )

    return S, grid[:, 0]


def solve_crank_nicolson(S_max, K, T, r, sigma, M, N):
    import numpy as np

    dS = S_max / M
    dt = T / N

    grid = np.zeros((M + 1, N + 1))
    S = np.linspace(0, S_max, M + 1)

    # Terminal condition
    grid[:, -1] = np.maximum(S - K, 0)

    time = np.linspace(0, T, N + 1)

    # Boundary conditions
    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * (T - time))

    # Precompute coefficients
    A = np.zeros((M - 1, M - 1))
    B = np.zeros((M - 1, M - 1))

    for i in range(1, M):
        alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
        beta = -0.5 * dt * (sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

        row = i - 1

        if i > 1:
            A[row, row - 1] = -alpha
            B[row, row - 1] = alpha

        A[row, row] = 1 - beta
        B[row, row] = 1 + beta

        if i < M - 1:
            A[row, row + 1] = -gamma
            B[row, row + 1] = gamma

    # Time stepping
    for j in reversed(range(N)):
        rhs = B @ grid[1:M, j + 1]

        # Left boundary (i = 1)
        i = 1
        alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
        rhs[0] += alpha * (grid[0, j] + grid[0, j + 1])

        # Right boundary (i = M-1)
        i = M - 1
        gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)
        rhs[-1] += gamma * (grid[M, j] + grid[M, j + 1])

        grid[1:M, j] = np.linalg.solve(A, rhs)

    return S, grid[:, 0]


def solve_rannacher(S_max, K, T, r, sigma, M, N, n_smooth=4):
    """
    Crank-Nicolson with Rannacher smoothing:
    - first n_smooth steps: implicit Euler
    - then: Crank-Nicolson
    """
    import numpy as np

    dt = T / N

    grid = np.zeros((M + 1, N + 1))
    S = np.linspace(0, S_max, M + 1)

    # Terminal condition
    grid[:, -1] = np.maximum(S - K, 0)

    time = np.linspace(0, T, N + 1)

    # Boundary conditions
    grid[0, :] = 0
    grid[-1, :] = S_max - K * np.exp(-r * (T - time))

    # Build matrices (same structure as CN)
    A = np.zeros((M - 1, M - 1))
    B = np.zeros((M - 1, M - 1))
    A_imp = np.zeros((M - 1, M - 1))  # implicit Euler

    for i in range(1, M):
        alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
        beta = -0.5 * dt * (sigma**2 * i**2 + r)
        gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)

        row = i - 1

        # CN matrices
        if i > 1:
            A[row, row - 1] = -alpha
            B[row, row - 1] = alpha

        A[row, row] = 1 - beta
        B[row, row] = 1 + beta

        if i < M - 1:
            A[row, row + 1] = -gamma
            B[row, row + 1] = gamma

        # Implicit Euler matrix
        alpha_e = 0.5 * dt * (sigma**2 * i**2 - r * i)
        beta_e = -dt * (sigma**2 * i**2 + r)
        gamma_e = 0.5 * dt * (sigma**2 * i**2 + r * i)

        if i > 1:
            A_imp[row, row - 1] = -alpha_e
        A_imp[row, row] = 1 - beta_e
        if i < M - 1:
            A_imp[row, row + 1] = -gamma_e

    # Time stepping
    for j in reversed(range(N)):
        # --- Rannacher smoothing phase ---
        if j >= N - n_smooth:
            rhs = grid[1:M, j + 1].copy()

            # boundary corrections
            rhs[0] += (
                0.5 * dt * (sigma**2 * 1**2 - r * 1) * (grid[0, j] + grid[0, j + 1])
            )
            rhs[-1] += (
                0.5
                * dt
                * (sigma**2 * (M - 1) ** 2 + r * (M - 1))
                * (grid[M, j] + grid[M, j + 1])
            )

            grid[1:M, j] = np.linalg.solve(A_imp, rhs)

        # --- Crank-Nicolson phase ---
        else:
            rhs = B @ grid[1:M, j + 1]

            # boundary corrections
            i = 1
            alpha = 0.25 * dt * (sigma**2 * i**2 - r * i)
            rhs[0] += alpha * (grid[0, j] + grid[0, j + 1])

            i = M - 1
            gamma = 0.25 * dt * (sigma**2 * i**2 + r * i)
            rhs[-1] += gamma * (grid[M, j] + grid[M, j + 1])

            grid[1:M, j] = np.linalg.solve(A, rhs)

    return S, grid[:, 0]


if __name__ == "__main__":
    S, price = solve_explicit(
        S_max=200,
        K=100,
        T=1.0,
        r=0.05,
        sigma=0.2,
        M=100,
        N=1000,
    )
