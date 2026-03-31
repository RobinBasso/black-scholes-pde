import matplotlib.pyplot as plt
import numpy as np

from src.solver import solve_crank_nicolson, solve_explicit
from src.utils import black_scholes_call


def convergence_study():
    # Parameters
    S_max = 200
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    # Grid sizes
    M_values = [25, 50, 100, 200]

    errors_cn = []
    errors_explicit = []

    for M in M_values:
        # Key difference in scaling
        N_cn = M  # CN doesn't need large N
        N_explicit = M * M * 2  # Explicit needs much finer time resolution

        # Solve
        S, price_cn = solve_crank_nicolson(S_max, K, T, r, sigma, M, N_cn)
        S, price_explicit = solve_explicit(S_max, K, T, r, sigma, M, N_explicit)

        # Analytical solution
        price_exact = black_scholes_call(S, K, T, r, sigma)

        # Errors
        error_cn = np.max(np.abs(price_cn - price_exact))
        error_explicit = np.max(np.abs(price_explicit - price_exact))

        errors_cn.append(error_cn)
        errors_explicit.append(error_explicit)

        print(
            f"M={M:4d} | CN error={error_cn:.5e} | Explicit error={error_explicit:.5e}"
        )

    # --- Plot ---
    plt.figure(figsize=(8, 6))

    plt.loglog(M_values, errors_cn, "o-", label="Crank–Nicolson", linewidth=2)
    plt.loglog(M_values, errors_explicit, "s-", label="Explicit", linewidth=2)

    # Reference slope O(M^-2)
    ref = [errors_cn[0] * (M_values[0] / m) ** 2 for m in M_values]
    plt.loglog(M_values, ref, "--", label="O(M⁻² reference)")

    plt.xlabel("Number of spatial points (M)")
    plt.ylabel("Max Error")
    plt.title("Convergence Study: Explicit vs Crank–Nicolson")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("tests/convergence_explicit_CN.png")
    plt.close()


if __name__ == "__main__":
    convergence_study()
