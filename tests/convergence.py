import matplotlib.pyplot as plt
import numpy as np

from src.solver import solve_crank_nicolson, solve_explicit, solve_rannacher
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
    plt.figure()
    plt.loglog(M_values, errors_cn, "o-", label="Crank–Nicolson", linewidth=2)
    plt.loglog(M_values, errors_explicit, "s-", label="Explicit", linewidth=2)
    # Reference slope O(M^-2)
    ref = [errors_cn[0] * (M_values[0] / m) ** 2 for m in M_values]
    plt.loglog(M_values, ref, "--", label=r"$O(M^{-2}$)")
    plt.xlabel("Number of spatial points (M)")
    plt.ylabel("Max Error")
    plt.title("Spatial Convergence")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("tests/spatial_convergence_explicit_CN.png")
    plt.close()


def time_convergence():
    S_max = 200
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    M = 50  # FIXED spatial resolution
    dt_stab = 1 / (sigma**2 * M**2)

    # N_values = [50, 100, 200, 400, 800]
    N_values = [
        int(T / dt_stab),
        int(2 * T / dt_stab),
        int(4 * T / dt_stab),
        int(8 * T / dt_stab),
        int(16 * T / dt_stab),
    ]

    errors_cn = []
    errors_explicit = []

    for N in N_values:
        S, price_cn = solve_crank_nicolson(S_max, K, T, r, sigma, M, N)
        S, price_explicit = solve_explicit(S_max, K, T, r, sigma, M, N)

        price_exact = black_scholes_call(S, K, T, r, sigma)

        error_cn = np.max(np.abs(price_cn - price_exact))
        error_explicit = np.max(np.abs(price_explicit - price_exact))

        errors_cn.append(error_cn)
        errors_explicit.append(error_explicit)
        print(
            f"N={N:4d} | CN error={error_cn:.5e} | Explicit error={error_explicit:.5e}"
        )

    # --- Plot ---
    plt.figure()
    plt.loglog(N_values, errors_cn, "o-", label="CN")
    plt.loglog(N_values, errors_explicit, "s-", label="Explicit")

    # Reference slopes
    ref1 = [errors_explicit[0] * (N_values[0] / n) for n in N_values]
    ref2 = [errors_cn[0] * (N_values[0] / n) ** 2 for n in N_values]

    plt.loglog(N_values, ref1, "--", label=r"$O(dt)$")
    plt.loglog(N_values, ref2, "--", label=r"$O(dt^2)$")

    plt.xlabel("Number of time steps (N)")
    plt.ylabel("Max Error")
    plt.title("Time Convergence")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("tests/time_convergence_explicit_CN.png")
    plt.close()


def time_convergence_rannacher():
    S_max = 200
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    M = 200

    N_values = [100, 200, 400, 800, 1600]

    errors = []

    for N in N_values:
        S, price = solve_rannacher(S_max, K, T, r, sigma, M, N)

        price_exact = black_scholes_call(S, K, T, r, sigma)

        error = np.max(np.abs(price - price_exact))
        errors.append(error)

        print(f"N={N:4d} | Rannacher error={error:.5e}")

    plt.figure()
    plt.loglog(N_values, errors, "o-", label="Rannacher (CN + smoothing)")

    # Reference slopes
    ref1 = [errors[0] * (N_values[0] / n) for n in N_values]
    ref2 = [errors[0] * (N_values[0] / n) ** 2 for n in N_values]
    plt.loglog(N_values, ref1, "--", label=r"$O(dt)$")
    plt.loglog(N_values, ref2, "--", label=r"$O(dt^2)$")

    plt.xlabel("Number of time steps (N)")
    plt.ylabel("Max Error")
    plt.title("Time Convergence with Rannacher Smoothing")

    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("tests/time_convergence_rannacher.png")
    plt.close()


if __name__ == "__main__":
    # convergence_study()
    # time_convergence()
    time_convergence_rannacher()
