import matplotlib.pyplot as plt
import numpy as np

from src.solver import solve_crank_nicolson, solve_explicit
from src.utils import black_scholes_call


def run_test():
    # Parameters
    S_max = 200
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    M = 100
    N = 1000

    # Numerical solution
    S, price_explicit = solve_explicit(S_max, K, T, r, sigma, M, N)
    S, price_cn = solve_crank_nicolson(S_max, K, T, r, sigma, M, N)
    # Analytical solution
    price_exact = black_scholes_call(S, K, T, r, sigma)

    # Plot comparison
    plt.figure()
    plt.plot(S, price_explicit, label="Explicit")
    plt.plot(S, price_cn, label="Crank-Nicolson")
    plt.plot(S, price_exact, "--", label="Analytical")
    plt.legend()
    plt.title("Method Comparison")
    plt.xlabel("Stock Price")
    plt.ylabel("Option Price")
    plt.savefig("tests/comparison_explicit_CN_analytical.png")
    plt.close()

    # Error
    error_explicit = np.abs(price_explicit - price_exact)
    error_cn = np.abs(price_cn - price_exact)

    plt.figure()
    plt.plot(S, error_explicit, label="Explicit Error")
    plt.plot(S, error_cn, label="CN Error")

    plt.legend()
    plt.title("Error Comparison")
    plt.xlabel("Stock Price")
    plt.ylabel("Error")
    plt.savefig("tests/error_explicit_CN.png")
    plt.close()

    print("Max error explicit:", np.max(error_explicit))
    print("Max error CN:", np.max(error_cn))


if __name__ == "__main__":
    run_test()
