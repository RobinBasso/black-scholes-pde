import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Analytical Black-Scholes price for European call
    """

    S = np.array(S)
    price = np.zeros_like(S)

    # Avoid log(0) by masking
    mask = S > 0

    S_pos = S[mask]

    d1 = (np.log(S_pos / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price[mask] = S_pos * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    # For S = 0 → price = 0 (already default)

    return price
