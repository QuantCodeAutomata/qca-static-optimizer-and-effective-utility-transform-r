"""Monte Carlo simulation of dual process and optimal controls."""
import numpy as np
from typing import Callable, Tuple

def simulate_Y_exact(
    y0: float, kappa: float, r: float, rho: float,
    T: float, n_steps: int, n_paths: int,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact simulation of Y_t = y0*exp[(rho-r-kappa^2/2)*t - kappa*W_t]."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    times = np.linspace(0, T, n_steps + 1)
    
    drift = rho - r - kappa**2 / 2  # log-drift
    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    
    log_Y = np.zeros((n_paths, n_steps + 1))
    log_Y[:, 0] = np.log(y0)
    
    for t in range(n_steps):
        log_Y[:, t+1] = log_Y[:, t] + drift * dt - kappa * dW[:, t]
    
    Y_paths = np.exp(log_Y)
    return times, Y_paths


def brownian_bridge_min(
    y_start: float, y_end: float, t_start: float, t_end: float,
    kappa: float, rng: np.random.Generator = None
) -> float:
    """Sample running minimum of lognormal Y on [t_start,t_end] via Brownian bridge.
    
    For a BM W_t on [0,T] with W_0=a, W_T=b, the minimum has distribution:
    P(min W < m) = exp(-2*(a-m)*(b-m)/(T)) for m < min(a,b).
    Here transform Y to log-space.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    dt = t_end - t_start
    a = np.log(y_start)  # log(Y_start)
    b = np.log(y_end)    # log(Y_end)
    
    # Sample minimum of Brownian bridge from a to b over [0,dt]
    # P(min > m) = 1 - exp(-2*(a-m)*(b-m)/dt) for m < min(a,b)
    # Invert: U ~ Uniform(0,1), m = (a+b - sqrt((a-b)^2 - 2*dt*log(1-U)))/2
    U = rng.uniform()
    min_ab = min(a, b)
    discriminant = (a - b)**2 - 2 * dt * np.log(max(1 - U, 1e-15))
    if discriminant < 0:
        return np.exp(min_ab)
    log_min = (a + b - np.sqrt(discriminant)) / 2
    return np.exp(log_min)


def simulate_habit_path(
    Y_paths: np.ndarray, times: np.ndarray,
    h0: float,
    z_fn: Callable[[float], float],
    z_inv_fn: Callable[[float], float]
) -> np.ndarray:
    """Construct H_t = z_inv(min{z(h0), min_{s<=t} Y_s})."""
    n_paths, n_steps_plus1 = Y_paths.shape
    H_paths = np.zeros_like(Y_paths)
    H_paths[:, 0] = h0
    
    z_h0 = z_fn(h0)
    
    for p in range(n_paths):
        running_min_Y = Y_paths[p, 0]
        min_z = z_h0
        
        for t in range(1, n_steps_plus1):
            running_min_Y = min(running_min_Y, Y_paths[p, t])
            current_min_z = min(z_h0, running_min_Y)
            H_paths[p, t] = z_inv_fn(current_min_z)
    
    return H_paths


def estimate_primal_utility(
    c_paths: np.ndarray, H_paths: np.ndarray, prim,
    times: np.ndarray, rho: float
) -> Tuple[float, float]:
    """E[int_0^T e^{-rho*t} u(D(H_t)*c_t - L(H_t)) dt] via trapezoidal rule."""
    n_paths = c_paths.shape[0]
    discount = np.exp(-rho * times)
    utilities = np.zeros_like(c_paths)
    
    for p in range(n_paths):
        xi = prim.D(H_paths[p]) * c_paths[p] - prim.L(H_paths[p])
        xi = np.maximum(xi, 1e-15)  # numerical floor
        utilities[p] = discount * prim.u(xi)
    
    # Trapezoidal integration per path
    path_integrals = np.trapz(utilities, times, axis=1)
    mean = np.mean(path_integrals)
    stderr = np.std(path_integrals) / np.sqrt(n_paths)
    return mean, stderr


def estimate_budget_usage(
    Y_paths: np.ndarray, c_paths: np.ndarray,
    times: np.ndarray, r: float, kappa: float, rho: float,
    y0: float
) -> Tuple[float, float]:
    """Estimate E[int_0^T K_t c_t dt] where K_t = exp[(-r-kappa^2/2)t - kappa*W_t].
    Since Y_t = y0*exp[(rho-r-kappa^2/2)t - kappa*W_t], K_t = Y_t/y0 * exp[-rho*t].
    """
    n_paths = c_paths.shape[0]
    K_paths = Y_paths / y0 * np.exp(-rho * times)
    integrands = K_paths * c_paths
    path_integrals = np.trapz(integrands, times, axis=1)
    mean = np.mean(path_integrals)
    stderr = np.std(path_integrals) / np.sqrt(n_paths)
    return mean, stderr
