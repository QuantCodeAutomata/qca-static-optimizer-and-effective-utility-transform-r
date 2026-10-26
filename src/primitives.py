"""Model primitives for Duality of Shortfall Aversion."""
from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import brentq
from typing import Union

ArrayLike = Union[float, np.ndarray]


class ModelPrimitives(ABC):
    """Abstract base for utility function u and habit transforms D, L."""

    @abstractmethod
    def u(self, xi: ArrayLike) -> ArrayLike:
        """Utility u(xi) for xi > 0."""

    @abstractmethod
    def u_prime(self, xi: ArrayLike) -> ArrayLike:
        """u'(xi) > 0."""

    @abstractmethod
    def u_double_prime(self, xi: ArrayLike) -> ArrayLike:
        """u''(xi) < 0."""

    @property
    @abstractmethod
    def m0(self) -> float:
        """lim_{xi->0+} u'(xi), in (0, inf]."""

    @abstractmethod
    def D(self, h: ArrayLike) -> ArrayLike:
        """D(h) > 0."""

    @abstractmethod
    def D_prime(self, h: ArrayLike) -> ArrayLike:
        """D'(h) <= 0."""

    @abstractmethod
    def D_double_prime(self, h: ArrayLike) -> ArrayLike:
        """"""

    @abstractmethod
    def L(self, h: ArrayLike) -> ArrayLike:
        """L(h) >= 0."""

    @abstractmethod
    def L_prime(self, h: ArrayLike) -> ArrayLike:
        """"""

    @abstractmethod
    def L_double_prime(self, h: ArrayLike) -> ArrayLike:
        """"""

    def I(self, q: ArrayLike) -> ArrayLike:
        """Inverse marginal utility I=(u')^{-1}(q) with zero-extension I(q)=0 for q >= m0."""
        scalar_input = np.isscalar(q)
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.zeros_like(q_arr)
        for idx in np.ndindex(q_arr.shape):
            qi = q_arr[idx]
            if qi >= self.m0 - 1e-15:
                result[idx] = 0.0
            elif qi <= 0:
                result[idx] = 1e15  # u'->0 as xi->inf
            else:
                try:
                    result[idx] = brentq(
                        lambda z: self.u_prime(z) - qi, 1e-12, 1e12, xtol=1e-12
                    )
                except ValueError:
                    result[idx] = 0.0
        return float(result.squeeze()) if scalar_input else result


class ExponentialFamily(ModelPrimitives):
    """Exponential utility: u(xi)=-(1/beta)*exp(-beta*xi), D(h)=1, L(h)=lam*h.

    m0 = beta (finite). I(q) = -(1/beta)*log(q/beta) for 0<q<beta, else 0.
    """

    def __init__(self, beta: float, lam: float):
        assert beta > 0, "beta must be positive"
        assert 0 < lam < 1, "lam must be in (0,1)"
        self.beta = beta
        self.lam = lam

    def u(self, xi):
        xi = np.asarray(xi, dtype=float)
        return -(1.0 / self.beta) * np.exp(-self.beta * xi)

    def u_prime(self, xi):
        xi = np.asarray(xi, dtype=float)
        return np.exp(-self.beta * xi)

    def u_double_prime(self, xi):
        xi = np.asarray(xi, dtype=float)
        return -self.beta * np.exp(-self.beta * xi)

    @property
    def m0(self):
        return self.beta

    def D(self, h):
        return np.ones_like(np.asarray(h, dtype=float))

    def D_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def D_double_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def L(self, h):
        return self.lam * np.asarray(h, dtype=float)

    def L_prime(self, h):
        return np.full_like(np.asarray(h, dtype=float), self.lam)

    def L_double_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def I(self, q):
        """Analytical inverse: I(q) = -(1/beta)*log(q/beta) for 0<q<beta, else 0."""
        scalar_input = np.isscalar(q)
        q_arr = np.atleast_1d(np.asarray(q, dtype=float))
        result = np.where(
            (q_arr > 0) & (q_arr < self.beta),
            -(1.0 / self.beta) * np.log(q_arr / self.beta),
            0.0,
        )
        return float(result.squeeze()) if scalar_input else result


class PowerFamily(ModelPrimitives):
    """Power utility: u(xi)=xi^(1-gamma)/(1-gamma), D(h)=h^(-alpha), L(h)=0.

    gamma>1, alpha in (0,1). m0=inf. I(q)=q^(-1/gamma).
    """

    def __init__(self, gamma: float, alpha: float):
        assert gamma > 1, "gamma must be > 1"
        assert 0 < alpha < 1, "alpha must be in (0,1)"
        self.gamma = gamma
        self.alpha = alpha

    def u(self, xi):
        xi = np.asarray(xi, dtype=float)
        # u(0) = -∞ for γ > 1; numpy otherwise warns on 0 ** (negative).
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = xi ** (1 - self.gamma) / (1 - self.gamma)
        return np.where(xi > 0, raw, -np.inf)

    def u_prime(self, xi):
        xi = np.asarray(xi, dtype=float)
        return xi ** (-self.gamma)

    def u_double_prime(self, xi):
        xi = np.asarray(xi, dtype=float)
        return -self.gamma * xi ** (-self.gamma - 1)

    @property
    def m0(self):
        return np.inf

    def D(self, h):
        return np.asarray(h, dtype=float) ** (-self.alpha)

    def D_prime(self, h):
        return -self.alpha * np.asarray(h, dtype=float) ** (-self.alpha - 1)

    def D_double_prime(self, h):
        return self.alpha * (self.alpha + 1) * np.asarray(h, dtype=float) ** (-self.alpha - 2)

    def L(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def L_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def L_double_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def I(self, q):
        """Analytical: I(q) = q^(-1/gamma)."""
        q_arr = np.asarray(q, dtype=float)
        scalar_input = np.isscalar(q)
        result = q_arr ** (-1.0 / self.gamma)
        return float(result) if scalar_input else result


class AppendixCFamily(ModelPrimitives):
    """Appendix C: u(xi)=-1/xi, D(h)=h^(-0.5), L(h)=0.

    m0=inf, I(q)=q^(-0.5).
    """

    @property
    def m0(self):
        return np.inf

    def u(self, xi):
        xi = np.asarray(xi, dtype=float)
        return -1.0 / xi

    def u_prime(self, xi):
        xi = np.asarray(xi, dtype=float)
        return 1.0 / xi ** 2

    def u_double_prime(self, xi):
        xi = np.asarray(xi, dtype=float)
        return -2.0 / xi ** 3

    def D(self, h):
        return np.asarray(h, dtype=float) ** (-0.5)

    def D_prime(self, h):
        return -0.5 * np.asarray(h, dtype=float) ** (-1.5)

    def D_double_prime(self, h):
        return 0.75 * np.asarray(h, dtype=float) ** (-2.5)

    def L(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def L_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def L_double_prime(self, h):
        return np.zeros_like(np.asarray(h, dtype=float))

    def I(self, q):
        q_arr = np.asarray(q, dtype=float)
        scalar_input = np.isscalar(q)
        result = q_arr ** (-0.5)
        return float(result) if scalar_input else result
