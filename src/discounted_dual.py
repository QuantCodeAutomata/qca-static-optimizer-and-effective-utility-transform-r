"""Discounted dual value function for rho > 0."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar, brentq
from src.primitives import ModelPrimitives
from src.static_block import StaticBlock
from src.inverse_maps import MonotoneInverse

class DiscountedDual:
    """Discounted dual J_rho(y,h) from Duality of Shortfall Aversion, Section 3."""
    
    def __init__(self, prim: ModelPrimitives, market_params: dict, rho: float):
        self.prim = prim
        self.r = market_params['r']
        self.mu = market_params['mu']
        self.sigma = market_params['sigma']
        self.rho = rho
        self.kappa = (self.mu - self.r) / self.sigma
        self.sb = StaticBlock(prim)
        self.gamma1, self.gamma2 = self.compute_roots()
        self._trunc_upper = np.inf  # scipy.integrate.quad handles ∞ via variable substitution
    
    def compute_roots(self):
        """Solve (kappa^2/2)*g*(g-1) + (rho-r)*g - rho = 0. Return (gamma1>1, gamma2<0)."""
        k2 = self.kappa**2
        # Coefficients: (k2/2)*g^2 + (rho-r - k2/2)*g - rho = 0
        a_coef = k2 / 2
        b_coef = (self.rho - self.r) - k2/2
        c_coef = -self.rho
        disc = b_coef**2 - 4*a_coef*c_coef
        g1 = (-b_coef + np.sqrt(disc)) / (2*a_coef)
        g2 = (-b_coef - np.sqrt(disc)) / (2*a_coef)
        if g1 > g2:
            return g1, g2
        else:
            return g2, g1
    
    def _prefactor(self):
        return 2.0 / (self.kappa**2 * (self.gamma1 - self.gamma2))
    
    def _split_quad(
        self,
        integrand,
        lo: float,
        hi: float,
        kink: float,
        tol: float = 1e-9,
    ) -> float:
        """quad(integrand, lo, hi) with a QUADPACK breakpoint at `kink`.

        Passing `points=[kink]` to scipy.integrate.quad makes the adaptive
        algorithm aware of the C⁰ kink at the branch boundary of ū / ū_h,
        preventing convergence failures without requiring manual sub-interval splits.
        """
        pts = [kink] if lo < kink < hi else []
        val, _ = quad(integrand, lo, hi, points=pts,
                      limit=500, epsabs=tol, epsrel=tol)
        return val

    def R_rho(self, y: float, a: float) -> float:
        """R_ρ(y,a) = pf·[y^γ₂ ∫₀^y s^{-γ₂-1}ū ds + y^γ₁ ∫_y^∞ s^{-γ₁-1}ū ds].

        Integrals split at branch kink b(a) for numerical stability.
        """
        g1, g2 = self.gamma1, self.gamma2
        pf = self._prefactor()
        eps = 1e-10
        b_a = self.sb.b_thresh(a)
        trunc = max(b_a * 200.0, 50.0)

        def f1(s: float) -> float:
            return s**(-g2 - 1) * self.sb.ubar(s, a)

        def f2(s: float) -> float:
            return s**(-g1 - 1) * self.sb.ubar(s, a)

        I1 = self._split_quad(f1, eps, y,     b_a)
        I2 = self._split_quad(f2, y,   trunc, b_a)
        return pf * (y**g2 * I1 + y**g1 * I2)

    def G_rho(self, y: float, a: float) -> float:
        """G_ρ(y,a) = same as R_ρ but with ū_h replacing ū."""
        g1, g2 = self.gamma1, self.gamma2
        pf = self._prefactor()
        eps = 1e-10
        b_a = self.sb.b_thresh(a)
        trunc = max(b_a * 200.0, 50.0)

        def f1(s: float) -> float:
            return s**(-g2 - 1) * self.sb.ubar_h(s, a)

        def f2(s: float) -> float:
            return s**(-g1 - 1) * self.sb.ubar_h(s, a)

        I1 = self._split_quad(f1, eps, y,     b_a)
        I2 = self._split_quad(f2, y,   trunc, b_a)
        return pf * (y**g2 * I1 + y**g1 * I2)
    
    def verify_dR_da_equals_G(self, y_test: float, a_test: float, eps: float = 1e-5) -> float:
        """Check partial_a R_rho = G_rho numerically."""
        R_up = self.R_rho(y_test, a_test + eps)
        R_dn = self.R_rho(y_test, a_test - eps)
        fd = (R_up - R_dn) / (2*eps)
        G = self.G_rho(y_test, a_test)
        return abs(fd - G)
    
    def _free_boundary_integrand(self, z: float, a: float) -> float:
        """∫_z^∞ s^{-γ₁-1} ū_h(s,a) ds.

        For s < b(a) the branch formula gives ū_h(s,a) = Γ(a) − s (exact).
        That lower piece is therefore analytically integrable:

            ∫_z^{min(z_up,b_a)} s^{-γ₁-1}(Γ(a) − s) ds
              = Γ(a)/γ₁ · (z^{-γ₁} − z_up^{-γ₁})
                + 1/(γ₁−1) · (z_up^{-γ₁+1} − z^{-γ₁+1})

        This avoids QUADPACK seeing the steep power-law behaviour near s=0.
        """
        g1 = self.gamma1          # > 1
        b_a = self.sb.b_thresh(a)
        Gamma_a = self.sb.Gamma_fn(a)
        trunc = max(b_a * 200.0, 50.0)

        if z >= b_a:
            # Entire integration is in the smooth upper branch; use QUADPACK.
            def integrand_up(s: float) -> float:
                return s**(-g1 - 1) * self.sb.ubar_h(s, a)
            val, _ = quad(integrand_up, z, trunc,
                          limit=500, epsabs=1e-9, epsrel=1e-9)
            return val

        # Lower piece [z, b_a]: ū_h(s,a) = Γ(a) − s  (branch exact)
        # ∫_z^{b_a} s^{-g1-1}(Γ_a − s) ds
        #   = Γ_a/g1 · (z^{-g1} − b_a^{-g1})
        #     + 1/(g1-1) · (b_a^{-g1+1} − z^{-g1+1})
        lower = (Gamma_a / g1) * (z**(-g1) - b_a**(-g1)) + \
                (b_a**(-g1 + 1) - z**(-g1 + 1)) / (g1 - 1)

        # Upper piece [b_a, trunc]: smooth, handled by QUADPACK
        def integrand_up(s: float) -> float:
            return s**(-g1 - 1) * self.sb.ubar_h(s, a)
        upper, _ = quad(integrand_up, b_a, trunc,
                        limit=500, epsabs=1e-9, epsrel=1e-9)

        return lower + upper
    
    def compute_free_boundary(self, a_grid: np.ndarray) -> np.ndarray:
        """For each a, solve int_z^inf s^(-gamma1-1)*ubar_h(s,a)ds = 0 for z_rho(a)."""
        z_vals = np.zeros(len(a_grid))
        for i, a in enumerate(a_grid):
            Gamma_a = self.sb.Gamma_fn(a)
            eps = 1e-8
            # Find bracket: check sign at eps and near Gamma_a
            try:
                f_low = self._free_boundary_integrand(eps, a)
                f_high = self._free_boundary_integrand(Gamma_a * 0.999, a)
                if f_low * f_high >= 0:
                    # Try to find sign change with finer search
                    test_pts = np.logspace(np.log10(eps), np.log10(Gamma_a * 0.999), 20)
                    f_vals = [self._free_boundary_integrand(t, a) for t in test_pts]
                    sign_changes = [(test_pts[j], test_pts[j+1]) for j in range(len(f_vals)-1)
                                   if f_vals[j] * f_vals[j+1] < 0]
                    if sign_changes:
                        lo, hi = sign_changes[0]
                    else:
                        z_vals[i] = Gamma_a * 0.5  # fallback
                        continue
                else:
                    lo, hi = eps, Gamma_a * 0.999
                result = brentq(lambda z: self._free_boundary_integrand(z, a), lo, hi, xtol=1e-10)
                z_vals[i] = result
            except Exception as e:
                z_vals[i] = self.sb.Gamma_fn(a) * 0.5  # fallback
        return z_vals
    
    def A_rho(self, a: float, z_rho_a: float) -> float:
        """A_ρ(a) = pf · ∫₀^{z_ρ(a)} s^{-γ₂-1} ū_h(s,a) ds.

        Split at branch kink b(a) when it falls inside the integration interval.
        """
        pf = self._prefactor()
        eps = 1e-10

        def integrand(s: float) -> float:
            return s**(-self.gamma2 - 1) * self.sb.ubar_h(s, a)

        val = self._split_quad(integrand, eps, z_rho_a, kink=self.sb.b_thresh(a))
        return pf * val
    
    def S_rho(self, y: float, a: float, z_rho_a: float, A_rho_a: float) -> float:
        """Stopping value: G_rho if y<=z_rho, else A_rho*y^gamma2."""
        if y <= z_rho_a:
            return self.G_rho(y, a)
        else:
            return A_rho_a * y**self.gamma2
    
    def J_rho(self, y: float, h: float, a_grid: np.ndarray, z_rho_vals: np.ndarray, A_rho_vals: np.ndarray) -> float:
        """J_rho(y,h) = R_rho(y,h) + int_h^inf S_rho(y,a) da."""
        R = self.R_rho(y, h)
        a_max = a_grid[-1]
        
        # Build interpolated z_rho and A_rho functions
        z_interp = np.interp
        A_interp = np.interp
        
        def integrand(a):
            z = np.interp(a, a_grid, z_rho_vals)
            A = np.interp(a, a_grid, A_rho_vals)
            return self.S_rho(y, a, float(z), float(A))
        
        if h >= a_max:
            return R
        
        val, _ = quad(integrand, h, a_max, limit=100, epsabs=1e-8, epsrel=1e-8)
        return R + val
    
    def J_rho_y(self, y: float, h: float, a_grid, z_rho_vals, A_rho_vals, eps: float = 1e-6) -> float:
        """Numerical dJ/dy."""
        Jp = self.J_rho(y + eps, h, a_grid, z_rho_vals, A_rho_vals)
        Jm = self.J_rho(y - eps, h, a_grid, z_rho_vals, A_rho_vals)
        return (Jp - Jm) / (2*eps)
    
    def J_rho_yy(self, y: float, h: float, a_grid, z_rho_vals, A_rho_vals, eps: float = 1e-5) -> float:
        """Numerical d^2J/dy^2."""
        Jp = self.J_rho(y + eps, h, a_grid, z_rho_vals, A_rho_vals)
        J0 = self.J_rho(y, h, a_grid, z_rho_vals, A_rho_vals)
        Jm = self.J_rho(y - eps, h, a_grid, z_rho_vals, A_rho_vals)
        return (Jp - 2*J0 + Jm) / eps**2
    
    def find_multiplier(self, x: float, h: float, a_grid, z_rho_vals, A_rho_vals) -> float:
        """Solve -J_rho_y(y*,h) = x for y* in the continuation region (y* > z_rho(h)).

        By convexity of J_rho the solution is unique; searching only in the
        continuation region avoids false roots from numerical noise in S_rho near y=0.
        """
        admissible_bound = self.sb.ell(h) / self.r
        if x <= admissible_bound:
            raise ValueError(f"x={x} not admissible: need x > ell(h)/r = {admissible_bound}")

        z_rho_h = float(np.interp(h, a_grid, z_rho_vals))

        def objective(y: float) -> float:
            return -self.J_rho_y(y, h, a_grid, z_rho_vals, A_rho_vals) - x

        # Bracket entirely in the continuation region
        y_lo = max(z_rho_h * 1.001, 1e-8)
        y_hi = max(y_lo * 2.0, 1.0)

        for _ in range(60):
            try:
                if objective(y_lo) > 0 > objective(y_hi):
                    break
            except Exception:
                pass
            y_hi *= 2.0

        return brentq(objective, y_lo, y_hi, xtol=1e-10, maxiter=200)
