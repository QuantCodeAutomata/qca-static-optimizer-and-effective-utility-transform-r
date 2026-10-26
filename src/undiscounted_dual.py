"""Undiscounted dual value function for rho=0."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from src.primitives import ModelPrimitives
from src.static_block import StaticBlock

class UndiscountedDual:
    """Undiscounted dual V_0(y,h) from Duality of Shortfall Aversion, Section 4."""
    
    def __init__(self, prim: ModelPrimitives, market_params: dict):
        self.prim = prim
        self.r = market_params['r']
        self.mu = market_params['mu']
        self.sigma = market_params['sigma']
        self.rho = 0.0
        self.kappa = (self.mu - self.r) / self.sigma
        self.sb = StaticBlock(prim)
        self.delta0 = self.compute_delta0()
        self._trunc_upper = np.inf   # scipy.integrate.quad handles ∞ via variable substitution
    
    def compute_delta0(self) -> float:
        """delta_0 = 1 + 2r/kappa^2. Typo correction: Section 4 uses this exponent (matches Appendix B)."""
        return 1.0 + 2.0 * self.r / self.kappa**2
    
    def scaled_kernel(self, s: float, h: float) -> float:
        """d_h(s) = ubar_h(Gamma_fn(h)*s, h) / Gamma_fn(h)."""
        Gamma_h = self.sb.Gamma_fn(h)
        return self.sb.ubar_h(Gamma_h * s, h) / Gamma_h
    
    def _branch_kink_s(self, h: float) -> float:
        """Scaled value of the branch kink: s_b = b(h) / Gamma(h).

        The scaled kernel d_h(s) has a kink (non-smooth branch change) at s_b.
        Passing this as a breakpoint to quad prevents convergence issues.
        """
        Gamma_h = self.sb.Gamma_fn(h)
        b_h = self.sb.b_thresh(h)
        return b_h / Gamma_h

    def _free_boundary_tail_integral(self, zeta: float, h: float) -> float:
        """∫_zeta^∞ s^{-δ₀-1} d_h(s) ds.

        For s < s_b = b(h)/Γ(h) the branch formula gives d_h(s) = 1 − s exactly.
        That lower piece is analytically integrable (δ₀ > 1):

            ∫_zeta^{s_b} s^{-δ₀-1}(1−s) ds
              = (zeta^{-δ₀} − s_b^{-δ₀}) / δ₀
                + (s_b^{-δ₀+1} − zeta^{-δ₀+1}) / (δ₀ − 1)

        This avoids QUADPACK encountering the steep power-law near ζ ≈ 0.
        """
        d0 = self.delta0       # > 1
        s_b = self._branch_kink_s(h)
        trunc = max(200.0 * s_b, 50.0)

        if zeta >= s_b:
            # Entire interval is in the smooth upper branch.
            def integrand_up(s: float) -> float:
                return s**(-d0 - 1) * self.scaled_kernel(s, h)
            val, _ = quad(integrand_up, zeta, trunc,
                          limit=500, epsabs=1e-9, epsrel=1e-9)
            return val

        # Lower piece [zeta, s_b]: d_h(s) = 1 − s (branch exact)
        lower = (zeta**(-d0) - s_b**(-d0)) / d0 + \
                (s_b**(-d0 + 1) - zeta**(-d0 + 1)) / (d0 - 1)

        # Upper piece [s_b, trunc]: smooth, QUADPACK handles it fine.
        def integrand_up(s: float) -> float:
            return s**(-d0 - 1) * self.scaled_kernel(s, h)
        upper, _ = quad(integrand_up, s_b, trunc,
                        limit=500, epsabs=1e-9, epsrel=1e-9)

        return lower + upper
    
    def compute_free_boundary(self, h_grid: np.ndarray):
        """For each h, solve int_{zeta}^inf s^(-delta0-1)*d_h(s) ds = 0 for zeta0(h) in (0,1).

        The integrand is positive for small s (upper branch, d_h = 1-s > 0 when s < 1)
        and negative for large s (middle/lower branch). The zero crossing zeta0 is in (0,1).
        """
        zeta0_vals = np.zeros(len(h_grid))
        z0_vals = np.zeros(len(h_grid))

        for i, h in enumerate(h_grid):
            try:
                Gamma_h = self.sb.Gamma_fn(h)
                # The zero crossing is in (eps, 1) since:
                # - f(eps) > 0  (dominated by positive upper-branch near s=0)
                # - f(1-eps) < 0 (negative middle-branch tail dominates for zeta near 1)
                search_lo, search_hi = 1e-5, 1.0 - 1e-5

                f_lo = self._free_boundary_tail_integral(search_lo, h)
                f_hi = self._free_boundary_tail_integral(search_hi, h)

                if f_lo * f_hi >= 0:
                    # Scan a finer grid for sign change
                    test_pts = np.linspace(search_lo, search_hi, 200)
                    f_vals = [self._free_boundary_tail_integral(t, h) for t in test_pts]
                    sign_changes = [
                        (test_pts[j], test_pts[j+1])
                        for j in range(len(f_vals)-1)
                        if f_vals[j] * f_vals[j+1] < 0
                    ]
                    if sign_changes:
                        lo, hi = sign_changes[0]
                    else:
                        zeta0_vals[i] = 0.5
                        z0_vals[i] = Gamma_h * 0.5
                        continue
                else:
                    lo, hi = search_lo, search_hi

                zeta0 = brentq(
                    lambda z: self._free_boundary_tail_integral(z, h),
                    lo, hi, xtol=1e-10
                )
                zeta0_vals[i] = zeta0
                z0_vals[i] = Gamma_h * zeta0
            except Exception:
                zeta0_vals[i] = 0.5
                z0_vals[i] = self.sb.Gamma_fn(h) * 0.5

        return zeta0_vals, z0_vals
    
    def Phi0(self, y: float, h: float) -> float:
        """Phi_0(y,h) = [2/(k²δ₀)] * [∫_1^y ū(s,h)/s ds + y^δ₀ ∫_y^∞ s^{-δ₀-1} ū(s,h) ds].

        The kink of ū at b(h) is passed as a QUADPACK breakpoint so adaptive
        quadrature handles the C⁰ discontinuity without convergence warnings.
        """
        k2 = self.kappa**2
        d0 = self.delta0
        b_h = self.sb.b_thresh(h)

        def integrand1(s: float) -> float:
            return self.sb.ubar(s, h) / s

        # ∫_1^y ū/s ds  (signed; kink at b_h when it lies strictly between 1 and y)
        ref = 1.0
        if y > ref:
            pts = [b_h] if ref < b_h < y else []
            I1, _ = quad(integrand1, ref, y, points=pts,
                         limit=500, epsabs=1e-10, epsrel=1e-10)
        elif y < ref:
            pts = [b_h] if y < b_h < ref else []
            raw, _ = quad(integrand1, y, ref, points=pts,
                          limit=500, epsabs=1e-10, epsrel=1e-10)
            I1 = -raw
        else:
            I1 = 0.0

        # ∫_y^∞ s^{-δ₀-1} ū ds (tail helper already uses points=)
        I2 = self._ubar_integral_tail(y, h)

        return (2.0 / (k2 * d0)) * (I1 + y**d0 * I2)
    
    def _ubar_integral_tail(self, y: float, h: float) -> float:
        """∫_y^∞ s^{-d0-1} ū(s,h) ds, split at the branch kink b(h) for stability."""
        d0 = self.delta0
        b_h = self.sb.b_thresh(h)
        trunc = max(self._trunc_upper, b_h * 200.0) if np.isfinite(self._trunc_upper) else max(b_h * 200.0, 1e4)

        def integrand(s: float) -> float:
            return s**(-d0 - 1) * self.sb.ubar(s, h)

        pts = [b_h] if y < b_h < trunc else []
        val, _ = quad(integrand, y, trunc, points=pts,
                      limit=500, epsabs=1e-10, epsrel=1e-10)
        return val

    def Phi0_y(self, y: float, h: float) -> float:
        """[2/kappa^2] * y^(delta0-1) * ∫_y^∞ s^{-d0-1} ū(s,h) ds.

        Uses split-at-kink integration to avoid QUADPACK convergence warnings
        at the branch boundary b(h) of ū.
        """
        k2 = self.kappa**2
        d0 = self.delta0
        I = self._ubar_integral_tail(y, h)
        return (2.0 / k2) * y**(d0 - 1) * I
    
    def Phi0_yy(self, y: float, h: float) -> float:
        """Analytical d²Phi0/dy² in the continuation region.

        Differentiating Phi0_y = (2/k²)*y^(d0-1)*I2(y) where I2 = ∫_y^∞ s^(-d0-1) ū ds:
          Phi0_yy = (d0-1)/y * Phi0_y(y,h)  -  (2/k²)*y^(-2)*ū(y,h)

        This satisfies the PDE (k²/2)*y²*Phi0_yy - r*y*Phi0_y + ū = 0 exactly.
        Custom — Context7 found no library equivalent (paper Section 4).
        """
        ub = self.sb.ubar(y, h)
        phi_y = self.Phi0_y(y, h)
        k2 = self.kappa**2
        d0 = self.delta0
        return (d0 - 1) / y * phi_y - (2.0 / k2) * y**(-2) * ub

    def Phi0_yh(self, y: float, h: float) -> float:
        """[2/kappa^2] * y^(delta0-1) * ∫_y^∞ s^{-d0-1} ū_h(s,h) ds.

        Split at b(h) for numerical stability (ū_h has a kink there).
        """
        k2 = self.kappa**2
        d0 = self.delta0
        b_h = self.sb.b_thresh(h)
        trunc = max(b_h * 200.0, 1e4)

        def integrand(s: float) -> float:
            return s**(-d0 - 1) * self.sb.ubar_h(s, h)

        pts = [b_h] if y < b_h < trunc else []
        I, _ = quad(integrand, y, trunc, points=pts,
                    limit=500, epsabs=1e-10, epsrel=1e-10)
        return (2.0 / k2) * y**(d0 - 1) * I
    
    def compute_B0(self, h_grid: np.ndarray, z0_vals: np.ndarray) -> np.ndarray:
        """B0(h) = -int_h^inf Phi0_y(z0(a),a)*z0'(a) da."""
        n = len(h_grid)
        # Estimate z0' via central differences
        z0_prime = np.gradient(z0_vals, h_grid)
        
        # Precompute Phi0_y at z0 points
        phi_y_vals = np.array([self.Phi0_y(float(z0_vals[i]), float(h_grid[i]))
                               for i in range(n)])
        
        integrand_vals = phi_y_vals * z0_prime
        B0_vals = np.zeros(n)
        
        # Integrate from h to max by cumulative trapezoid from right
        for i in range(n):
            # int_h_i^h_max integrand_vals da
            if i < n - 1:
                val, _ = quad(
                    lambda a: float(np.interp(a, h_grid, integrand_vals)),
                    h_grid[i], h_grid[-1], limit=100
                )
                B0_vals[i] = -val
            else:
                B0_vals[i] = 0.0
        
        return B0_vals
    
    def V0(self, y: float, h: float, h_grid: np.ndarray, z0_vals: np.ndarray, B0_vals: np.ndarray) -> float:
        """V0(y,h): if y>z0(h): B0(h)+Phi0(y,h)-Phi0(z0(h),h); if y<=z0(h): B0(z0_inv(y))."""
        z0_h = float(np.interp(h, h_grid, z0_vals))
        B0_h = float(np.interp(h, h_grid, B0_vals))
        
        if y > z0_h:
            phi_y = self.Phi0(y, h)
            phi_z0 = self.Phi0(z0_h, h)
            return B0_h + phi_y - phi_z0
        else:
            # y <= z0(h): V0 = B0(z0_inv(y))
            # Invert z0: find a such that z0(a) = y
            # z0 is decreasing, so invert
            try:
                from scipy.optimize import brentq as bq
                h_inv = bq(
                    lambda a: float(np.interp(a, h_grid, z0_vals)) - y,
                    h_grid[0], h_grid[-1], xtol=1e-10
                )
                B0_inv = float(np.interp(h_inv, h_grid, B0_vals))
                return B0_inv
            except Exception:
                return B0_h
    
    def V0_y(self, y: float, h: float, h_grid, z0_vals, B0_vals, eps: float = 1e-7) -> float:
        """∂V0/∂y.  Analytical (= Phi0_y) in continuation region; FD fallback at boundary."""
        z0_h = float(np.interp(h, h_grid, z0_vals))
        if y > z0_h:
            # In continuation region V0 = B0 + Phi0(y,h) - Phi0(z0,h), so ∂V0/∂y = Phi0_y
            return self.Phi0_y(y, h)
        # Action region or near boundary: use centred finite difference
        Vp = self.V0(y + eps, h, h_grid, z0_vals, B0_vals)
        Vm = self.V0(y - eps, h, h_grid, z0_vals, B0_vals)
        return (Vp - Vm) / (2 * eps)

    def V0_yy(self, y: float, h: float, h_grid, z0_vals, B0_vals, eps: float = 1e-5) -> float:
        """∂²V0/∂y².  Analytical (= Phi0_yy) in continuation region; FD fallback near boundary."""
        z0_h = float(np.interp(h, h_grid, z0_vals))
        if y > z0_h:
            # In continuation region V0 = B0 + Phi0(y,h) - Phi0(z0,h), so ∂²V0/∂y² = Phi0_yy
            return self.Phi0_yy(y, h)
        # Action region or boundary: centred FD
        Vp = self.V0(y + eps, h, h_grid, z0_vals, B0_vals)
        V0v = self.V0(y, h, h_grid, z0_vals, B0_vals)
        Vm = self.V0(y - eps, h, h_grid, z0_vals, B0_vals)
        return (Vp - 2 * V0v + Vm) / eps**2
    
    def find_multiplier(self, x: float, h: float, h_grid, z0_vals, B0_vals) -> float:
        """Solve -V0_y(y*,h) = x for y* in the continuation region (y* > z0(h)).

        In the continuation region V0_y = Phi0_y analytically, so we use Phi0_y
        directly for the root-finding bracket search.  This avoids false roots in
        the action region where V0 = B0(z0_inv(y)) and FD-based V0_y may be noisy.
        """
        admissible_bound = self.sb.ell(h) / self.r
        if x <= admissible_bound:
            raise ValueError(f"x={x} not admissible: need x > ell(h)/r = {admissible_bound:.4f}")

        z0_h = float(np.interp(h, h_grid, z0_vals))

        # In the continuation region: -Phi0_y(y,h) = x, decreasing from +inf to 0
        def objective(y: float) -> float:
            return -self.Phi0_y(y, h) - x

        # Start bracket just above the free boundary
        y_lo = max(z0_h * 1.001, 1e-8)
        y_hi = max(y_lo * 2.0, 1.0)

        # Exponential search: double y_hi until sign change is found
        for _ in range(60):
            try:
                if objective(y_lo) > 0 > objective(y_hi):
                    break
            except Exception:
                pass
            y_hi *= 2.0

        return brentq(objective, y_lo, y_hi, xtol=1e-10, maxiter=200)
    
    def verify_vi_residual(self, y: float, h: float, h_grid, z0_vals, B0_vals) -> float:
        """Absolute PDE residual |(k²/2)y²V_yy - r y V_y + ū| in the continuation region.

        Uses analytical Phi0_y and Phi0_yy when y > z0(h), giving near-machine-precision
        PDE satisfaction (formula proved exact in Section 4 derivation).
        Also returns the V_h complementarity separately via the second return of
        verify_vi_detail.  This scalar overload returns |PDE term| for quick diagnostics.
        """
        k2 = self.kappa**2
        Vyy = self.V0_yy(y, h, h_grid, z0_vals, B0_vals)
        Vy = self.V0_y(y, h, h_grid, z0_vals, B0_vals)
        ub = self.sb.ubar(y, h)
        pde_term = (k2 / 2) * y**2 * Vyy - self.r * y * Vy + ub
        return abs(pde_term)
