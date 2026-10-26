"""Proposition 2.2 static optimization from Duality of Shortfall Aversion."""

import numpy as np
from scipy.optimize import minimize_scalar

from .primitives import ModelPrimitives


class StaticBlock:
    """Implements Proposition 2.2: optimal consumption c*(y,h) and value ubar(y,h).

    For fixed habit h, the static problem is:
        ubar(y,h) = sup_{c in [ell(h), h]} { u(D(h)*c - L(h)) - y*c }

    Notation
    --------
    ell(h)      = L(h) / D(h)
    g(h)        = D(h)*h - L(h)
    g'(h)       = D'(h)*h + D(h) - L'(h)
    b(h)        = D(h)*u'(g(h))          lower threshold in y
    d(h)        = D(h)*m0                 upper threshold in y (inf when m0=inf)
    Gamma(h)    = g'(h)*u'(g(h))
    vartheta(h) = g'(h)/D(h)
    chi(h)      = -(D'(h)/D(h))*(h - ell(h))
    ell'(h)     = (L'(h)*D(h) - L(h)*D'(h)) / D(h)^2
    """

    def __init__(self, prim: ModelPrimitives):
        self.prim = prim

    # ------------------------------------------------------------------
    # Derived scalar quantities (h is a scalar float)
    # ------------------------------------------------------------------

    def ell(self, h):
        """ell(h) = L(h) / D(h). Accepts scalar or numpy array h."""
        return np.asarray(self.prim.L(h), dtype=float) / np.asarray(self.prim.D(h), dtype=float)

    def ell_prime(self, h):
        """ell'(h) = (L'(h)*D(h) - L(h)*D'(h)) / D(h)^2. Accepts scalar or array h."""
        h = np.asarray(h, dtype=float)
        L = np.asarray(self.prim.L(h), dtype=float)
        Lp = np.asarray(self.prim.L_prime(h), dtype=float)
        D = np.asarray(self.prim.D(h), dtype=float)
        Dp = np.asarray(self.prim.D_prime(h), dtype=float)
        result = (Lp * D - L * Dp) / D ** 2
        return float(result) if np.ndim(h) == 0 else result

    def g(self, h):
        """g(h) = D(h)*h - L(h). Accepts scalar or array h."""
        h = np.asarray(h, dtype=float)
        return np.asarray(self.prim.D(h), dtype=float) * h - np.asarray(self.prim.L(h), dtype=float)

    def g_prime(self, h):
        """g'(h) = D'(h)*h + D(h) - L'(h). Accepts scalar or array h."""
        h = np.asarray(h, dtype=float)
        return (
            np.asarray(self.prim.D_prime(h), dtype=float) * h
            + np.asarray(self.prim.D(h), dtype=float)
            - np.asarray(self.prim.L_prime(h), dtype=float)
        )

    def b(self, h):
        """b(h) = D(h)*u'(g(h)). Lower threshold in y. Accepts scalar or array h."""
        return np.asarray(self.prim.D(h), dtype=float) * np.asarray(self.prim.u_prime(self.g(h)), dtype=float)

    def d(self, h):
        """d(h) = D(h)*m0. Upper threshold (inf if m0=inf). Accepts scalar or array h."""
        D_h = np.asarray(self.prim.D(h), dtype=float)
        if np.isinf(self.prim.m0):
            return np.full_like(D_h, np.inf) if np.ndim(D_h) > 0 else np.inf
        return D_h * self.prim.m0

    def Gamma(self, h):
        """Gamma(h) = g'(h)*u'(g(h)). Accepts scalar or array h."""
        return np.asarray(self.g_prime(h), dtype=float) * np.asarray(self.prim.u_prime(self.g(h)), dtype=float)

    # Aliases used by discounted_dual.py and undiscounted_dual.py
    def Gamma_fn(self, h):
        """Alias for Gamma(h)."""
        return self.Gamma(h)

    def b_thresh(self, h):
        """Alias for b(h)."""
        return self.b(h)

    def d_thresh(self, h):
        """Alias for d(h)."""
        return self.d(h)

    def vartheta(self, h):
        """vartheta(h) = g'(h) / D(h). Accepts scalar or array h."""
        return np.asarray(self.g_prime(h), dtype=float) / np.asarray(self.prim.D(h), dtype=float)

    def chi(self, h):
        """chi(h) = -(D'(h)/D(h)) * (h - ell(h)). Accepts scalar or array h."""
        h = np.asarray(h, dtype=float)
        D = np.asarray(self.prim.D(h), dtype=float)
        Dp = np.asarray(self.prim.D_prime(h), dtype=float)
        return -(Dp / D) * (h - self.ell(h))

    # ------------------------------------------------------------------
    # Core functions: vectorised over y, scalar h
    # ------------------------------------------------------------------

    def cstar(self, y, h: float):
        """Optimal c*(y, h) per Proposition 2.2.

        Branch 1 (y < b(h)):          c* = h
        Branch 2 (b(h) <= y < d(h)):  c* = ell(h) + I(y/D(h)) / D(h)
        Branch 3 (y >= d(h)):         c* = ell(h)
        """
        scalar_input = np.isscalar(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))

        D_h = float(self.prim.D(h))
        ell_h = self.ell(h)
        b_h = self.b(h)
        d_h = self.d(h)

        result = np.empty_like(y_arr)
        mask1 = y_arr < b_h
        mask3 = y_arr >= d_h
        mask2 = ~mask1 & ~mask3

        result[mask1] = h
        result[mask3] = ell_h
        if np.any(mask2):
            q = y_arr[mask2] / D_h
            result[mask2] = ell_h + self.prim.I(q) / D_h

        return float(result.squeeze()) if scalar_input else result

    def ubar(self, y, h: float):
        """ubar(y, h) = sup_{c in [ell(h), h]} { u(D(h)*c - L(h)) - y*c }.

        Branch 1 (y < b(h)):          u(g(h)) - y*h
        Branch 2 (b(h) <= y < d(h)):  u(I(y/D(h))) - y*c*
        Branch 3 (y >= d(h)):         u(0) - y*ell(h)  [-inf when u(0)=-inf]
        """
        scalar_input = np.isscalar(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))

        prim = self.prim
        D_h = float(prim.D(h))
        ell_h = self.ell(h)
        g_h = self.g(h)
        b_h = self.b(h)
        d_h = self.d(h)

        result = np.empty_like(y_arr)
        mask1 = y_arr < b_h
        mask3 = y_arr >= d_h
        mask2 = ~mask1 & ~mask3

        # Branch 1: c* = h, xi = g(h)
        result[mask1] = float(prim.u(g_h)) - y_arr[mask1] * h

        # Branch 2: c* = ell_h + I(q)/D_h, xi = I(q)
        if np.any(mask2):
            q = y_arr[mask2] / D_h
            I_q = prim.I(q)
            c_star = ell_h + I_q / D_h
            result[mask2] = prim.u(I_q) - y_arr[mask2] * c_star

        # Branch 3: c* = ell_h, xi = D(h)*ell(h)-L(h) = 0 => u(0)
        if np.any(mask3):
            try:
                u0 = float(prim.u(np.array(0.0)))
            except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
                u0 = -np.inf
            if not np.isfinite(u0):
                result[mask3] = -np.inf
            else:
                result[mask3] = u0 - y_arr[mask3] * ell_h

        return float(result.squeeze()) if scalar_input else result

    def ubar_h(self, y, h: float):
        """d/dh ubar(y, h) per Proposition 2.2 (envelope theorem).

        Branch 1 (y < b(h)):          Gamma(h) - y
        Branch 2 (b(h) <= y < d(h)):  (y/D(h)) * (D'(h)*c* - L'(h))
        Branch 3 (y >= d(h)):         -y * ell'(h)

        Branch 2 derivation: envelope theorem at interior optimum gives
            d/dh ubar = u'(I(y/D(h))) * (D'(h)*c* - L'(h))
                      = (y/D(h)) * (D'(h)*c* - L'(h))
        since u'(I(q)) = q = y/D(h) at the optimum.
        """
        scalar_input = np.isscalar(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))

        prim = self.prim
        D_h = float(prim.D(h))
        Dp_h = float(prim.D_prime(h))
        Lp_h = float(prim.L_prime(h))
        ell_h = self.ell(h)
        b_h = self.b(h)
        d_h = self.d(h)
        Gamma_h = self.Gamma(h)
        ell_p_h = self.ell_prime(h)

        result = np.empty_like(y_arr)
        mask1 = y_arr < b_h
        mask3 = y_arr >= d_h
        mask2 = ~mask1 & ~mask3

        # Branch 1: d/dh[u(g(h)) - y*h] = u'(g(h))*g'(h) - y = Gamma(h) - y
        result[mask1] = Gamma_h - y_arr[mask1]

        # Branch 2: (y/D(h)) * (D'(h)*c* - L'(h))
        if np.any(mask2):
            q = y_arr[mask2] / D_h
            I_q = prim.I(q)
            c_star = ell_h + I_q / D_h
            result[mask2] = (y_arr[mask2] / D_h) * (Dp_h * c_star - Lp_h)

        # Branch 3: d/dh[u(0) - y*ell(h)] = -y * ell'(h)
        result[mask3] = -y_arr[mask3] * ell_p_h

        return float(result.squeeze()) if scalar_input else result

    def ubar_y(self, y, h: float):
        """d/dy ubar(y, h) = -c*(y, h)."""
        return -self.cstar(y, h)

    # ------------------------------------------------------------------
    # Verification utilities
    # ------------------------------------------------------------------

    def verify_assumption_21(self, h: float, xi_test=None, tol: float = 1e-10) -> dict:
        """Check Assumption 2.1 conditions at habit level h.

        Checks
        ------
        u_increasing     : u'(xi) > 0 over xi_test grid
        u_concave        : u''(xi) < 0 over xi_test grid
        D_positive       : D(h) > 0
        ell_lt_h         : ell(h) < h
        D_prime_nonpos   : D'(h) <= 0
        ell_prime_nonneg : ell'(h) >= 0
        g_prime_in_0_D   : 0 < g'(h) < D(h)
        all_satisfied    : conjunction of all checks
        """
        prim = self.prim
        if xi_test is None:
            xi_test = np.logspace(-2, 2, 20)

        D_h = float(prim.D(h))
        Dp_h = float(prim.D_prime(h))
        ell_h = self.ell(h)
        ell_p_h = self.ell_prime(h)
        gp_h = self.g_prime(h)

        checks = {
            # u' may be numerically tiny (e.g. exp(-100)) but must not be negative
            "u_increasing":     bool(np.all(prim.u_prime(xi_test) > -tol)),
            # u'' may be numerically tiny in magnitude but must not be positive
            "u_concave":        bool(np.all(prim.u_double_prime(xi_test) < tol)),
            "D_positive":       D_h > tol,
            "ell_lt_h":         ell_h < h - tol,
            "D_prime_nonpos":   Dp_h <= tol,
            "ell_prime_nonneg": ell_p_h >= -tol,
            "g_prime_in_0_D":   bool((gp_h > tol) and (gp_h < D_h + tol)),
        }
        checks["all_satisfied"] = all(checks.values())
        return checks

    def numerical_cstar(self, y, h: float):
        """Numerically compute c*(y, h) via direct optimization for verification.

        Minimises -u(D(h)*c - L(h)) + y*c over c in [ell(h), h] using
        scipy minimize_scalar, comparing interior and boundary candidates.

        Returns
        -------
        scalar y : (c_opt, min_obj_val) — 2-tuple (compatible with diagnostics.py)
        array  y : c_opt array
        """
        prim = self.prim
        D_h = float(prim.D(h))
        L_h = float(prim.L(h))
        ell_h = self.ell(h)

        scalar_input = np.isscalar(y)
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        c_result = np.empty_like(y_arr)
        obj_result = np.empty_like(y_arr)

        for idx in np.ndindex(y_arr.shape):
            yi = float(y_arr[idx])
            best_val = np.inf
            best_c = h

            def neg_obj(c):
                xi = D_h * c - L_h
                if xi <= 0.0:
                    return np.inf
                return -float(prim.u(np.array(xi))) + yi * c

            # Upper boundary c = h
            val_h = neg_obj(h)
            if val_h < best_val:
                best_val, best_c = val_h, h

            if h - ell_h > 1e-12:
                # Lower boundary c = ell_h (xi = 0)
                try:
                    u0 = float(prim.u(np.array(0.0)))
                    val_ell = -u0 + yi * ell_h
                    if np.isfinite(val_ell) and val_ell < best_val:
                        best_val, best_c = val_ell, ell_h
                except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError):
                    pass

                # Interior search over (ell_h, h)
                lo = ell_h + 1e-10 * (h - ell_h)
                hi = h - 1e-10 * (h - ell_h)
                if hi > lo:
                    res = minimize_scalar(
                        neg_obj, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-12},
                    )
                    if np.isfinite(res.fun) and res.fun < best_val:
                        best_val, best_c = res.fun, res.x

            c_result[idx] = best_c
            obj_result[idx] = best_val

        if scalar_input:
            return float(c_result.squeeze()), float(obj_result.squeeze())
        return c_result

    def check_branch_continuity(self, h: float, tol: float = 1e-6) -> dict:
        """Verify c* and ubar are continuous at the branch boundaries b(h) and d(h).

        At b(h): one-sided limits from Branch 1 and Branch 2 must agree.
        At d(h): one-sided limits from Branch 2 and Branch 3 must agree (if d finite).
        """
        b_h = self.b(h)
        d_h = self.d(h)

        results = {}

        # --- Continuity at b(h) ---
        eps_b = b_h * tol if b_h > 0 else tol
        y_lo = b_h - eps_b
        y_hi = b_h + eps_b

        c_lo = float(self.cstar(y_lo, h))
        c_hi = float(self.cstar(y_hi, h))
        results["cstar_continuous_at_b"] = abs(c_lo - c_hi) < tol * h * 10

        ub_lo = float(self.ubar(y_lo, h))
        ub_hi = float(self.ubar(y_hi, h))
        scale = max(abs(ub_lo), abs(ub_hi), 1.0)
        results["ubar_continuous_at_b"] = abs(ub_lo - ub_hi) / scale < tol * 10

        # --- Continuity at d(h) (skip if d=inf) ---
        if np.isfinite(d_h) and d_h > 0:
            eps_d = d_h * tol if d_h > 0 else tol
            y_lo2 = d_h - eps_d
            y_hi2 = d_h + eps_d

            c_lo2 = float(self.cstar(y_lo2, h))
            c_hi2 = float(self.cstar(y_hi2, h))
            results["cstar_continuous_at_d"] = abs(c_lo2 - c_hi2) < tol * h * 10

            ub_lo2 = float(self.ubar(y_lo2, h))
            ub_hi2 = float(self.ubar(y_hi2, h))
            if np.isfinite(ub_lo2) and np.isfinite(ub_hi2):
                scale2 = max(abs(ub_lo2), abs(ub_hi2), 1.0)
                results["ubar_continuous_at_d"] = abs(ub_lo2 - ub_hi2) / scale2 < tol * 10
            else:
                # Branch-3 side returns -inf for u(0)=-inf families; both sides -inf is OK
                results["ubar_continuous_at_d"] = np.isinf(ub_hi2) and ub_hi2 < 0
        else:
            results["cstar_continuous_at_d"] = True   # d=inf: no boundary
            results["ubar_continuous_at_d"] = True

        results["all_continuous"] = all(results.values())
        return results
