"""A stateful, separable-convex MMA optimizer with multiple constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MMAOptions:
    """Numerical settings for the MMA outer iteration."""

    move_limit: float | None = None
    initial_asymptote: float = 0.5
    asymptote_increase: float = 1.2
    asymptote_decrease: float = 0.7
    asymptote_minimum: float = 0.01
    asymptote_maximum: float = 10.0
    regularization: float = 1e-5
    dual_tolerance: float = 1e-9
    dual_max_iterations: int = 200

    def __post_init__(self) -> None:
        if self.move_limit is not None and self.move_limit <= 0.0:
            raise ValueError("move_limit must be positive")
        if self.initial_asymptote <= 0.0:
            raise ValueError("initial_asymptote must be positive")
        if self.asymptote_increase < 1.0:
            raise ValueError("asymptote_increase must be at least one")
        if not 0.0 < self.asymptote_decrease <= 1.0:
            raise ValueError("asymptote_decrease must be in (0, 1]")
        if not 0.0 < self.asymptote_minimum < self.asymptote_maximum:
            raise ValueError("asymptote limits must satisfy 0 < min < max")
        if self.regularization <= 0.0:
            raise ValueError("regularization must be positive")
        if self.dual_tolerance <= 0.0:
            raise ValueError("dual_tolerance must be positive")
        if self.dual_max_iterations < 1:
            raise ValueError("dual_max_iterations must be positive")


@dataclass(frozen=True)
class MMAResult:
    """Result of one MMA subproblem solve."""

    x: FloatArray
    multipliers: FloatArray
    approximated_constraints: FloatArray
    dual_iterations: int
    dual_success: bool


def _asymptote_coefficients(
    gradient: FloatArray,
    x: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    regularization: float,
) -> tuple[FloatArray, FloatArray]:
    """Return reciprocal coefficients matching value and gradient at x."""
    span = upper - lower
    p = np.maximum(gradient, 0.0)
    q = np.maximum(-gradient, 0.0)
    stabilizer = 0.001 * (p + q) + regularization / span
    return (
        (p + stabilizer) * (upper - x) ** 2,
        (q + stabilizer) * (x - lower) ** 2,
    )


def _approximation(
    x: FloatArray,
    x0: FloatArray,
    value0: FloatArray,
    p: FloatArray,
    q: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    """Evaluate one or more reciprocal MMA approximations."""
    return value0 + np.sum(
        p * (1.0 / (upper - x) - 1.0 / (upper - x0))
        + q * (1.0 / (x - lower) - 1.0 / (x0 - lower)),
        axis=-1,
    )


class MMA:
    """Stateful MMA optimizer for differentiable bound-constrained problems.

    Call :meth:`step` once per outer iteration with the objective and
    inequality constraints evaluated at the current design. Constraint values
    must use the convention ``g(x) <= 0``. The optimizer retains design and
    asymptote history between calls.

    Parameters
    ----------
    config:
        Moving-asymptote and dual-solver settings.
    """

    def __init__(self, config: MMAOptions | None = None) -> None:
        self.config = config or MMAOptions()
        self._previous_x: FloatArray | None = None
        self._previous_previous_x: FloatArray | None = None
        self._lower_asymptote: FloatArray | None = None
        self._upper_asymptote: FloatArray | None = None

    def reset(self) -> None:
        """Clear iteration history before starting a new optimization."""
        self._previous_x = None
        self._previous_previous_x = None
        self._lower_asymptote = None
        self._upper_asymptote = None

    def step(
        self,
        x: ArrayLike,
        objective: float,
        objective_gradient: ArrayLike,
        constraints: ArrayLike,
        constraint_jacobian: ArrayLike,
        lower_bounds: ArrayLike,
        upper_bounds: ArrayLike,
        *,
        move_limit: float | ArrayLike | None = None,
    ) -> MMAResult:
        """Solve one local MMA approximation and update optimizer history.

        ``constraint_jacobian`` has shape ``(n_constraints, n_variables)``.
        Empty one-dimensional inputs are accepted when there are no
        constraints. Bounds are global design bounds; an optional move limit
        further restricts this iteration's step.
        """
        x0 = np.asarray(x, dtype=float)
        df0 = np.asarray(objective_gradient, dtype=float)
        g0 = np.atleast_1d(np.asarray(constraints, dtype=float))
        dg0 = np.asarray(constraint_jacobian, dtype=float)
        xmin = np.broadcast_to(np.asarray(lower_bounds, dtype=float), x0.shape)
        xmax = np.broadcast_to(np.asarray(upper_bounds, dtype=float), x0.shape)
        if x0.ndim != 1 or x0.size == 0:
            raise ValueError("x must be a nonempty one-dimensional array")
        n = x0.size
        if df0.shape != (n,):
            raise ValueError("objective_gradient must have shape (n_variables,)")
        if xmin.shape != (n,) or xmax.shape != (n,):
            raise ValueError("bounds must broadcast to x")
        if (
            np.any(~np.isfinite(x0))
            or np.any(~np.isfinite(df0))
            or np.any(~np.isfinite(xmin))
            or np.any(~np.isfinite(xmax))
        ):
            raise ValueError("x, gradients, and bounds must be finite")
        if np.any(~np.isfinite(g0)) or np.any(~np.isfinite(dg0)):
            raise ValueError("constraints and constraint_jacobian must be finite")
        if np.any(xmin >= xmax) or np.any(x0 < xmin) or np.any(x0 > xmax):
            raise ValueError("x must lie within nondegenerate global bounds")

        m = g0.size
        if m == 0:
            dg0 = np.empty((0, n), dtype=float)
        elif dg0.shape != (m, n):
            raise ValueError(
                "constraint_jacobian must have shape (n_constraints, n_variables)"
            )
        if not np.isfinite(objective):
            raise ValueError("objective must be finite")

        cfg = self.config
        span = xmax - xmin
        low, upp = self._moving_asymptotes(x0, span)
        move = cfg.move_limit if move_limit is None else move_limit
        if move is None:
            move_array = span
        else:
            move_array = np.broadcast_to(np.asarray(move, dtype=float), x0.shape)
            if np.any(~np.isfinite(move_array)) or np.any(move_array <= 0.0):
                raise ValueError("move_limit must contain finite positive values")
        alpha = np.maximum.reduce((
            xmin,
            low + 0.1 * (x0 - low),
            x0 - move_array,
        ))
        beta = np.minimum.reduce((
            xmax,
            upp - 0.1 * (upp - x0),
            x0 + move_array,
        ))
        if np.any(alpha >= beta):
            raise ValueError("MMA move limits leave no feasible step")

        p0, q0 = _asymptote_coefficients(
            df0, x0, low, upp, cfg.regularization
        )
        if m:
            p_constraints = np.empty((m, n), dtype=float)
            q_constraints = np.empty((m, n), dtype=float)
            for i in range(m):
                p_constraints[i], q_constraints[i] = _asymptote_coefficients(
                    dg0[i], x0, low, upp, cfg.regularization
                )
        else:
            p_constraints = np.empty((0, n), dtype=float)
            q_constraints = np.empty((0, n), dtype=float)

        def primal(multipliers: FloatArray) -> FloatArray:
            p = p0 + multipliers @ p_constraints if m else p0
            q = q0 + multipliers @ q_constraints if m else q0
            sqrt_p = np.sqrt(np.maximum(p, np.finfo(float).tiny))
            sqrt_q = np.sqrt(np.maximum(q, np.finfo(float).tiny))
            unconstrained = (
                sqrt_q * upp + sqrt_p * low
            ) / (sqrt_p + sqrt_q)
            return np.clip(unconstrained, alpha, beta)

        def objective_approximation(y: FloatArray) -> float:
            return float(_approximation(
                y, x0, np.asarray(objective), p0, q0, low, upp
            ))

        def constraints_approximation(y: FloatArray) -> FloatArray:
            if not m:
                return np.empty(0, dtype=float)
            return _approximation(
                y, x0, g0, p_constraints, q_constraints, low, upp
            )

        if m:
            def negative_dual(lam: FloatArray) -> tuple[float, FloatArray]:
                y = primal(lam)
                g_approx = constraints_approximation(y)
                value = objective_approximation(y) + float(lam @ g_approx)
                return -value, -g_approx

            dual = minimize(
                negative_dual,
                np.zeros(m, dtype=float),
                method="L-BFGS-B",
                jac=True,
                bounds=[(0.0, None)] * m,
                options={
                    "ftol": cfg.dual_tolerance,
                    "gtol": cfg.dual_tolerance,
                    "maxiter": cfg.dual_max_iterations,
                    "maxls": 40,
                },
            )
            multipliers = np.asarray(dual.x, dtype=float)
            x_new = primal(multipliers)
            approximated_constraints = constraints_approximation(x_new)
            dual_iterations = int(dual.nit)
            dual_success = bool(dual.success)
        else:
            multipliers = np.empty(0, dtype=float)
            x_new = primal(multipliers)
            approximated_constraints = np.empty(0, dtype=float)
            dual_iterations = 0
            dual_success = True

        self._advance_history(x0, low, upp)
        return MMAResult(
            x=x_new.copy(),
            multipliers=multipliers.copy(),
            approximated_constraints=approximated_constraints.copy(),
            dual_iterations=dual_iterations,
            dual_success=dual_success,
        )

    def _moving_asymptotes(
        self,
        x: FloatArray,
        span: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        cfg = self.config
        if self._lower_asymptote is None or self._upper_asymptote is None:
            distance = cfg.initial_asymptote * span
            return x - distance, x + distance

        factor = np.ones_like(x)
        if self._previous_x is not None and self._previous_previous_x is not None:
            trend = (x - self._previous_x) * (
                self._previous_x - self._previous_previous_x
            )
            factor[trend > 0.0] = cfg.asymptote_increase
            factor[trend < 0.0] = cfg.asymptote_decrease

        lower = x - factor * (self._previous_x - self._lower_asymptote)
        upper = x + factor * (self._upper_asymptote - self._previous_x)
        min_distance = cfg.asymptote_minimum * span
        max_distance = cfg.asymptote_maximum * span
        lower = np.minimum(x - min_distance, np.maximum(x - max_distance, lower))
        upper = np.maximum(x + min_distance, np.minimum(x + max_distance, upper))
        return lower, upper

    def _advance_history(
        self,
        x: FloatArray,
        lower: FloatArray,
        upper: FloatArray,
    ) -> None:
        if self._previous_x is not None:
            self._previous_previous_x = self._previous_x.copy()
        self._previous_x = x.copy()
        self._lower_asymptote = lower.copy()
        self._upper_asymptote = upper.copy()


__all__ = ["MMA", "MMAOptions", "MMAResult"]
