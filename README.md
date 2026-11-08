# mmapy source package

`mmapy` is a small, stateful implementation of the Method of Moving
Asymptotes (MMA). It solves a separable convex approximation at each call and
retains the design and asymptote history for the next outer iteration.

The package is independent of finite-element and topology-optimization
libraries. Supply objective and inequality-constraint values and gradients at
the current point. Constraints use the convention `g(x) <= 0`.

## Use with Scikit-Topt

```bash
pip install scikit-topt
```

## Example

```python
import numpy as np
from mmapy import MMA, MMAOptions

optimizer = MMA(MMAOptions(move_limit=0.2))
x = np.array([0.8, 0.8])
lower = np.zeros(2)
upper = np.ones(2)

for _ in range(30):
    # Minimize x[0]**2 + x[1]**2 subject to x[0] + x[1] >= 0.9.
    objective = float(x @ x)
    objective_gradient = 2.0 * x
    constraints = np.array([0.9 - x.sum()])
    constraint_jacobian = np.array([[-1.0, -1.0]])
    result = optimizer.step(
        x,
        objective,
        objective_gradient,
        constraints,
        constraint_jacobian,
        lower,
        upper,
    )
    x = result.x

print(x)
```

Create one `MMA` object per optimization run. Its `step` method returns an
`MMAResult` with the next design, dual multipliers, approximate constraint
values, and dual-solver status. Call `reset()` to reuse the object for a new
problem. Multiple inequality constraints are supported.

## Development

```bash
PYTHONPATH=src pytest
```

This source package is bundled into the Scikit-Topt wheel. Installing
Scikit-Topt makes the `mmapy` module available without installing another
distribution. `mmapy` provides the generic optimizer and numerical options;
Scikit-Topt adapts its public `MMA_Config` to those options for topology
optimization. The solver focuses on continuous, differentiable problems with
box bounds and inequality constraints.
