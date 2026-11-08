import numpy as np

from mmapy import MMA, MMAOptions


def test_mma_solves_a_linear_single_constraint_subproblem():
    optimizer = MMA(MMAOptions(move_limit=0.4))
    result = optimizer.step(
        x=[0.8],
        objective=0.8,
        objective_gradient=[1.0],
        constraints=[0.5 - 0.8],
        constraint_jacobian=[[-1.0]],
        lower_bounds=[0.0],
        upper_bounds=[1.0],
    )

    assert 0.5 <= result.x[0] < 0.8
    assert abs(result.approximated_constraints[0]) < 1e-6
    assert result.multipliers[0] >= 0.0
    assert result.dual_success


def test_mma_supports_multiple_inequality_constraints():
    optimizer = MMA(MMAOptions(move_limit=0.4))
    result = optimizer.step(
        x=[0.8, 0.8],
        objective=1.6,
        objective_gradient=[1.0, 1.0],
        constraints=[0.4 - 0.8, 0.6 - 0.8],
        constraint_jacobian=[[-1.0, 0.0], [0.0, -1.0]],
        lower_bounds=[0.0, 0.0],
        upper_bounds=[1.0, 1.0],
    )

    assert result.x[0] >= 0.4
    assert result.x[1] >= 0.6
    assert np.all(np.abs(result.approximated_constraints) < 2e-3)
    assert result.multipliers.shape == (2,)


def test_mma_converges_on_a_quadratic_problem_with_a_linear_constraint():
    optimizer = MMA(MMAOptions(move_limit=0.2))
    x = np.array([0.8, 0.8])
    for _ in range(30):
        result = optimizer.step(
            x=x,
            objective=float(x @ x),
            objective_gradient=2.0 * x,
            constraints=[0.9 - x.sum()],
            constraint_jacobian=[[-1.0, -1.0]],
            lower_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
        )
        assert result.dual_success
        x = result.x

    np.testing.assert_allclose(x, [0.45, 0.45], atol=1e-5)
    assert x.sum() >= 0.9 - 1e-6


def test_mma_keeps_asymptote_history_and_reset_clears_it():
    optimizer = MMA(MMAOptions(move_limit=0.2))
    args = dict(
        objective=0.0,
        objective_gradient=[-1.0],
        constraints=[],
        constraint_jacobian=np.empty((0, 1)),
        lower_bounds=[0.0],
        upper_bounds=[1.0],
    )
    first = optimizer.step(x=[0.5], **args)
    assert optimizer._lower_asymptote is not None
    assert optimizer._upper_asymptote is not None

    optimizer.step(x=first.x, **args)
    assert optimizer._previous_previous_x is not None

    optimizer.reset()
    assert optimizer._previous_x is None
    assert optimizer._previous_previous_x is None
    assert optimizer._lower_asymptote is None
    assert optimizer._upper_asymptote is None


def test_mma_rejects_inconsistent_constraint_jacobian_shape():
    optimizer = MMA()
    try:
        optimizer.step(
            x=[0.5, 0.5],
            objective=1.0,
            objective_gradient=[1.0, 1.0],
            constraints=[-0.1],
            constraint_jacobian=[[1.0]],
            lower_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
        )
    except ValueError as error:
        assert "constraint_jacobian" in str(error)
    else:
        raise AssertionError("shape mismatch should have raised ValueError")
