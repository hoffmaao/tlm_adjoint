from firedrake import *
from tlm_adjoint.firedrake import *

from .test_base import *
from .test_base import referenced_vars

import gc
import numpy as np
import pytest
import ufl

pytestmark = pytest.mark.skipif(
    DEFAULT_COMM.size not in {1, 4},
    reason="tests must be run in serial, or with 4 processes")


@pytest.mark.firedrake
@seed_test
def test_GaussNewton(setup_test, test_leaks):
    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    eps = Constant(0.1234, static=True)

    def forward(F):
        u = Function(space, name="u")
        solve(inner(grad(trial), grad(test)) * dx == -inner(F, test) * dx,
              u,
              DirichletBC(space, 0.0, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        return u

    u_ref = Function(space, name="u_ref")
    interpolate_expression(u_ref, X[0] * sin(2.0 * pi * X[0]) * sin(pi * X[1]))

    def forward_J(F):
        u = forward(F)
        J = Functional(name="J")
        J.assign(0.5 * dot(grad(u - u_ref), grad(u - u_ref)) * dx  # Likelihood
                 + 0.5 * eps * dot(F, F) * dx)  # Prior
        return J

    def R_inv_action(x):
        y = var_new_conjugate_dual(x)
        assemble(inner(grad(ufl.conj(x)), grad(test)) * dx, tensor=y)
        return y

    def B_inv_action(x):
        y = var_new_conjugate_dual(x)
        assemble(ufl.conj(eps) * inner(ufl.conj(x), test) * dx, tensor=y)
        return y

    F = Function(space, name="F")
    interpolate_expression(F, sin(pi * X[0]) * exp(X[1]))

    H = Hessian(forward_J)
    H_GN = GaussNewton(forward, R_inv_action, B_inv_action=B_inv_action)

    for i in range(20):
        dm = Function(space, static=True)
        dm_arr = np.random.random(var_local_size(dm))
        if issubclass(var_dtype(dm), np.complexfloating):
            dm_arr = dm_arr + 1.0j * np.random.random(var_local_size(dm))
        var_set_values(dm, dm_arr)
        del dm_arr

        _, _, H_action = H.action(F, var_copy(dm, static=True))
        H_action_GN = H_GN.action(F, dm)

        H_action_error = var_copy(H_action)
        var_axpy(H_action_error, -1.0, H_action_GN)
        assert var_linf_norm(H_action_error) < 1.0e-18


@pytest.mark.firedrake
@seed_test
def test_CachedGaussNewton(setup_test):
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)
    eps = Constant(0.1234, static=True)

    def forward(F):
        u = Function(space, name="u")
        solve(inner(grad(trial), grad(test)) * dx == -inner(F, test) * dx,
              u,
              DirichletBC(space, 0.0, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        return u

    u_ref = Function(space, name="u_ref")
    interpolate_expression(u_ref, X[0] * sin(2.0 * pi * X[0]) * sin(pi * X[1]))

    def forward_J(F):
        u = forward(F)
        J = Functional(name="J")
        J.assign(0.5 * dot(grad(u - u_ref), grad(u - u_ref)) * dx  # Likelihood
                 + 0.5 * eps * dot(F, F) * dx)  # Prior
        return J

    def R_inv_action(x):
        y = var_new_conjugate_dual(x)
        assemble(inner(grad(ufl.conj(x)), grad(test)) * dx, tensor=y)
        return y

    def B_inv_action(x):
        y = var_new_conjugate_dual(x)
        assemble(ufl.conj(eps) * inner(ufl.conj(x), test) * dx, tensor=y)
        return y

    F = Function(space, name="F")
    interpolate_expression(F, sin(pi * X[0]) * exp(X[1]))

    H = Hessian(forward_J)
    start_manager()
    u = forward(F)
    stop_manager()
    H_GN = CachedGaussNewton(u, R_inv_action, B_inv_action=B_inv_action)

    for i in range(20):
        dm = Function(space, static=True)
        dm_arr = np.random.random(var_local_size(dm))
        if issubclass(var_dtype(dm), np.complexfloating):
            dm_arr = dm_arr + 1.0j * np.random.random(var_local_size(dm))
        var_set_values(dm, dm_arr)
        del dm_arr

        _, _, H_action = H.action(F, var_copy(dm, static=True))
        H_action_GN = H_GN.action(F, dm)

        H_action_error = var_copy(H_action)
        var_axpy(H_action_error, -1.0, H_action_GN)
        assert var_linf_norm(H_action_error) < 1.0e-18


@pytest.mark.firedrake
@seed_test
def test_CachedGaussNewton_releases_tangent_linears(setup_test):
    # Each action builds a TangentLinearMap whose tangent-linear variables are
    # stored on the forward variables, and the map's finalizer removes them.
    # The forward variables keep that storage between actions, so a map must
    # register every variable it writes to, or the tangent-linears of each
    # action after the first stay behind
    configure_checkpointing("memory", {"drop_references": False})

    mesh = UnitSquareMesh(10, 10)
    X = SpatialCoordinate(mesh)
    space = FunctionSpace(mesh, "Lagrange", 1)
    test, trial = TestFunction(space), TrialFunction(space)

    def forward(F):
        u = Function(space, name="u")
        solve(inner(grad(trial), grad(test)) * dx == -inner(F, test) * dx,
              u,
              DirichletBC(space, 0.0, "on_boundary"),
              solver_parameters=ls_parameters_cg)
        return u

    def R_inv_action(x):
        y = var_new_conjugate_dual(x)
        assemble(inner(ufl.conj(x), test) * dx, tensor=y)
        return y

    F = Function(space, name="F")
    interpolate_expression(F, sin(pi * X[0]) * exp(X[1]))
    dm = Function(space, name="dm", static=True)
    interpolate_expression(dm, X[0] + X[1])

    start_manager()
    u = forward(F)
    stop_manager()
    H_GN = CachedGaussNewton(u, R_inv_action)

    def referenced():
        gc.collect()
        return len(referenced_vars())

    H_action_GN = H_GN.action(F, dm)
    n_referenced = referenced()
    for _ in range(4):
        H_action_GN_repeat = H_GN.action(F, dm)
        H_action_error = var_copy(H_action_GN)
        var_axpy(H_action_error, -1.0, H_action_GN_repeat)
        assert var_linf_norm(H_action_error) == 0.0
        del H_action_GN_repeat, H_action_error
    assert referenced() == n_referenced
