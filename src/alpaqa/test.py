#%%

print("starting tests")

import alpaqa as pa
import numpy as np
from pprint import pprint

print("1")
solver = pa.PANOCSolver(pa.PANOCParams(), pa.LBFGSDirection(pa.LBFGSParams()))
print("2")
assert str(solver) == "PANOCSolver<LBFGS>"
print("3")
solver = pa.PANOCSolver(pa.PANOCParams(), pa.LBFGSParams())
print("4")
assert str(solver) == "PANOCSolver<LBFGS>"


class Dir(pa.PANOCDirection):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return "Dir"


assert str(Dir()) == "Dir"
solver = pa.PANOCSolver(pa.PANOCParams(), Dir())
assert str(solver) == "PANOCSolver<Dir>"

l = pa.LBFGSParams(cbfgs=pa.LBFGSParamsCBFGS(α=5))
assert l.cbfgs.α == 5
l.cbfgs.α = 100
assert l.cbfgs.α == 100

import casadi as cs

hess_prod = lambda L, x, v: cs.gradient(cs.jtimes(L, x, v, False), x)

n = 2
m = 2
x = cs.SX.sym("x", n)
λ = cs.SX.sym("λ", m)
v = cs.SX.sym("v", n)

Q = np.array([[1.5, 0.5], [0.5, 1.5]])
f_ = x.T @ Q @ x
g_ = x
L = f_ + cs.dot(λ, g_) if m > 0 else f_

f = cs.Function("f", [x], [f_])
grad_f = cs.Function("grad_f", [x], [cs.gradient(f_, x)])
g = cs.Function("g", [x], [g_])
grad_g_prod = cs.Function("grad_g_prod", [x, λ], [cs.jtimes(g_, x, λ, True)])
grad_gi = lambda x, i: grad_g_prod(x, np.eye(1, m, i))
Hess_L = cs.Function("Hess_L", [x, λ], [cs.hessian(L, x)[0]])
Hess_L_prod = cs.Function("Hess_L_prod", [x, λ, v], [hess_prod(L, x, v)])


class TestProblem(pa.Problem):
    def __init__(self):
        pa.Problem.__init__(self, n, m)
        self.D.lowerbound = [-np.inf, 0.5]
        self.D.upperbound = [+np.inf, +np.inf]

    def eval_f(self, x):
        return f(x)

    def eval_grad_f(self, x, grad_fx: np.ndarray):
        grad_fx[:] = np.ravel(grad_f(x))

    def eval_g(self, x, gg):
        gg[:] = np.ravel(g(x))

    def eval_grad_g_prod(self, x, y, gg):
        gg[:] = np.ravel(grad_g_prod(x, y))

    def eval_grad_gi(self, x, i, gg):
        gg[:] = np.ravel(grad_gi(x, i))

    def eval_hess_L(self, x, y, H):
        H[:, :] = Hess_L(x, y)

    def eval_hess_L_prod(self, x, y, v, Hv):
        Hv[:] = np.ravel(Hess_L_prod(x, y, v))


p = TestProblem()

x0 = np.array([3, 3])
y0 = np.zeros((m, ))
Σ = 1e3 * np.ones((m, ))
ε = 1e-8
solver = pa.PANOCSolver(
    pa.PANOCParams(max_iter=200, print_interval=1),
    pa.LBFGSParams(memory=5),
)
x, y, err_z, stats = solver(p, Σ, ε, x0, y0)
print(x)
print(y)
print(err_z)
pprint(stats)

solver = pa.PANOCSolver(
    pa.PANOCParams(max_iter=200, print_interval=1),
    pa.LBFGSParams(memory=5),
)
almparams = pa.ALMParams(max_iter=20, print_interval=1)
almsolver = pa.ALMSolver(almparams, solver)
x, y, stats = almsolver(p, x=x0, y=y0)

print(x)
print(y)
pprint(stats)

solver = pa.StructuredPANOCLBFGSSolver(
    pa.StructuredPANOCLBFGSParams(max_iter=200, print_interval=1),
    pa.LBFGSParams(memory=5),
)
almparams = pa.ALMParams(max_iter=20, print_interval=1)
almsolver = pa.ALMSolver(almparams, solver)
x, y, stats = almsolver(p, x=x0, y=y0)


class CustomInnerSolver(pa.InnerSolver):
    def __init__(self):
        super().__init__()
        self.solver = pa.PANOCSolver(
            pa.PANOCParams(max_iter=200, print_interval=1),
            pa.LBFGSParams(memory=5),
        )

    def get_name(self):
        return self.solver.get_name()

    def stop(self):
        return self.solver.stop()

    def __call__(self, problem, Σ, ε, always_overwrite_results, x, y):
        # TODO: always_overwrite_results
        x, y, err_z, stats = self.solver(problem, Σ, ε, x, y)

        def accumulate(acc: dict, s: dict):
            for k, v in s.items():
                if not k in ["status", "ε", "accumulator"]:
                    acc[k] = acc[k] + v if k in acc else v

        stats["accumulator"] = {"accumulate": accumulate}
        return x, y, err_z, stats


solver = CustomInnerSolver()
almparams = pa.ALMParams(max_iter=20, print_interval=1)
almsolver = pa.ALMSolver(almparams, solver)
x, y, stats = almsolver(p, x=x0, y=y0)

print(x)
print(y)
pprint(stats)

try:
    old_x0 = x0
    x0 = np.zeros((666, ))
    sol = almsolver(p, x=x0, y=y0)
except ValueError as e:
    assert e.args[0] == "Length of x does not match problem size problem.n"

x0 = old_x0

# %%

n = 2
m = 2
x = cs.SX.sym("x", n)
λ = cs.SX.sym("λ", m)
v = cs.SX.sym("v", n)

Q = np.array([[1.5, 0.5], [0.5, 1.5]])
f_ = 0.5 * x.T @ Q @ x
g_ = x
f = cs.Function("f", [x], [f_])
g = cs.Function("g", [x], [g_])

name = "testproblem"
p = pa.generate_and_compile_casadi_problem(f, g, name=name)
p.D.lowerbound = [-np.inf, 0.5]
p.D.upperbound = [+np.inf, +np.inf]
solver = pa.StructuredPANOCLBFGSSolver(
    pa.StructuredPANOCLBFGSParams(max_iter=200, print_interval=1),
    pa.LBFGSParams(memory=5),
)
almparams = pa.ALMParams(max_iter=20, print_interval=1)
almsolver = pa.ALMSolver(almparams, solver)
x, y, stats = almsolver(p, x=x0, y=y0)

print(x)
print(y)
pprint(stats)

# %%

n = 2
m = 2
x = cs.SX.sym("x", n)
p = cs.SX.sym("p", 3)

p0 = np.array([1.5, 0.5, 1.5])

Q = cs.vertcat(cs.horzcat(p[0], p[1]), cs.horzcat(p[1], p[2]))
f_ = 0.5 * x.T @ Q @ x
g_ = x
f = cs.Function("f", [x, p], [f_])
g = cs.Function("g", [x, p], [g_])

name = "testproblem"
prob = pa.generate_and_compile_casadi_problem(f, g, name=name)
prob.D.lowerbound = [-np.inf, 0.5]
prob.D.upperbound = [+np.inf, +np.inf]
prob.param = p0
solver = pa.StructuredPANOCLBFGSSolver(
    pa.StructuredPANOCLBFGSParams(max_iter=200, print_interval=1),
    pa.LBFGSParams(memory=5),
)
almparams = pa.ALMParams(max_iter=20, print_interval=1)
almsolver = pa.ALMSolver(almparams, solver)
x, y, stats = almsolver(prob, x=x0, y=y0)

print(x)
print(y)
pprint(stats)

# %% Make sure that the problem is copied

if False:
    prob.param = [1, 2, 3]
    assert np.all(prob.param == [1, 2, 3])
    prob1 = pa.ProblemWithParamWithCounters(prob)
    print(prob.param)
    print(prob1.param)
    assert np.all(prob.param == [1, 2, 3])
    assert np.all(prob1.param == [1, 2, 3])
    prob1.param = [42, 43, 44]
    print(prob.param)
    print(prob1.param)
    assert np.all(prob.param == [1, 2, 3])
    assert np.all(prob1.param == [42, 43, 44])
    print(prob.f([1, 2]))
    print(prob1.f([1, 2]))
    assert prob.f([1, 2]) == 21 / 2
    assert prob1.f([1, 2]) == 390 / 2
    assert prob1.evaluations.f == 2

    prob2 = pa.ProblemWithCounters(prob)  # params are not copied!
    print(prob.f([1, 2]))
    print(prob2.f([1, 2]))
    assert prob.f([1, 2]) == 21 / 2
    assert prob2.f([1, 2]) == 21 / 2
    prob.param = [2, 1, 3]
    print(prob.f([1, 2]))
    print(prob2.f([1, 2]))
    assert prob.f([1, 2]) == 18 / 2
    assert prob2.f([1, 2]) == 18 / 2
    assert prob1.evaluations.f == 2
    assert prob2.evaluations.f == 4
else:
    from copy import copy
    try:
        prob1 = copy(prob)
        assert False
    except:
        pass
    prob1 = prob

# %%

print(prob1.n, prob1.m, prob1.C, prob1.D)

prob1.param = p0
x, y, stats = almsolver(prob1)  # without initial guess

print(x)
print(y)
pprint(stats)
print(prob1.evaluations.f)
print(prob1.evaluations.grad_f)
print(prob1.evaluations.ψ_grad_ψ)
print(prob1.evaluations.grad_L)
print(prob1.evaluations.ψ)

# %%

f = lambda x: float(np.cosh(x) - x * x + x)
grad_f = lambda x: np.sinh(x) - 2 * x + 1
C = pa.Box([10], [-2.5])
x0 = [5]
x, stats = pa.panoc(f, grad_f, C, x0, 1e-12, pa.PANOCParams(print_interval=1),
                    pa.LBFGSParams())
print(x)
pprint(stats)

# %%

f = lambda x: float(np.cosh(x) - x * x + x)
grad_f = lambda x: np.sinh(x) - 2 * x + 1
C = pa.Box([10], [-2.5])
x, stats = pa.panoc(f, grad_f, C, params=pa.PANOCParams(print_interval=1))
print(x)
pprint(stats)

# %%

# Rosenbrock without general constraints
x1, x2 = cs.SX.sym("x1"), cs.SX.sym("x2")
p = cs.SX.sym("p")
f_expr = (1 - x1) ** 2 + p * (x2 - x1 ** 2) ** 2
x = cs.vertcat(x1, x2)
f = cs.Function("f", [x, p], [f_expr])
g = None
prob = pa.generate_and_compile_casadi_problem(f, g)

prob.C.lowerbound = [-0.25, -0.5]       # -0.25 <= x1 <= 0.9
prob.C.upperbound = [0.9, 0.8]          # -0.5  <= x2 <= 0.8
prob.param = [100.]

inner_solver = pa.StructuredPANOCLBFGSSolver(
    panoc_params={
        'max_iter': 1000,
        'stop_crit': pa.PANOCStopCrit.ApproxKKT,
    },
    lbfgs_params={
        'memory': 10,
    },
)

solver = pa.ALMSolver(
    alm_params={
        'ε': 1e-14,
        'δ': 1e-14,
        'Σ_0': 0,
        'σ_0': 2,
        'Δ': 20,
    },
    inner_solver=inner_solver
)

x0 = np.array([0.1, 1.8]) # decision variables
x_sol, y_sol, stats = solver(prob, x0)

# Print the results
print(stats["status"])
print(f"Solution:      {x_sol}")
print(f"Multipliers:   {y_sol}")
print(f"Cost:          {prob.eval_f(x_sol)}")
print(f"ε:             {stats['ε']}")
print(f"δ:             {stats['δ']}")
err = np.linalg.norm(x_sol - [0.8947558975956834, 0.8])
print(err)
assert err <= 1e-12
from pprint import pprint
pprint(stats)


# %%

try:
    pa.PANOCParams(max_iter=1e3)
    assert False
except RuntimeError as e:
    print(e)

# %%

p = pa.Problem(2, 2)
try:
    p.eval_f(x_sol)
    assert False
except NotImplementedError as e:
    print(type(e), e)

# %%

print("Success!")