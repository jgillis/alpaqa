import casadi as cs
from typing import Tuple


def generate_casadi_problem(
    f: cs.Function,
    g: cs.Function,
    second_order: bool = False,
    name: str = "alpaqa_problem",
) -> Tuple[cs.CodeGenerator, int, int, int]:
    """Convert the objective and constraint functions into a CasADi code
    generator.

    :param f:            Objective function.
    :param g:            Constraint function.
    :param second_order: Whether to generate functions for evaluating Hessians.
    :param name: Optional string description of the problem (used for filename).

    :return:   * Code generator that generates the functions and derivatives
                 used by the solvers.
               * Dimensions of the decision variables (primal dimension).
               * Number of nonlinear constraints (dual dimension).
               * Number of parameters.
    """

    assert f.n_in() in [1, 2]
    assert f.n_in() == g.n_in()
    assert f.size1_in(0) == g.size1_in(0)
    if f.n_in() == 2:
        assert f.size1_in(1) == g.size1_in(1)
    assert f.n_out() == 1
    assert g.n_out() == 1
    n = f.size1_in(0)
    m = g.size1_out(0)
    p = f.size1_in(1) if f.n_in() == 2 else 0
    xp = (f.sx_in(0), f.sx_in(1)) if f.n_in() == 2 else (f.sx_in(0), )
    xp_def = (f.sx_in(0), f.sx_in(1)) if f.n_in() == 2 \
        else (f.sx_in(0), cs.SX.sym("p", 0))
    xp_names = (f.name_in(0), f.name_in(1)) if f.n_in() == 2 \
          else (f.name_in(0), "p")
    x = xp[0]
    y = cs.SX.sym("y", m)
    v = cs.SX.sym("v", n)
    Σ = cs.SX.sym("Σ", m)
    zl = cs.SX.sym("zl", m)
    zu = cs.SX.sym("zu", m)

    L = f(*xp) + cs.dot(y, g(*xp)) if m > 0 else f(*xp)
    ζ = g(*xp) + (y / Σ)
    ẑ = cs.fmax(zl, cs.fmin(ζ, zu))
    d = ζ - ẑ
    ŷ = Σ * d
    ψ = f(*xp) + 0.5 * cs.dot(ŷ, d)

    cgname = f"{name}.c"
    cg = cs.CodeGenerator(cgname)
    cg.add(cs.Function(
        "f",
        [*xp_def],
        [f(*xp)],
        [*xp_names],
        ["f"],
    ))
    cg.add(cs.Function(
        "g",
        [*xp_def],
        [g(*xp)],
        [*xp_names],
        ["g"],
    ))
    cg.add(
        cs.Function(
            "psi_grad_psi",
            [*xp_def, y, Σ, zl, zu],
            [ψ, cs.gradient(ψ, x)],
            [*xp_names, "y", "Σ", "zl", "zu"],
            ["ψ", "grad_ψ"],
        ))
    cg.add(
        cs.Function(
            "grad_psi",
            [*xp_def, y, Σ, zl, zu],
            [cs.gradient(ψ, x)],
            [*xp_names, "y", "Σ", "zl", "zu"],
            ["grad_ψ"],
        ))
    cg.add(
        cs.Function(
            "grad_L",
            [*xp_def, y],
            [cs.gradient(L, x)],
            [*xp_names, "y"],
            ["grad_L"],
        ))
    cg.add(
        cs.Function(
            "psi",
            [*xp_def, y, Σ, zl, zu],
            [ψ, ŷ],
            [*xp_names, "y", "Σ", "zl", "zu"],
            ["ψ", "ŷ"],
        ))

    if second_order:
        cg.add(
            cs.Function(
                "hess_L",
                [*xp_def, y],
                [cs.hessian(L, x)[0]],
                [*xp_names, "y"],
                ["hess_L"],
            ))
        cg.add(
            cs.Function(
                "hess_L_prod",
                [*xp_def, y, v],
                [cs.gradient(cs.jtimes(L, x, v, False), x)],
                [*xp_names, "y", "v"],
                ["hess_L_prod"],
            ))
    return cg, n, m, p