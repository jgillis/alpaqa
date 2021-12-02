#pragma once

#include <alpaqa/util/problem.hpp>

#include <pybind11/pybind11.h>

template <class ProblemBase = alpaqa::Problem>
class ProblemTrampoline : ProblemBase {
    using ProblemBase::ProblemBase;
    alpaqa::real_t eval_f(alpaqa::crvec x) const override { PYBIND11_OVERRIDE(alpaqa::real_t, ProblemBase, eval_f, x); }
    void eval_grad_f(alpaqa::crvec x,alpaqa:: rvec grad_fx) const override { PYBIND11_OVERRIDE(void, ProblemBase, eval_grad_f, x, grad_fx); }
    void eval_g(alpaqa::crvec x, alpaqa::rvec gx) const override { PYBIND11_OVERRIDE(void, ProblemBase, eval_g, x, gx); }
    void eval_grad_g_prod(alpaqa::crvec x, alpaqa::crvec y, alpaqa::rvec grad_gxy) const override { PYBIND11_OVERRIDE(void, ProblemBase, eval_grad_g_prod, x, y, grad_gxy); }
    void eval_grad_gi(alpaqa::crvec x, unsigned i, alpaqa::rvec grad_gi) const override { PYBIND11_OVERRIDE(void, ProblemBase, eval_grad_gi, x, i, grad_gi); }
    void eval_hess_L_prod(alpaqa::crvec x, alpaqa::crvec y, alpaqa::crvec v, alpaqa::rvec Hv) const override { PYBIND11_OVERRIDE(void, ProblemBase, eval_hess_L_prod, x, y, v, Hv); }
    void eval_hess_L(alpaqa::crvec x, alpaqa::crvec y, alpaqa::rmat H) const override { PYBIND11_OVERRIDE(void, ProblemBase, eval_hess_L, x, y, H); }
};
