#include <alpaqa/util/problem.hpp>

namespace alpaqa {

real_t Problem::eval_f(crvec) const { throw not_implemented_error("eval_f"); }
void Problem::eval_grad_f(crvec, rvec) const {
    throw not_implemented_error("eval_grad_f");
}
void Problem::eval_g(crvec, rvec) const {
    throw not_implemented_error("eval_g");
}
void Problem::eval_grad_g_prod(crvec, crvec, rvec) const {
    throw not_implemented_error("eval_grad_g_prod");
}
void Problem::eval_grad_gi(crvec, unsigned, rvec) const {
    throw not_implemented_error("eval_grad_gi");
}
void Problem::eval_hess_L_prod(crvec, crvec, crvec, rvec) const {
    throw not_implemented_error("eval_hess_L_prod");
}
void Problem::eval_hess_L(crvec, crvec, rmat) const {
    throw not_implemented_error("eval_hess_L");
}

real_t Problem::eval_f_grad_f(crvec x, rvec grad_fx) const {
    eval_grad_f(x, grad_fx);
    return eval_f(x);
}
void Problem::eval_g_grad_g_prod(crvec x, crvec y, rvec g,
                                 rvec grad_gxy) const {
    eval_g(x, g);
    eval_grad_g_prod(x, y, grad_gxy);
}

real_t Problem::eval_ψ_ŷ(crvec x, crvec y, real_t ρ, rvec ŷ) const {
    if (m == 0) /* [[unlikely]] */
        return eval_f(x);

    real_t dᵀŷ = eval_ŷ_dᵀŷ(x, y, ρ, ŷ);
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t f = eval_f(x);
    real_t ψ = f + 0.5 * dᵀŷ;
    return ψ;
}

real_t Problem::eval_ψ_ŷ(crvec x, crvec y, crvec Σ, rvec ŷ) const {
    if (m == 0) /* [[unlikely]] */
        return eval_f(x);

    real_t dᵀŷ = eval_ŷ_dᵀŷ(x, y, Σ, ŷ);
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t f = eval_f(x);
    real_t ψ = f + 0.5 * dᵀŷ;
    return ψ;
}

void Problem::eval_grad_ψ(crvec x, crvec y, real_t ρ, rvec grad_ψ, rvec work_n,
                          rvec work_m) const {
    eval_ŷ_dᵀŷ(x, y, ρ, work_m);
    eval_grad_f(x, grad_ψ);
    eval_grad_g_prod(x, work_m, work_n);
    grad_ψ += work_n;
}

void Problem::eval_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ, rvec work_n,
                          rvec work_m) const {
    eval_ŷ_dᵀŷ(x, y, Σ, work_m);
    eval_grad_f(x, grad_ψ);
    eval_grad_g_prod(x, work_m, work_n);
    grad_ψ += work_n;
}

real_t Problem::eval_ŷ_dᵀŷ(crvec x, crvec y, real_t ρ, rvec ŷ) const {
    if (m == 0) /* [[unlikely]] */
        return 0;

    // g(x)
    eval_g(x, ŷ);
    // ζ = g(x) + ρ⁻¹y
    ŷ += (1 / ρ) * y;
    // d = ζ - Π(ζ, D)
    ŷ = projecting_difference(ŷ, D);
    // dᵀŷ, ŷ = ρ d
    real_t dᵀŷ = ρ * ŷ.dot(ŷ);
    ŷ *= ρ;
    return dᵀŷ;
}

real_t Problem::eval_ŷ_dᵀŷ(crvec x, crvec y, crvec Σ, rvec ŷ) const {
    if (m == 0) /* [[unlikely]] */
        return 0;

    // g(x)
    eval_g(x, ŷ);
    // ζ = g(x) + Σ⁻¹y
    ŷ += Σ.asDiagonal().inverse() * y;
    // d = ζ - Π(ζ, D)
    ŷ = projecting_difference(ŷ, D);
    // dᵀŷ, ŷ = Σ d
    real_t dᵀŷ = 0;
    for (unsigned i = 0; i < m; ++i) {
        dᵀŷ += ŷ(i) * Σ(i) * ŷ(i); // TODO: vectorize
        ŷ(i) = Σ(i) * ŷ(i);
    }
    return dᵀŷ;
}

void Problem::eval_grad_ψ_from_ŷ(crvec x, crvec ŷ, rvec grad_ψ,
                                 rvec work_n) const {
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    eval_grad_f(x, grad_ψ);
    if (m != 0) /* [[likely]] */ {
        eval_grad_g_prod(x, ŷ, work_n);
        grad_ψ += work_n;
    }
}

#if 0 
// Reference implementation. TODO: verify in tests
real_t Problem::eval_ψ_grad_ψ(crvec x, crvec y, real_t ρ, rvec grad_ψ,
                              rvec work_n, rvec work_m) const {
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = eval_ψ_ŷ(x, y, ρ, work_m);
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    eval_grad_ψ_from_ŷ(x, work_m, grad_ψ, work_n);
    return ψ;
}

real_t Problem::eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ,
                              rvec work_n, rvec work_m) const {
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = eval_ψ_ŷ(x, y, Σ, work_m);
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    eval_grad_ψ_from_ŷ(x, work_m, grad_ψ, work_n);
    return ψ;
}
#else
real_t Problem::eval_ψ_grad_ψ(crvec x, crvec y, real_t ρ, rvec grad_ψ,
                              rvec work_n, rvec work_m) const {
    if (m == 0) /* [[unlikely]] */
        return eval_f_grad_f(x, grad_ψ);

    auto &ŷ       = work_m;
    auto &grad_gŷ = work_n;

    real_t dᵀŷ = eval_ŷ_dᵀŷ(x, y, ρ, ŷ);
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t f = eval_f_grad_f(x, grad_ψ);
    real_t ψ = f + 0.5 * dᵀŷ;
    // ∇ψ(x) = ∇f(x) + ∇g(x) ŷ
    eval_grad_g_prod(x, ŷ, grad_gŷ);
    grad_ψ += grad_gŷ;
    return ψ;
}

real_t Problem::eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ,
                              rvec work_n, rvec work_m) const {
    if (m == 0) /* [[unlikely]] */
        return eval_f_grad_f(x, grad_ψ);

    auto &ŷ       = work_m;
    auto &grad_gŷ = work_n;

    real_t dᵀŷ = eval_ŷ_dᵀŷ(x, y, Σ, ŷ);
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t f = eval_f_grad_f(x, grad_ψ);
    real_t ψ = f + 0.5 * dᵀŷ;
    // ∇ψ(x) = ∇f(x) + ∇g(x) ŷ
    eval_grad_g_prod(x, ŷ, grad_gŷ);
    grad_ψ += grad_gŷ;
    return ψ;
}
#endif

std::unique_ptr<Problem> Problem::clone() const & {
    return std::unique_ptr<Problem>(new Problem(*this));
}
std::unique_ptr<Problem> Problem::clone() && {
    return std::unique_ptr<Problem>(new Problem(std::move(*this)));
}

} // namespace alpaqa
