#include <alpaqa/util/problem.hpp>

#include <iomanip>
#include <ostream>

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
real_t Problem::eval_f_g(crvec x, rvec g) const {
    eval_g(x, g);
    return eval_f(x);
}

real_t Problem::eval_f_grad_f_g(crvec x, rvec grad_fx, rvec g) const {
    eval_g(x, g);
    return eval_f_grad_f(x, grad_fx);
}

void Problem::eval_grad_f_grad_g_prod(crvec x, crvec y, rvec grad_f,
                                      rvec grad_gxy) const {
    eval_grad_f(x, grad_f);
    eval_grad_g_prod(x, y, grad_gxy);
}

void Problem::eval_grad_L(crvec x, crvec y, rvec grad_L, rvec work_n) const {
    // ∇L = ∇f(x) + ∇g(x) y
    eval_grad_f_grad_g_prod(x, y, grad_L, work_n);
    grad_L += work_n;
}

real_t Problem::eval_ψ_ŷ(crvec x, crvec y, crvec Σ, rvec ŷ) const {
    if (m == 0) /* [[unlikely]] */
        return eval_f(x);

    real_t f   = eval_f_g(x, ŷ);
    real_t dᵀŷ = calc_ŷ_dᵀŷ(ŷ, y, Σ);
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = f + 0.5 * dᵀŷ;
    return ψ;
}

void Problem::eval_grad_ψ_from_ŷ(crvec x, crvec ŷ, rvec grad_ψ,
                                 rvec work_n) const {
    if (m == 0) /* [[unlikely]] */ {
        eval_grad_f(x, grad_ψ);
    } else {
        eval_grad_L(x, ŷ, grad_ψ, work_n);
    }
}

void Problem::eval_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ, rvec work_n,
                          rvec work_m) const {
    if (m == 0) /* [[unlikely]] */ {
        eval_grad_f(x, grad_ψ);
    } else {
        eval_g(x, work_m);
        (void)calc_ŷ_dᵀŷ(work_m, y, Σ);
        eval_grad_ψ_from_ŷ(x, work_m, grad_ψ, work_n);
    }
}

#if 0 
// Reference implementation. TODO: verify in tests
real_t Problem::eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ,
                              rvec work_n, rvec work_m) const {
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t ψ = eval_ψ_ŷ(x, y, Σ, work_m);
    // ∇ψ = ∇f(x) + ∇g(x) ŷ
    eval_grad_ψ_from_ŷ(x, work_m, grad_ψ, work_n);
    return ψ;
}
#else
real_t Problem::eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ,
                              rvec work_n, rvec work_m) const {
    if (m == 0) /* [[unlikely]] */
        return eval_f_grad_f(x, grad_ψ);

    auto &ŷ = work_m;
    // ψ(x) = f(x) + ½ dᵀŷ
    real_t f   = eval_f_g(x, ŷ);
    real_t dᵀŷ = calc_ŷ_dᵀŷ(ŷ, y, Σ);
    real_t ψ   = f + 0.5 * dᵀŷ;
    // ∇ψ(x) = ∇f(x) + ∇g(x) ŷ
    eval_grad_L(x, ŷ, grad_ψ, work_n);
    return ψ;
}
#endif

real_t Problem::calc_ŷ_dᵀŷ(rvec g_ŷ, crvec y, crvec Σ) const {
    if (Σ.size() == 1) {
        // ζ = g(x) + Σ⁻¹y
        g_ŷ += (1 / Σ(0)) * y;
        // d = ζ - Π(ζ, D)
        g_ŷ = projecting_difference(g_ŷ, D);
        // dᵀŷ, ŷ = Σ d
        real_t dᵀŷ = Σ(0) * g_ŷ.dot(g_ŷ);
        g_ŷ *= Σ(0);
        return dᵀŷ;
    } else {
        // ζ = g(x) + Σ⁻¹y
        g_ŷ += Σ.asDiagonal().inverse() * y;
        // d = ζ - Π(ζ, D)
        g_ŷ = projecting_difference(g_ŷ, D);
        // dᵀŷ, ŷ = Σ d
        real_t dᵀŷ = 0;
        for (unsigned i = 0; i < m; ++i) {
            dᵀŷ += g_ŷ(i) * Σ(i) * g_ŷ(i); // TODO: vectorize
            g_ŷ(i) = Σ(i) * g_ŷ(i);
        }
        return dᵀŷ;
    }
}

std::unique_ptr<Problem> Problem::clone() const & {
    return std::unique_ptr<Problem>(new Problem(*this));
}
std::unique_ptr<Problem> Problem::clone() && {
    return std::unique_ptr<Problem>(new Problem(std::move(*this)));
}

std::ostream &operator<<(std::ostream &os, const EvalCounter &c) {
    auto cnt = [](auto t) { return std::chrono::duration<double>(t).count(); };
    os << "                 f:" << std::setw(6) << c.f << "  (" << cnt(c.time.f)
       << " s)\r\n";
    os << "            grad_f:" << std::setw(6) << c.grad_f << "  ("
       << cnt(c.time.grad_f) << " s)\r\n";
    os << "          f_grad_f:" << std::setw(6) << c.f_grad_f << "  ("
       << cnt(c.time.f_grad_f) << " s)\r\n";
    os << "               f_g:" << std::setw(6) << c.f_g << "  ("
       << cnt(c.time.f_g) << " s)\r\n";
    os << "        f_grad_f_g:" << std::setw(6) << c.f_grad_f_g << "  ("
       << cnt(c.time.f_grad_f_g) << " s)\r\n";
    os << "grad_f_grad_g_prod:" << std::setw(6) << c.grad_f_grad_g_prod << "  ("
       << cnt(c.time.grad_f_grad_g_prod) << " s)\r\n";
    os << "                 g:" << std::setw(6) << c.g << "  (" << cnt(c.time.g)
       << " s)\r\n";
    os << "       grad_g_prod:" << std::setw(6) << c.grad_g_prod << "  ("
       << cnt(c.time.grad_g_prod) << " s)\r\n";
    os << "           grad_gi:" << std::setw(6) << c.grad_gi << "  ("
       << cnt(c.time.grad_gi) << " s)\r\n";
    os << "            grad_L:" << std::setw(6) << c.grad_L << "  ("
       << cnt(c.time.grad_L) << " s)\r\n";
    os << "       hess_L_prod:" << std::setw(6) << c.hess_L_prod << "  ("
       << cnt(c.time.hess_L_prod) << " s)\r\n";
    os << "            hess_L:" << std::setw(6) << c.hess_L << "  ("
       << cnt(c.time.hess_L) << " s)\r\n";
    os << "                 ψ:" << std::setw(6) << c.ψ << "  (" << cnt(c.time.ψ)
       << " s)\r\n";
    os << "            grad_ψ:" << std::setw(6) << c.grad_ψ << "  ("
       << cnt(c.time.grad_ψ) << " s)\r\n";
    os << "     grad_ψ_from_ŷ:" << std::setw(6) << c.grad_ψ_from_ŷ << "  ("
       << cnt(c.time.grad_ψ_from_ŷ) << " s)\r\n";
    os << "          ψ_grad_ψ:" << std::setw(6) << c.ψ_grad_ψ << "  ("
       << cnt(c.time.ψ_grad_ψ) << " s)";
    return os;
}

} // namespace alpaqa
