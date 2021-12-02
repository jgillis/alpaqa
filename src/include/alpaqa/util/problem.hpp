#pragma once

#include "box.hpp"

#include <cassert>
#include <chrono>
#include <functional>
#include <iosfwd>
#include <memory>
#include <stdexcept>
#include <type_traits>

namespace alpaqa {

struct not_implemented_error : std::logic_error {
    using std::logic_error::logic_error;
};

/**
 * @class Problem
 * @brief   Problem description for minimization problems.
 * 
 * @f[ \begin{aligned}
 *  & \underset{x}{\text{minimize}}
 *  & & f(x) &&&& f : \mathbb{R}^n \rightarrow \mathbb{R} \\
 *  & \text{subject to}
 *  & & \underline{x} \le x \le \overline{x} \\
 *  &&& \underline{z} \le g(x) \le \overline{z} &&&& g :
 *  \mathbb{R}^n \rightarrow \mathbb{R}^m
 * \end{aligned} @f]
 */
struct Problem {
    /// Number of decision variables, dimension of x
    unsigned int n;
    /// Number of constraints, dimension of g(x) and z
    unsigned int m;
    /// Constraints of the decision variables, @f$ x \in C @f$
    Box C{vec::Constant(n, +inf), vec::Constant(n, -inf)};
    /// Other constraints, @f$ g(x) \in D @f$
    Box D{vec::Constant(m, +inf), vec::Constant(m, -inf)};

    Problem(unsigned int n, unsigned int m) : n(n), m(m) {}
    Problem(unsigned int n, unsigned int m, Box C, Box D)
        : n(n), m(m), C{std::move(C)}, D{std::move(D)} {}

    Problem(const Problem &) = default;
    Problem &operator=(const Problem &) = default;
    Problem(Problem &&)                 = default;
    Problem &operator=(Problem &&) = default;

    virtual std::unique_ptr<Problem> clone() const &;
    virtual std::unique_ptr<Problem> clone() &&;
    virtual ~Problem() = default;

    /// @name Basic functions
    /// @{

    /// Function that evaluates the cost, @f$ f(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    virtual real_t eval_f(crvec x) const;
    /// Function that evaluates the gradient of the cost, @f$ \nabla f(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] grad_fx
    ///         Gradient of cost function @f$ \nabla f(x) \in \mathbb{R}^n @f$
    virtual void eval_grad_f(crvec x, rvec grad_fx) const;
    /// Function that evaluates the constraints, @f$ g(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [out] gx
    ///         Value of the constraints @f$ g(x) \in \mathbb{R}^m @f$
    virtual void eval_g(crvec x, rvec gx) const;
    /// Function that evaluates the gradient of the constraints times a vector,
    /// @f$ \nabla g(x)\,y @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Vector @f$ y \in \mathbb{R}^m @f$ to multiply the gradient by
    /// @param  [out] grad_gxy
    ///         Gradient of the constraints
    ///         @f$ \nabla g(x)\,y \in \mathbb{R}^n @f$
    virtual void eval_grad_g_prod(crvec x, crvec y, rvec grad_gxy) const;
    /// Function that evaluates the gradient of one specific constraints,
    /// @f$ \nabla g_i(x) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] i
    ///         Which constraint @f$ 0 \le i \lt m @f$
    /// @param  [out] grad_gi
    ///         Gradient of the constraint
    ///         @f$ \nabla g_i(x) \mathbb{R}^n @f$
    virtual void eval_grad_gi(crvec x, unsigned i, rvec grad_gi) const;
    /// Function that evaluates the Hessian of the Lagrangian multiplied by a
    /// vector,
    /// @f$ \nabla_{xx}^2L(x, y)\,v @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Lagrange multipliers @f$ y \in \mathbb{R}^m @f$
    /// @param  [in] v
    ///         Vector to multiply by @f$ v \in \mathbb{R}^n @f$
    /// @param  [out] Hv
    ///         Hessian-vector product
    ///         @f$ \nabla_{xx}^2 L(x, y)\,v \in \mathbb{R}^{n} @f$
    virtual void eval_hess_L_prod(crvec x, crvec y, crvec v, rvec Hv) const;
    /// Function that evaluates the Hessian of the Lagrangian,
    /// @f$ \nabla_{xx}^2L(x, y) @f$
    /// @param  [in] x
    ///         Decision variable @f$ x \in \mathbb{R}^n @f$
    /// @param  [in] y
    ///         Lagrange multipliers @f$ y \in \mathbb{R}^m @f$
    /// @param  [out] H
    ///         Hessian @f$ \nabla_{xx}^2 L(x, y) \in \mathbb{R}^{n\times n} @f$
    virtual void eval_hess_L(crvec x, crvec y, rmat H) const;

    /// @}

    /// @name Combined evaluations
    /// @{

    /// Evaluate both @f$ f(x) @f$ and its gradient, @f$ \nabla f(x) @f$.
    virtual real_t eval_f_grad_f(crvec x, rvec grad_fx) const;
    /// Evaluate both @f$ f(x) @f$ and @f$ g(x) @f$.
    virtual real_t eval_f_g(crvec x, rvec g) const;
    /// Evaluate @f$ f(x) @f$, its gradient @f$ \nabla f(x) @f$ and @f$ g(x) @f$.
    virtual real_t eval_f_grad_f_g(crvec x, rvec grad_fx, rvec g) const;
    /// Evaluate both @f$ \nabla f(x) @f$ and @f$ \nabla g(x)\,y @f$.
    virtual void eval_grad_f_grad_g_prod(crvec x, crvec y, rvec grad_f,
                                         rvec grad_gxy) const;
    /// Evaluate the gradient of the Lagrangian
    /// @f$ \nabla_x L(x, y) = \nabla f(x) + \nabla g(x)\,y @f$
    virtual void eval_grad_L(crvec x, crvec y, rvec grad_L, rvec work_n) const;

    /// @}

    /// @name Augmented Lagrangian
    /// @{

    /// Calculate both ψ(x) and the vector ŷ that can later be used to compute
    /// ∇ψ.
    /// @f[ \psi(x) = f(x) + \tfrac{1}{2}
    ///   \text{dist}_\Sigma^2\left(g(x) + \Sigma^{-1}y,\;D\right) @f]
    /// @f[ \hat y = \Sigma\, \left(g(x) + \Sigma^{-1}y - \Pi_D\left(g(x)
    ///   + \Sigma^{-1}y\right)\right) @f]
    virtual real_t eval_ψ_ŷ(crvec x, ///< [in]  Decision variable @f$ x @f$
                            crvec y, ///< [in]  Lagrange multipliers @f$ y @f$
                            crvec Σ, ///< [in]  Penalty weights @f$ \Sigma @f$
                            rvec ŷ   ///< [out] @f$ \hat y @f$
    ) const;
    /// Calculate ∇ψ(x) using ŷ.
    virtual void
    eval_grad_ψ_from_ŷ(crvec x,     ///< [in]  Decision variable @f$ x @f$
                       crvec ŷ,     ///< [in]  @f$ \hat y @f$
                       rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                       rvec work_n  ///<       Dimension @f$ n @f$
    ) const;
    /// Calculate the gradient ∇ψ(x).
    /// @f[ \nabla \psi(x) = \nabla f(x) + \nabla g(x)\,\hat y(x) @f]
    virtual void eval_grad_ψ(crvec x, ///< [in]  Decision variable @f$ x @f$
                             crvec y, ///< [in]  Lagrange multipliers @f$ y @f$
                             crvec Σ, ///< [in]  Penalty weights @f$ \Sigma @f$
                             rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                             rvec work_n, ///<       Dimension @f$ n @f$
                             rvec work_m  ///<       Dimension @f$ m @f$
    ) const;
    /// Calculate both ψ(x) and its gradient ∇ψ(x).
    /// @f[ \psi(x) = f(x) + \tfrac{1}{2}
    /// \text{dist}_\Sigma^2\left(g(x) + \Sigma^{-1}y,\;D\right) @f]
    /// @f[ \nabla \psi(x) = \nabla f(x) + \nabla g(x)\,\hat y(x) @f]
    virtual real_t
    eval_ψ_grad_ψ(crvec x,     ///< [in]  Decision variable @f$ x @f$
                  crvec y,     ///< [in]  Lagrange multipliers @f$ y @f$
                  crvec Σ,     ///< [in]  Penalty weights @f$ \Sigma @f$
                  rvec grad_ψ, ///< [out] @f$ \nabla \psi(x) @f$
                  rvec work_n, ///<       Dimension @f$ n @f$
                  rvec work_m  ///<       Dimension @f$ m @f$
    ) const;

    /// @}

    /// @name Helpers
    /// @{

    /// Given g(x), compute the intermediate results ŷ and dᵀŷ that can later be
    /// used to compute ψ(x) and ∇ψ(x).
    /// @param[inout]   g_ŷ
    ///                 Input @f$ g(x) @f$, outputs @f$ \hat y @f$
    /// @param[in]      y
    ///                 Lagrange multipliers @f$ y @f$
    /// @param[in]      Σ
    ///                 Penalty weights @f$ \Sigma @f$
    /// @return The inner product @f$ d^\top \hat y @f$
    real_t calc_ŷ_dᵀŷ(rvec g_ŷ, crvec y, crvec Σ) const;

    /// @}
};

struct LambdaProblem : Problem {
    using Problem::Problem;

    std::function<real_t(crvec)> f;
    std::function<void(crvec, rvec)> grad_f;
    std::function<void(crvec, rvec)> g;
    std::function<void(crvec, crvec, rvec)> grad_g_prod;
    std::function<void(crvec, unsigned, rvec)> grad_gi;
    std::function<void(crvec, crvec, crvec, rvec)> hess_L_prod;
    std::function<void(crvec, crvec, rmat)> hess_L;

    real_t eval_f(crvec x) const override { return f(x); }
    void eval_grad_f(crvec x, rvec grad_fx) const override {
        return grad_f(x, grad_fx);
    }
    void eval_g(crvec x, rvec gx) const override { return g(x, gx); }
    void eval_grad_g_prod(crvec x, crvec y, rvec grad_gxy) const override {
        return grad_g_prod(x, y, grad_gxy);
    }
    void eval_grad_gi(crvec x, unsigned int i, rvec gr_gi) const override {
        return grad_gi(x, i, gr_gi);
    }
    void eval_hess_L_prod(crvec x, crvec y, crvec v, rvec Hv) const override {
        return hess_L_prod(x, y, v, Hv);
    }
    void eval_hess_L(crvec x, crvec y, rmat H) const override {
        return hess_L(x, y, H);
    }

    LambdaProblem(const LambdaProblem &) = default;
    LambdaProblem &operator=(const LambdaProblem &) = default;
    LambdaProblem(LambdaProblem &&)                 = default;
    LambdaProblem &operator=(LambdaProblem &&) = default;

    std::unique_ptr<Problem> clone() const & override {
        return std::unique_ptr<LambdaProblem>(new LambdaProblem(*this));
    }
    std::unique_ptr<Problem> clone() && override {
        return std::unique_ptr<LambdaProblem>(
            new LambdaProblem(std::move(*this)));
    }
};

class ProblemWithParam : public Problem {
  public:
    using Problem::Problem;
    ProblemWithParam(unsigned n, unsigned m, unsigned p)
        : Problem{n, m}, param{vec::Constant(p, NaN)} {}

    void set_param(vec p) {
        assert(p.size() == param.size());
        param = std::move(p);
    }
    vec &get_param() { return param; }
    const vec &get_param() const { return param; }

    vec param;

    ProblemWithParam(const ProblemWithParam &) = default;
    ProblemWithParam &operator=(const ProblemWithParam &) = default;
    ProblemWithParam(ProblemWithParam &&)                 = default;
    ProblemWithParam &operator=(ProblemWithParam &&) = default;

    std::unique_ptr<Problem> clone() const & override {
        return std::unique_ptr<ProblemWithParam>(new ProblemWithParam(*this));
    }
    std::unique_ptr<Problem> clone() && override {
        return std::unique_ptr<ProblemWithParam>(
            new ProblemWithParam(std::move(*this)));
    }
};

struct LambdaProblemWithParam : ProblemWithParam {
    using ProblemWithParam::ProblemWithParam;

    std::function<real_t(crvec, crvec)> f;
    std::function<void(crvec, crvec, rvec)> grad_f;
    std::function<void(crvec, crvec, rvec)> g;
    std::function<void(crvec, crvec, crvec, rvec)> grad_g_prod;
    std::function<void(crvec, crvec, unsigned, rvec)> grad_gi;
    std::function<void(crvec, crvec, crvec, crvec, rvec)> hess_L_prod;
    std::function<void(crvec, crvec, crvec, rmat)> hess_L;

    real_t eval_f(crvec x) const override { return f(x, param); }
    void eval_grad_f(crvec x, rvec grad_fx) const override {
        return grad_f(x, param, grad_fx);
    }
    void eval_g(crvec x, rvec gx) const override { return g(x, param, gx); }
    void eval_grad_g_prod(crvec x, crvec y, rvec grad_gxy) const override {
        return grad_g_prod(x, param, y, grad_gxy);
    }
    void eval_grad_gi(crvec x, unsigned int i, rvec gr_gi) const override {
        return grad_gi(x, param, i, gr_gi);
    }
    void eval_hess_L_prod(crvec x, crvec y, crvec v, rvec Hv) const override {
        return hess_L_prod(x, param, y, v, Hv);
    }
    void eval_hess_L(crvec x, crvec y, rmat H) const override {
        return hess_L(x, param, y, H);
    }

    LambdaProblemWithParam(const LambdaProblemWithParam &) = default;
    LambdaProblemWithParam &operator=(const LambdaProblemWithParam &) = default;
    LambdaProblemWithParam(LambdaProblemWithParam &&)                 = default;
    LambdaProblemWithParam &operator=(LambdaProblemWithParam &&) = default;

    std::unique_ptr<Problem> clone() const & override {
        return std::unique_ptr<LambdaProblemWithParam>(
            new LambdaProblemWithParam(*this));
    }
    std::unique_ptr<Problem> clone() && override {
        return std::unique_ptr<LambdaProblemWithParam>(
            new LambdaProblemWithParam(std::move(*this)));
    }
};

struct EvalCounter {
    unsigned f{};
    unsigned grad_f{};
    unsigned f_grad_f{};
    unsigned f_g{};
    unsigned f_grad_f_g{};
    unsigned grad_f_grad_g_prod{};
    unsigned g{};
    unsigned grad_g_prod{};
    unsigned grad_gi{};
    unsigned grad_L{};
    unsigned hess_L_prod{};
    unsigned hess_L{};
    unsigned ψ{};
    unsigned grad_ψ{};
    unsigned grad_ψ_from_ŷ{};
    unsigned ψ_grad_ψ{};

    struct EvalTimer {
        std::chrono::nanoseconds f{};
        std::chrono::nanoseconds grad_f{};
        std::chrono::nanoseconds f_grad_f{};
        std::chrono::nanoseconds f_g{};
        std::chrono::nanoseconds f_grad_f_g{};
        std::chrono::nanoseconds grad_f_grad_g_prod{};
        std::chrono::nanoseconds g{};
        std::chrono::nanoseconds grad_g_prod{};
        std::chrono::nanoseconds grad_gi{};
        std::chrono::nanoseconds grad_L{};
        std::chrono::nanoseconds hess_L_prod{};
        std::chrono::nanoseconds hess_L{};
        std::chrono::nanoseconds ψ{};
        std::chrono::nanoseconds grad_ψ{};
        std::chrono::nanoseconds grad_ψ_from_ŷ{};
        std::chrono::nanoseconds ψ_grad_ψ{};
    } time;

    void reset() { *this = {}; }
};

std::ostream &operator<<(std::ostream &, const EvalCounter &);

inline EvalCounter::EvalTimer &operator+=(EvalCounter::EvalTimer &a,
                                          const EvalCounter::EvalTimer &b) {
    a.f += b.f;
    a.grad_f += b.grad_f;
    a.f_grad_f += b.f_grad_f;
    a.f_g += b.f_g;
    a.f_grad_f_g += b.f_grad_f_g;
    a.grad_f_grad_g_prod += b.grad_f_grad_g_prod;
    a.g += b.g;
    a.grad_g_prod += b.grad_g_prod;
    a.grad_gi += b.grad_gi;
    a.grad_L += b.grad_L;
    a.hess_L_prod += b.hess_L_prod;
    a.hess_L += b.hess_L;
    a.ψ += b.ψ;
    a.grad_ψ += b.grad_ψ;
    a.grad_ψ_from_ŷ += b.grad_ψ_from_ŷ;
    a.ψ_grad_ψ += b.ψ_grad_ψ;
    return a;
}

inline EvalCounter &operator+=(EvalCounter &a, const EvalCounter &b) {
    a.f += b.f;
    a.grad_f += b.grad_f;
    a.f_grad_f += b.f_grad_f;
    a.f_g += b.f_g;
    a.f_grad_f_g += b.f_grad_f_g;
    a.grad_f_grad_g_prod += b.grad_f_grad_g_prod;
    a.g += b.g;
    a.grad_g_prod += b.grad_g_prod;
    a.grad_gi += b.grad_gi;
    a.grad_L += b.grad_L;
    a.hess_L_prod += b.hess_L_prod;
    a.hess_L += b.hess_L;
    a.ψ += b.ψ;
    a.grad_ψ += b.grad_ψ;
    a.grad_ψ_from_ŷ += b.grad_ψ_from_ŷ;
    a.ψ_grad_ψ += b.ψ_grad_ψ;
    a.time += b.time;
    return a;
}

inline EvalCounter operator+(EvalCounter a, const EvalCounter &b) {
    return a += b;
}

template <class ProblemT>
class ProblemWithCounters : public ProblemT {
  public:
    static_assert(std::is_base_of_v<Problem, ProblemT>);
    using ProblemT::ProblemT;

    ProblemWithCounters(const ProblemT &p) : ProblemT{p} {}
    ProblemWithCounters(ProblemT &&p) : ProblemT{std::move(p)} {}

    std::unique_ptr<Problem> clone() const & override {
        if constexpr (std::is_copy_constructible_v<ProblemT>) {
            return std::unique_ptr<ProblemWithCounters>(
                new ProblemWithCounters(*this));
        } else {
            throw std::logic_error("ProblemWithCounters<" +
                                   std::string(typeid(ProblemT).name()) +
                                   "> cannot be cloned");
        }
    }
    std::unique_ptr<Problem> clone() && override {
        if constexpr (std::is_copy_constructible_v<ProblemT>) {
            return std::unique_ptr<ProblemWithCounters>(
                new ProblemWithCounters(std::move(*this)));
        } else {
            throw std::logic_error("ProblemWithCounters<" +
                                   std::string(typeid(ProblemT).name()) +
                                   "> cannot be cloned");
        }
    }

    real_t eval_f(crvec x) const override;
    void eval_grad_f(crvec x, rvec grad_fx) const override;
    void eval_g(crvec x, rvec gx) const override;
    void eval_grad_g_prod(crvec x, crvec y, rvec grad_gxy) const override;
    void eval_grad_gi(crvec x, unsigned i, rvec grad_gi) const override;
    void eval_hess_L_prod(crvec x, crvec y, crvec v, rvec Hv) const override;
    void eval_hess_L(crvec x, crvec y, rmat H) const override;

    real_t eval_f_grad_f(crvec x, rvec grad_fx) const override;
    real_t eval_f_g(crvec x, rvec g) const override;
    real_t eval_f_grad_f_g(crvec x, rvec grad_fx, rvec g) const override;
    void eval_grad_f_grad_g_prod(crvec x, crvec y, rvec grad_f,
                                 rvec grad_gxy) const override;
    void eval_grad_L(crvec x, crvec y, rvec grad_L, rvec work_n) const override;

    real_t eval_ψ_ŷ(crvec x, crvec y, crvec Σ, rvec ŷ) const override;
    void eval_grad_ψ_from_ŷ(crvec x, crvec ŷ, rvec grad_ψ,
                            rvec work_n) const override;
    void eval_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ, rvec work_n,
                     rvec work_m) const override;
    real_t eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ, rvec work_n,
                         rvec work_m) const override;

    mutable EvalCounter evaluations;

  private:
    template <class TimeT, class FunT>
    static auto timed(TimeT &time, const FunT &f) -> decltype(f()) {
        if constexpr (std::is_same_v<decltype(f()), void>) {
            auto t0 = std::chrono::steady_clock::now();
            f();
            auto t1 = std::chrono::steady_clock::now();
            time += t1 - t0;
        } else {
            auto t0  = std::chrono::steady_clock::now();
            auto res = f();
            auto t1  = std::chrono::steady_clock::now();
            time += t1 - t0;
            return res;
        }
    }
};

#ifndef DOXYGEN
template <class ProblemT>
ProblemWithCounters(ProblemT &) -> ProblemWithCounters<ProblemT>;
template <class ProblemT>
ProblemWithCounters(const ProblemT &) -> ProblemWithCounters<ProblemT>;
template <class ProblemT>
ProblemWithCounters(ProblemT &&) -> ProblemWithCounters<ProblemT>;
#endif

template <class ProblemT>
real_t ProblemWithCounters<ProblemT>::eval_f(crvec x) const {
    ++evaluations.f;
    return timed(evaluations.time.f, [&] { return ProblemT::eval_f(x); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_f(crvec x, rvec grad_fx) const {
    ++evaluations.grad_f;
    return timed(evaluations.time.grad_f,
                 [&] { return ProblemT::eval_grad_f(x, grad_fx); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_g(crvec x, rvec gx) const {
    ++evaluations.g;
    return timed(evaluations.time.g, [&] { return ProblemT::eval_g(x, gx); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_g_prod(crvec x, crvec y,
                                                     rvec grad_gxy) const {
    ++evaluations.grad_g_prod;
    return timed(evaluations.time.grad_g_prod,
                 [&] { return ProblemT::eval_grad_g_prod(x, y, grad_gxy); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_gi(crvec x, unsigned i,
                                                 rvec grad_gi) const {
    ++evaluations.grad_gi;
    return timed(evaluations.time.grad_gi,
                 [&] { return ProblemT::eval_grad_gi(x, i, grad_gi); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_hess_L_prod(crvec x, crvec y, crvec v,
                                                     rvec Hv) const {
    ++evaluations.hess_L_prod;
    return timed(evaluations.time.hess_L_prod,
                 [&] { return ProblemT::eval_hess_L_prod(x, y, v, Hv); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_hess_L(crvec x, crvec y,
                                                rmat H) const {
    ++evaluations.hess_L;
    return timed(evaluations.time.hess_L,
                 [&] { return ProblemT::eval_hess_L(x, y, H); });
}

template <class ProblemT>
real_t ProblemWithCounters<ProblemT>::eval_f_grad_f(crvec x,
                                                    rvec grad_fx) const {
    ++evaluations.f_grad_f;
    return timed(evaluations.time.f_grad_f,
                 [&] { return ProblemT::eval_f_grad_f(x, grad_fx); });
}
template <class ProblemT>
real_t ProblemWithCounters<ProblemT>::eval_f_g(crvec x, rvec g) const {
    ++evaluations.f_g;
    return timed(evaluations.time.f_g,
                 [&] { return ProblemT::eval_f_g(x, g); });
}
template <class ProblemT>
real_t ProblemWithCounters<ProblemT>::eval_f_grad_f_g(crvec x, rvec grad_fx,
                                                      rvec g) const {
    ++evaluations.f_grad_f_g;
    return timed(evaluations.time.f_grad_f_g,
                 [&] { return ProblemT::eval_f_grad_f_g(x, grad_fx, g); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_f_grad_g_prod(
    crvec x, crvec y, rvec grad_f, rvec grad_gxy) const {
    ++evaluations.grad_f_grad_g_prod;
    return timed(evaluations.time.grad_f_grad_g_prod, [&] {
        return ProblemT::eval_grad_f_grad_g_prod(x, y, grad_f, grad_gxy);
    });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_L(crvec x, crvec y, rvec grad_L,
                                                rvec work_n) const {
    ++evaluations.grad_L;
    return timed(evaluations.time.grad_L,
                 [&] { return ProblemT::eval_grad_L(x, y, grad_L, work_n); });
}
template <class ProblemT>
real_t ProblemWithCounters<ProblemT>::eval_ψ_ŷ(crvec x, crvec y, crvec Σ,
                                               rvec ŷ) const {
    ++evaluations.ψ;
    return timed(evaluations.time.ψ,
                 [&] { return ProblemT::eval_ψ_ŷ(x, y, Σ, ŷ); });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_ψ_from_ŷ(crvec x, crvec ŷ,
                                                       rvec grad_ψ,
                                                       rvec work_n) const {
    ++evaluations.grad_ψ_from_ŷ;
    return timed(evaluations.time.grad_ψ_from_ŷ, [&] {
        return ProblemT::eval_grad_ψ_from_ŷ(x, ŷ, grad_ψ, work_n);
    });
}
template <class ProblemT>
void ProblemWithCounters<ProblemT>::eval_grad_ψ(crvec x, crvec y, crvec Σ,
                                                rvec grad_ψ, rvec work_n,
                                                rvec work_m) const {
    ++evaluations.grad_ψ;
    return timed(evaluations.time.grad_ψ, [&] {
        return ProblemT::eval_grad_ψ(x, y, Σ, grad_ψ, work_n, work_m);
    });
}
template <class ProblemT>
real_t ProblemWithCounters<ProblemT>::eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ,
                                                    rvec grad_ψ, rvec work_n,
                                                    rvec work_m) const {
    ++evaluations.ψ_grad_ψ;
    return timed(evaluations.time.ψ_grad_ψ, [&] {
        return ProblemT::eval_ψ_grad_ψ(x, y, Σ, grad_ψ, work_n, work_m);
    });
}

} // namespace alpaqa
