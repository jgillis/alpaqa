#include <alpaqa/interop/casadi/CasADiFunctionWrapper.hpp>
#include <alpaqa/interop/casadi/CasADiLoader.hpp>

#include <casadi/core/external.hpp>

#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace alpaqa {

template <class F>
auto wrap_load(const std::string &so_name, const char *name, F f) {
    try {
        return f();
    } catch (const std::invalid_argument &e) {
        throw std::invalid_argument("Unable to load function '" + so_name +
                                    ":" + name + "': " + e.what());
    }
}

template <class T, class... Args>
auto wrapped_load(const std::string &so_name, const char *name,
                  Args &&...args) {
    return wrap_load(so_name, name, [&] {
        return T(casadi::external(name, so_name), std::forward<Args>(args)...);
    });
}

using dim                  = std::pair<casadi_int, casadi_int>;
constexpr static auto dims = [](auto... a) {
    if constexpr ((... && std::is_constructible_v<dim, decltype(a)>))
        return std::array{a...};
    else
        return std::array{std::make_pair<casadi_int, casadi_int>(a, 1)...};
};

struct CasADiFunctionsWithParam {
    constexpr static bool WithParam = true;
    CasADiFunctionEvaluator<1 + WithParam, 1> f;
    CasADiFunctionEvaluator<1 + WithParam, 1> g;
    CasADiFunctionEvaluator<5 + WithParam, 1> grad_ψ;
    CasADiFunctionEvaluator<5 + WithParam, 2> ψ_grad_ψ;
    CasADiFunctionEvaluator<2 + WithParam, 1> grad_L;
    CasADiFunctionEvaluator<5 + WithParam, 2> ψ;
    std::optional<CasADiFunctionEvaluator<3 + WithParam, 1>> hess_L_prod;
    std::optional<CasADiFunctionEvaluator<2 + WithParam, 1>> hess_L;
};

CasADiProblem::CasADiProblem(const std::string &so_name, unsigned n, unsigned m,
                             unsigned p, bool second_order)
    : ProblemWithParam{n, m} {

    auto load_g_unknown_dims = [&] {
        CasADiFunctionEvaluator<2, 1> g{casadi::external("g", so_name)};
        if (g.fun.size2_in(0) != 1)
            throw std::invalid_argument(
                "First input argument should be a column vector.");
        if (g.fun.size2_in(1) != 1)
            throw std::invalid_argument(
                "Second input argument should be a column vector.");
        if (g.fun.size2_out(0) != 1)
            throw std::invalid_argument(
                "First output argument should be a column vector.");
        if (n == 0)
            n = g.fun.size1_in(0);
        if (m == 0)
            m = g.fun.size1_out(0);
        if (p == 0)
            p = g.fun.size1_in(1);
        g.validate_dimensions({dim(n, 1), dim(p, 1)}, {dim(m, 1)});
        return g;
    };

    auto load_g_known_dims = [&] {
        CasADiFunctionEvaluator<2, 1> g{casadi::external("g", so_name),
                                        {dim(n, 1), dim(p, 1)},
                                        {dim(m, 1)}};
        return g;
    };

    CasADiFunctionEvaluator<2, 1> g =
        (n == 0 || m == 0 || p == 0)
            // If not all dimensions are specified, load the function "g" to
            // determine the missing dimensions.
            ? wrap_load(so_name, "g", load_g_unknown_dims)
            // Otherwise, load the function "g" and compare its dimensions to
            // the dimensions specified by the user.
            : wrap_load(so_name, "g", load_g_known_dims);

    this->n     = n;
    this->m     = m;
    this->param = vec::Constant(p, NaN);

    impl = std::make_unique<CasADiFunctionsWithParam>(CasADiFunctionsWithParam{
        wrapped_load<CasADiFunctionEvaluator<2, 1>>(so_name, "f", dims(n, p),
                                                    dims(1)),
        std::move(g),
        wrapped_load<CasADiFunctionEvaluator<6, 1>>(
            so_name, "grad_psi", dims(n, p, m, m, m, m), dims(n)),
        wrapped_load<CasADiFunctionEvaluator<6, 2>>(
            so_name, "psi_grad_psi", dims(n, p, m, m, m, m), dims(1, n)),
        wrapped_load<CasADiFunctionEvaluator<3, 1>>(so_name, "grad_L",
                                                    dims(n, p, m), dims(n)),
        wrapped_load<CasADiFunctionEvaluator<6, 2>>(
            so_name, "psi", dims(n, p, m, m, m, m), dims(1, m)),
        {},
        {},
    });
    if (second_order) {
        impl->hess_L_prod = wrapped_load<CasADiFunctionEvaluator<4, 1>>( //
            so_name, "hess_L_prod", dims(n, p, m, n), dims(n));
        impl->hess_L      = wrapped_load<CasADiFunctionEvaluator<3, 1>>( //
            so_name, "hess_L", dims(n, p, m), dims(dim(n, n)));
    }
}

CasADiProblem::CasADiProblem(CasADiProblem &&) = default;
CasADiProblem &CasADiProblem::operator=(CasADiProblem &&) = default;

CasADiProblem::~CasADiProblem() = default;

real_t CasADiProblem::eval_f(crvec x) const {
    real_t f;
    impl->f({x.data()}, {&f});
    return f;
}

void CasADiProblem::eval_g(crvec x, rvec g) const {
    impl->g({x.data(), param.data()}, {g.data()});
}

void CasADiProblem::eval_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ, rvec,
                                rvec) const {
#if 0
    impl->grad_ψ({x.data(), param.data(), y.data(), Σ.data(),
                    D.lowerbound.data(), D.upperbound.data()},
                   {grad_ψ.data()});
#else
    // This seems to be faster than having a specialized function. Possibly 
    // cache-related?
    real_t ψ;
    impl->ψ_grad_ψ({x.data(), param.data(), y.data(), Σ.data(),
                    D.lowerbound.data(), D.upperbound.data()},
                   {&ψ, grad_ψ.data()});
#endif
}

real_t CasADiProblem::eval_ψ_grad_ψ(crvec x, crvec y, crvec Σ, rvec grad_ψ,
                                    rvec, rvec) const {
    real_t ψ;
    impl->ψ_grad_ψ({x.data(), param.data(), y.data(), Σ.data(),
                    D.lowerbound.data(), D.upperbound.data()},
                   {&ψ, grad_ψ.data()});
    return ψ;
}

void CasADiProblem::eval_grad_L(crvec x, crvec y, rvec grad_L, rvec) const {
    impl->grad_L({x.data(), param.data(), y.data()}, {grad_L.data()});
}

real_t CasADiProblem::eval_ψ_ŷ(crvec x, crvec y, crvec Σ, rvec ŷ) const {
    real_t ψ;
    impl->ψ({x.data(), param.data(), y.data(), Σ.data(), D.lowerbound.data(),
             D.upperbound.data()},
            {&ψ, ŷ.data()});
    return ψ;
}

CasADiProblem load_CasADi_problem_with_param(const std::string &filename,
                                             unsigned n, unsigned m, unsigned p,
                                             bool second_order) {
    return {filename, n, m, p, second_order};
}

} // namespace alpaqa