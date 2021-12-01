#include <alpaqa/interop/casadi/CasADiFunctionWrapper.hpp>
#include <alpaqa/interop/casadi/CasADiLoader.hpp>

#include <casadi/core/external.hpp>

#include <memory>
#include <optional>
#include <stdexcept>

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

constexpr static auto dims = [](auto... a) {
    return std::array<casadi_int, sizeof...(a)>{a...};
};
using dim = std::pair<casadi_int, casadi_int>;

CasADiProblem load_CasADi_problem(const std::string &so_name, unsigned n,
                                  unsigned m, bool second_order) {

    auto load_g_unknown_dims = [&] {
        CasADiFunctionEvaluator<1, 1> g{casadi::external("g", so_name)};
        if (g.fun.size2_in(0) != 1)
            throw std::invalid_argument(
                "First input argument should be a column vector.");
        if (g.fun.size2_out(0) != 1)
            throw std::invalid_argument(
                "First output argument should be a column vector.");
        if (n == 0)
            n = g.fun.size1_in(0);
        if (m == 0)
            m = g.fun.size1_out(0);
        g.validate_dimensions({dim(n, 1)}, {dim(m, 1)});
        return g;
    };

    auto load_g_known_dims = [&] {
        CasADiFunctionEvaluator<1, 1> g{
            casadi::external("g", so_name), {dim(n, 1)}, {dim(m, 1)}};
        return g;
    };

    CasADiFunctionEvaluator<1, 1> g =
        (n == 0 || m == 0)
            // If not all dimensions are specified, load the function "g" to
            //determine the missing dimensions.
            ? wrap_load(so_name, "g", load_g_unknown_dims)
            // Otherwise, load the function "g" and compare its dimensions to
            // the dimensions specified by the user.
            : wrap_load(so_name, "g", load_g_known_dims);

    auto prob = CasADiProblem(n, m);

    prob.f      = wrapped_load<CasADiFun_1Vi1So>(so_name, "f", n);
    prob.grad_f = wrapped_load<CasADiFun_1Vi1Vo>(so_name, "grad_f", n, n);
    prob.g      = CasADiFun_1Vi1Vo(std::move(g));
    auto grad_g =
        wrapped_load<CasADiFun_2Vi1Vo>(so_name, "grad_g", dims(n, m), n);
    prob.grad_g_prod = grad_g;
    vec w            = vec::Zero(m);
    prob.grad_gi     = //
        [grad_g, w](crvec x, unsigned i, rvec g) mutable {
            w(i) = 1;
            grad_g(x, w, g);
            w(i) = 0;
        };
    if (second_order) {
        prob.hess_L =                                              //
            wrapped_load<CasADiFun_2Vi1Mo>(so_name, "hess_L",      //
                                           dims(n, m), dim(n, n)); //
        prob.hess_L_prod =                                         //
            wrapped_load<CasADiFun_3Vi1Vo>(so_name, "hess_L_prod", //
                                           dims(n, m, n), n);      //
    } else {
        prob.hess_L_prod = [](crvec, crvec, crvec, rvec) {
            throw not_implemented_error(
                "CasADiProblem::hess_L_prod not supported");
        };
        prob.hess_L = [](crvec, crvec, rmat) {
            throw not_implemented_error("CasADiProblem::hess_L not supported");
        };
    }
    return prob;
}

struct CasADiFunctions {
    CasADiFun_2Vi1So f;
    CasADiFun_2Vi1Vo grad_f;
    CasADiFun_2Vi1Vo g;
    CasADiFun_3Vi1Vo grad_g_prod;
    std::optional<CasADiFun_3Vi1Mo> hess_L;
    std::optional<CasADiFun_4Vi1Vo> hess_L_prod;
};

CasADiProblemWithParam
load_CasADi_problem_with_param(const std::string &so_name, unsigned n,
                               unsigned m, unsigned p, bool second_order) {

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

    auto prob = CasADiProblemWithParam(n, m);

    prob.f = wrapped_load<CasADiFun_2Vi1So>(so_name, "f", dims(n, p));
    prob.grad_f =
        wrapped_load<CasADiFun_2Vi1Vo>(so_name, "grad_f", dims(n, p), n);
    prob.g = CasADiFun_2Vi1Vo(std::move(g));
    prob.grad_g_prod =
        wrapped_load<CasADiFun_3Vi1Vo>(so_name, "grad_g", dims(n, p, m), n);
    vec w        = vec::Zero(m);
    prob.grad_gi = [grad_g{prob.grad_g_prod}, w](crvec x, crvec param,
                                                 unsigned i, rvec g) mutable {
        w(i) = 1;
        grad_g(x, param, w, g);
        w(i) = 0;
    };
    if (second_order) {
        prob.hess_L_prod = wrapped_load<CasADiFun_4Vi1Vo>( //
            so_name, "hess_L_prod", dims(n, p, m, n), n);
        prob.hess_L      = wrapped_load<CasADiFun_3Vi1Mo>( //
            so_name, "hess_L", dims(n, p, m), dim(n, n));
    } else {
        prob.hess_L_prod = [](crvec, crvec, crvec, crvec, rvec) {
            throw not_implemented_error(
                "CasADiProblemWithParam::hess_L_prod not supported");
        };
        prob.hess_L = [](crvec, crvec, crvec, rmat) {
            throw not_implemented_error(
                "CasADiProblemWithParam::hess_L not supported");
        };
    }
    return prob;
}

} // namespace alpaqa