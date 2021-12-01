#include <alpaqa/reference-problems/himmelblau.hpp>

namespace alpaqa {
namespace problems {

LambdaProblem himmelblau_problem() {
    constexpr static auto sq = [](auto x) { return x * x; };
    LambdaProblem prob{2, 0};
    prob.C.lowerbound << -1, -1;
    prob.C.upperbound << 4, 1.8;
    prob.f = [](crvec x) {
        return sq(sq(x(0)) + x(1) - 11) + sq(x(0) + sq(x(1)) - 7);
    };
    prob.grad_f = [](crvec x, rvec g) {
        g(0) = 2 * (2 * x(0) * (sq(x(0)) + x(1) - 11) + x(0) + sq(x(1)) - 7);
        g(1) = 2 * (sq(x(0)) + 2 * x(1) * (x(0) + sq(x(1)) - 7) + x(1) - 11);
    };
    prob.g           = [](crvec, rvec) {};
    prob.grad_g_prod = [](crvec, crvec, rvec grad) { grad.setZero(); };
    prob.grad_gi     = [](crvec, unsigned, rvec grad_gi) { grad_gi.setZero(); };
    prob.hess_L_prod = [](crvec x, crvec, crvec v, rvec Hv) {
        real_t H00 = 4 * (sq(x(0)) + x(1) - 11) + 8 * sq(x(0)) + 2;
        real_t H01 = 4 * x(0) + 4 * x(1);
        real_t H10 = 4 * x(0) + 4 * x(1);
        real_t H11 = 4 * (x(0) + sq(x(1)) - 7) + 8 * sq(x(1)) + 2;
        Hv(0)      = H00 * v(0) + H01 * v(1);
        Hv(1)      = H10 * v(0) + H11 * v(1);
    };
    prob.hess_L = [](crvec x, crvec, rmat H) {
        H(0, 0) = 4 * (sq(x(0)) + x(1) - 11) + 8 * sq(x(0)) + 2;
        H(0, 1) = 4 * x(0) + 4 * x(1);
        H(1, 0) = 4 * x(0) + 4 * x(1);
        H(1, 1) = 4 * (x(0) + sq(x(1)) - 7) + 8 * sq(x(1)) + 2;
    };
    return prob;
}

} // namespace problems
} // namespace alpaqa