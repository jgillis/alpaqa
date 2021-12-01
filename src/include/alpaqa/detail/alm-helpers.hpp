#pragma once

#include <alpaqa/decl/alm.hpp>
#include <stdexcept>

namespace alpaqa::detail {

inline void project_y(rvec y,     // inout
                      crvec z_lb, // in
                      crvec z_ub, // in
                      real_t M    // in
) {
    // TODO: Handle NaN correctly
    auto max_lb = [M](real_t y, real_t z_lb) {
        real_t y_lb = z_lb == -inf ? 0 : -M;
        return std::max(y, y_lb);
    };
    auto min_ub = [M](real_t y, real_t z_ub) {
        real_t y_ub = z_ub == inf ? 0 : M;
        return std::min(y, y_ub);
    };
    y = y.binaryExpr(z_lb, max_lb).binaryExpr(z_ub, min_ub);
}

inline void update_penalty_weights(const ALMParams &params, real_t Δ,
                                   bool first_iter, rvec e, rvec old_e,
                                   real_t norm_e, real_t old_norm_e,
                                   crvec Σ_old, rvec Σ) {
    if (norm_e <= params.δ) {
        Σ = Σ_old;
        return;
    }
    if (params.single_penalty_factor) {
        if (first_iter || norm_e > params.θ * old_norm_e) {
            real_t new_Σ = std::fmin(params.Σ_max, Δ * Σ_old(0));
            Σ.setConstant(new_Σ);
        } else {
            Σ = Σ_old;
        }
    } else {
        for (Eigen::Index i = 0; i < e.rows(); ++i) {
            if (first_iter || std::abs(e(i)) > params.θ * std::abs(old_e(i))) {
                Σ(i) = std::fmin(params.Σ_max,
                                 std::fmax(Δ * std::abs(e(i)) / norm_e, 1) *
                                     Σ_old(i));
            } else {
                Σ(i) = Σ_old(i);
            }
        }
    }
}

inline void initialize_penalty(const Problem &p, const ALMParams &params,
                               crvec x0, rvec Σ) {
    real_t f0 = p.eval_f(x0);
    vec g0(p.m);
    p.eval_g(x0, g0);
    // TODO: reuse evaluations of f ang g in PANOC?
    real_t σ = params.σ₀ * std::max(real_t(1), std::abs(f0)) /
               std::max(real_t(1), 0.5 * g0.squaredNorm());
    σ = std::max(σ, params.Σ_min);
    σ = std::min(σ, params.Σ_max);
    Σ.fill(σ);
}

} // namespace alpaqa::detail