#include <alpaqa/inner/structured-panoc-lbfgs.hpp>

#include <stdexcept>

void alpaqa::StructuredPANOCLBFGSParams::verify() const {
    Lipschitz.verify();
    if (max_iter == 0)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "max_iter cannot be 0");
    if (max_time.count() == 0)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "max_time cannot be 0");
    if (τ_min <= 0 || τ_min >= 1)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "τ_min must be in (0, 1)");
    if (L_min < 0)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "L_min must be positive");
    if (L_max < 0)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "L_max must be positive");
    if (L_max < L_min)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "L_max must be less than L_min");
    if (fpr_shortcut_accept_factor < 0 || fpr_shortcut_accept_factor >= 1)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "fpr_shortcut_accept_factor "
                                    "must be in [0, 1)");
    if (fpr_shortcut_history == 0)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "fpr_shortcut_history cannot be 0");
    if (quadratic_upperbound_tolerance_factor < 0)
        throw std::invalid_argument("StructuredPANOCLBFGSParams::"
                                    "quadratic_upperbound_tolerance_factor "
                                    "must be positive");
}