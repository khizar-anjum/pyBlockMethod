#include "pyblockgrid/poisson_kernels.hpp"
#include <cmath>

namespace pyblockgrid {

double PoissonKernels::calculate(int block_kind, int nu_i, int nu_j,
                                  double r, double r0, double theta,
                                  double eta, double alpha) const {
    if (block_kind == 1)
        return first_kind(nu_i, nu_j, r, r0, theta, eta, alpha);
    else if (block_kind == 2)
        return second_kind(nu_i, r, r0, theta, eta);
    else
        return third_kind(r, r0, theta, eta);
}

void PoissonKernels::calculate_vector(int block_kind, int nu_i, int nu_j,
                                       double r, double r0, double theta,
                                       const VecXd& eta, double alpha,
                                       VecXd& result) const {
    int n = static_cast<int>(eta.size());
    result.resize(n);
    for (int i = 0; i < n; ++i)
        result(i) = calculate(block_kind, nu_i, nu_j, r, r0, theta, eta(i), alpha);
}

double PoissonKernels::first_kind(int nu_i, int nu_j, double r, double r0,
                                   double theta, double eta, double alpha) const {
    double lambda = 1.0 / ((2 - nu_i * nu_j - (1 - nu_i) * (1 - nu_j)) * alpha);
    double r1 = std::pow(r / r0, lambda);
    double lt = lambda * theta;
    double le = lambda * eta;
    double sign = (nu_j % 2 == 0) ? 1.0 : -1.0;

    if (nu_i == nu_j) {
        // (3.17) or (3.12)
        return lambda * (main_kernel(r1, lt, le) + sign * main_kernel(r1, lt, -le));
    } else {
        // (3.17) + (3.13): mixed BC
        // Reflection across the far edge: the third kernel argument is pi - lambda*eta
        // (Volkov 3.13 with the lambda-scaled angle from 3.17), NOT lambda*(pi - eta).
        double lpe = M_PI - le;
        return lambda * (
            main_kernel(r1, lt, le) + sign * main_kernel(r1, lt, -le)
            - sign * (main_kernel(r1, lt, lpe) + sign * main_kernel(r1, lt, -lpe))
        );
    }
}

double PoissonKernels::second_kind(int nu_i, double r, double r0,
                                    double theta, double eta) const {
    double r1 = r / r0;
    double sign = (nu_i % 2 == 0) ? 1.0 : -1.0;
    return main_kernel(r1, theta, eta) + sign * main_kernel(r1, theta, -eta);
}

double PoissonKernels::third_kind(double r, double r0,
                                   double theta, double eta) const {
    return main_kernel(r / r0, theta, eta);
}

} // namespace pyblockgrid
