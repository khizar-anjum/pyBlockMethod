#pragma once

#include "types.hpp"
#include <cmath>

namespace pyblockgrid {

class PoissonKernels {
public:
    // Fundamental Poisson kernel for unit disk
    static double main_kernel(double r1, double theta1, double eta1) {
        double r1sq = r1 * r1;
        return (1.0 - r1sq) / (2.0 * M_PI * (1.0 - 2.0 * r1 * std::cos(theta1 - eta1) + r1sq));
    }

    double calculate(int block_kind, int nu_i, int nu_j,
                     double r, double r0, double theta,
                     double eta, double alpha) const;

    // Vectorized version for boundary estimation
    void calculate_vector(int block_kind, int nu_i, int nu_j,
                          double r, double r0, double theta,
                          const VecXd& eta, double alpha,
                          VecXd& result) const;

private:
    double first_kind(int nu_i, int nu_j, double r, double r0,
                      double theta, double eta, double alpha) const;
    double second_kind(int nu_i, double r, double r0,
                       double theta, double eta) const;
    double third_kind(double r, double r0, double theta, double eta) const;
};

} // namespace pyblockgrid
