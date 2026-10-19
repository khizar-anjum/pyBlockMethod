#pragma once

#include "types.hpp"

namespace pyblockgrid {

class CarrierFunctions {
public:
    explicit CarrierFunctions(double tolerance = 1e-10) : tol_(tolerance) {}

    double calculate(int block_kind, int boundary_id,
                     double r, double theta,
                     const VecXd& k, const VecXd& a, const VecXd& b,
                     double alpha) const;

private:
    double tol_;

    double first_kind_nn(double r, double theta, const VecXd& k,
                         const VecXd& a, const VecXd& b, double alpha) const;
    double first_kind_nd(double r, double theta, const VecXd& k,
                         const VecXd& a, const VecXd& b, double alpha) const;
    double first_kind_dn(double r, double theta, const VecXd& k,
                         const VecXd& a, const VecXd& b, double alpha) const;
    double first_kind_dd(double r, double theta, const VecXd& k,
                         const VecXd& a, const VecXd& b, double alpha) const;
    double second_kind(int boundary_id, double r, double theta,
                       const VecXd& k, const VecXd& a) const;
};

} // namespace pyblockgrid
