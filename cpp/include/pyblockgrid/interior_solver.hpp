#pragma once

#include "solution_state.hpp"
#include "carrier_functions.hpp"
#include "poisson_kernels.hpp"

namespace pyblockgrid {

class InteriorSolver {
public:
    InteriorSolver(const CarrierFunctions& cf, const PoissonKernels& pk)
        : carrier_calc_(cf), poisson_calc_(pk) {}

    void solve(SolutionState& state, double tolerance = 1e-10) const;

private:
    const CarrierFunctions& carrier_calc_;
    const PoissonKernels& poisson_calc_;
};

} // namespace pyblockgrid
