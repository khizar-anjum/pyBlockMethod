#pragma once

#include "solution_state.hpp"

namespace pyblockgrid {

class BoundaryEstimator {
public:
    void estimate(SolutionState& state, int max_iter = 100) const;
};

} // namespace pyblockgrid
