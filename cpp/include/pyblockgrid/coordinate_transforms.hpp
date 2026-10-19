#pragma once

#include "types.hpp"
#include <cmath>

namespace pyblockgrid {

// Normalize angle to [0, 2pi)
inline double normalize_angle(double theta) {
    theta = std::fmod(theta, 2 * M_PI);
    if (theta < 0) theta += 2 * M_PI;
    return theta;
}

// Convert single point from Cartesian to polar relative to center and reference angle
inline void cartesian_to_polar(const Vec2d& point, const Vec2d& center,
                               double ref_theta, double& r, double& theta) {
    Vec2d d = point - center;
    r = d.norm();
    theta = normalize_angle(std::atan2(d(1), d(0)) - ref_theta);
}

// Convert single point from polar to Cartesian relative to center and reference angle
inline Vec2d polar_to_cartesian(double r, double theta,
                                const Vec2d& center, double ref_theta) {
    double abs_theta = theta + ref_theta;
    return center + Vec2d(r * std::cos(abs_theta), r * std::sin(abs_theta));
}

} // namespace pyblockgrid
