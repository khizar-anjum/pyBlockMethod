#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>

namespace pyblockgrid {

// Common type aliases
using Vec2d = Eigen::Vector2d;
using MatX2d = Eigen::Matrix<double, Eigen::Dynamic, 2>;
using VecXd = Eigen::VectorXd;
using VecXi = Eigen::VectorXi;
using MatXd = Eigen::MatrixXd;
using BoolVec = Eigen::Matrix<bool, Eigen::Dynamic, 1>;
using BoolMat = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>;

// Grid with validity mask: valid(i,j)==true means point is inside domain
struct MaskedGrid {
    MatXd data;
    BoolMat valid;

    MaskedGrid() = default;
    MaskedGrid(int rows, int cols)
        : data(MatXd::Zero(rows, cols)),
          valid(BoolMat::Constant(rows, cols, false)) {}
};

// 2D masked grid (ny x nx x 2) stored as two separate matrices
struct MaskedGrid2D {
    MatXd x, y;
    BoolMat valid;

    MaskedGrid2D() = default;
    MaskedGrid2D(int rows, int cols)
        : x(MatXd::Zero(rows, cols)),
          y(MatXd::Zero(rows, cols)),
          valid(BoolMat::Constant(rows, cols, false)) {}
};

enum class BlockKind : int { First = 1, Second = 2, Third = 3 };

enum class BoundaryType : int { Main = 0, Hole = 1 };

} // namespace pyblockgrid
