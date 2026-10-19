#include "pyblockgrid/carrier_functions.hpp"
#include <cmath>

namespace pyblockgrid {

namespace {
// Volkov's vartheta_{jk} (Eq. 3.7): Neumann-edge basis for a mixed first-kind
// block; absorbs a flux datum r^k on the edge at theta=0.
double vartheta(double r, double th, double k, double alpha, double tol) {
    double kp1 = k + 1;
    double c = std::cos(kp1 * alpha * M_PI);
    double rk = std::pow(r, kp1);
    if (std::abs(c) < tol) {
        double s = std::sin(kp1 * alpha * M_PI);
        return rk * (th * std::cos(kp1 * th) + std::log(r) * std::sin(kp1 * th))
               / (kp1 * alpha * M_PI * s);
    }
    return -rk * std::sin(kp1 * th) / (kp1 * c);
}

// Volkov's chi_{jk} (Eq. 3.8): Dirichlet-edge basis for a mixed first-kind
// block; reproduces a value datum r^k on the Dirichlet edge (theta=0) while
// carrying zero flux on the Neumann edge (theta=alpha*pi). Zero far-edge flux
// forces the cos(k(theta-alpha*pi)) phase; plain cos(k*theta) only works for
// the resonant k where sin(k*alpha*pi)=0.
double chi(double r, double th, double k, double alpha, double tol) {
    double c = std::cos(k * alpha * M_PI);
    double rk = std::pow(r, k);
    if (std::abs(c) < tol) {
        return rk * (std::cos(k * th) - (th * std::cos(k * th)
                     + std::log(r) * std::sin(k * th)) / (alpha * M_PI));
    }
    return rk * std::cos(k * (th - alpha * M_PI)) / c;
}

// Volkov's eta_{jk} (Eq. 3.5): Neumann-edge basis for a Neumann-Neumann
// first-kind block; absorbs a flux datum r^k on the edge at theta=alpha*pi.
// (Distinct from vartheta, the Neumann basis for a *mixed* block.)
double eta(double r, double th, double k, double alpha, double tol) {
    double kp1 = k + 1;
    double s = std::sin(kp1 * alpha * M_PI);
    double rk = std::pow(r, kp1);
    if (std::abs(s) < tol) {
        double c = std::cos(kp1 * alpha * M_PI);
        return rk * (std::log(r) * std::cos(kp1 * th) - th * std::sin(kp1 * th))
               / (kp1 * alpha * M_PI * c);
    }
    return rk * std::cos(kp1 * th) / (kp1 * s);
}
}  // namespace

double CarrierFunctions::calculate(int block_kind, int boundary_id,
                                   double r, double theta,
                                   const VecXd& k, const VecXd& a, const VecXd& b,
                                   double alpha) const {
    if (block_kind == 1) {
        switch (boundary_id) {
            case 0: return first_kind_nn(r, theta, k, a, b, alpha);
            case 1: return first_kind_nd(r, theta, k, a, b, alpha);
            case 2: return first_kind_dn(r, theta, k, a, b, alpha);
            case 3: return first_kind_dd(r, theta, k, a, b, alpha);
        }
    } else if (block_kind == 2) {
        return second_kind(boundary_id, r, theta, k, a);
    }
    // Third kind: return 0
    return 0.0;
}

// Neumann-Neumann (Volkov Eq. 3.4): both edges Neumann -> both series via eta.
// eta imposes its flux at theta=alpha*pi, so the incoming a-series uses eta(r,theta)
// and the outgoing b-series uses eta(r, alpha*pi - theta).
double CarrierFunctions::first_kind_nn(double r, double theta, const VecXd& k,
                                        const VecXd& a, const VecXd& b,
                                        double alpha) const {
    double sum = 0;
    for (int i = 0; i < static_cast<int>(k.size()); ++i) {
        sum += a(i) * eta(r, theta, k(i), alpha, tol_)
             + b(i) * eta(r, alpha * M_PI - theta, k(i), alpha, tol_);
    }
    return sum;
}

// Neumann-Dirichlet (Volkov Eq. 3.6): edge j-1 (incoming, theta=alpha*pi) is
// Neumann -> a-series via vartheta; edge j (outgoing, theta=0) is Dirichlet ->
// b-series via chi.
double CarrierFunctions::first_kind_nd(double r, double theta, const VecXd& k,
                                        const VecXd& a, const VecXd& b,
                                        double alpha) const {
    double sum = 0;
    for (int i = 0; i < static_cast<int>(k.size()); ++i) {
        sum += a(i) * vartheta(r, theta, k(i), alpha, tol_)
             + b(i) * chi(r, theta, k(i), alpha, tol_);
    }
    return sum;
}

// Dirichlet-Neumann (Volkov Eq. 3.9): edge j-1 (incoming, theta=alpha*pi) is
// Dirichlet -> a-series via chi; edge j (outgoing, theta=0) is Neumann ->
// b-series via vartheta.
double CarrierFunctions::first_kind_dn(double r, double theta, const VecXd& k,
                                        const VecXd& a, const VecXd& b,
                                        double alpha) const {
    double sum = 0;
    for (int i = 0; i < static_cast<int>(k.size()); ++i) {
        sum += a(i) * chi(r, alpha * M_PI - theta, k(i), alpha, tol_)
             + b(i) * vartheta(r, alpha * M_PI - theta, k(i), alpha, tol_);
    }
    return sum;
}

double CarrierFunctions::first_kind_dd(double r, double theta, const VecXd& k,
                                        const VecXd& a, const VecXd& b,
                                        double alpha) const {
    int n = static_cast<int>(k.size());
    double sum1 = 0, sum2 = 0;

    for (int i = 0; i < n; ++i) {
        double ki = k(i);
        double sin_val = std::sin(ki * alpha * M_PI);
        bool cond = std::abs(sin_val) < tol_;

        double zeta_jk, zeta_1k;
        if (!cond) {
            zeta_jk = std::sin(ki * theta);
            zeta_1k = std::sin(ki * alpha * M_PI);
        } else {
            zeta_jk = theta * std::cos(ki * theta) + std::log(r) * std::sin(ki * theta);
            zeta_1k = alpha * M_PI * std::cos(ki * alpha * M_PI)
                     + std::log(1.0) * std::sin(ki * alpha * M_PI);
        }

        double r_pow = std::pow(r, ki);
        sum1 += (a(i) / zeta_1k) * r_pow * zeta_jk;
        sum2 += b(i) * r_pow *
                (std::cos(ki * theta) - (std::cos(ki * alpha * M_PI) / zeta_1k) * zeta_jk);
    }
    return sum1 + sum2;
}

double CarrierFunctions::second_kind(int boundary_id, double r, double theta,
                                      const VecXd& k, const VecXd& a) const {
    int nu_jq = boundary_id / 2;  // integer division
    int n = static_cast<int>(k.size());
    double sum = 0;
    for (int i = 0; i < n; ++i) {
        double ki = k(i);
        double r_pow = std::pow(r, ki);
        double term;
        if (nu_jq == 1) {
            term = std::cos(ki * theta);
        } else {
            term = r * std::sin((ki + 1) * theta) / (ki + 1);
        }
        sum += a(i) * r_pow * term;
    }
    return sum;
}

} // namespace pyblockgrid
