/**
 * \file gauleg.hpp
 * \brief This file defines the function to evaluate points and weights for
 *        Gauss Legendre quadrature rule.
 *
 * This File is a part of the 2D-Parametric BEM package
 */
#ifndef GAULEGHPP
#define GAULEGHPP

#include "logweight_quadrature.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <cmath>
#define _USE_MATH_DEFINES

/* @brief Compute Gaussian quadrature nodes and weights for n nodes over
 * interval [a,b] \param[in] a,b Interval [a,b] endpoints \param[in] n Number of
 * quadrature points \param[out] xq,wq Pair of Gauss-Legendre quadrature points
 * and weights
 */
inline std::pair<Eigen::RowVectorXd, Eigen::RowVectorXd>
gauleg(double a, double b, int n, double eps = 1.e-13) {
  int i, j, m;
  double xmid, xlen, p1, p2, p3, dp1, z, z1, wqi;
  Eigen::RowVectorXd xq(n), wq(n);

  m = (n + 1) / 2;
  xmid = 0.5 * (a + b);
  xlen = 0.5 * (b - a);

  // get roots
  for (i = 0; i < m; i++) {

    // i-th root guess
    z = std::cos(M_PI * (i + 1 - 0.25) / (n + 0.5));

    // get i-th root
    do {
      p1 = 1.;
      p2 = 0.;
      for (j = 1; j <= n; j++) {
        p3 = p2;
        p2 = p1;
        p1 = ((2. * j - 1.) * z * p2 - (j - 1.) * p3) / j;
      }
      dp1 = n * (z * p1 - p2) / (z * z - 1.0);
      z1 = z;
      z = z1 - p1 / dp1;
    } while (std::abs(z - z1) > eps);

    // set nodes
    xq(i) = xmid - xlen * z;
    xq(n - 1 - i) = xmid + xlen * z;

    // set weights
    wqi = 2. * xlen / ((1. - z * z) * dp1 * dp1);
    wq(i) = wqi;
    wq(n - 1 - i) = wqi;
  }

  // return
  return std::make_pair(xq, wq);
}

inline QuadRule getGaussQR(unsigned N) {
  // Getting standard Gauss Legendre Quadrature weights and nodes
  Eigen::RowVectorXd weights, points;
  std::tie(points, weights) =
      gauleg(-1, 1, N, std::numeric_limits<double>::epsilon());
  QuadRule gauss;
  gauss.dim = 1;
  gauss.n = N;
  gauss.x = points;
  gauss.w = weights;
  return gauss;
}

#endif
