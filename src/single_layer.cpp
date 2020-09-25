/**
 * \file single_layer.cpp
 * \brief This file declares the functions to evaluate the entries of
 *        Galerkin matrices based on the bilinear form induced by the
 *        Single Layer BIO, using the transformations given in
 *        \f$\ref{ss:quadapprox}\f$ in the Lecture Notes for Advanced Numerical
 *        Methods for CSE.
 *
 * This File is a part of the 2D-Parametric BEM package
 */

#include "single_layer.hpp"

#include <iomanip>
#include <limits>
#include <math.h>
#include <vector>

#include "abstract_bem_space.hpp"
#include "abstract_parametrized_curve.hpp"
#include "discontinuous_space.hpp"
#include "gauleg.hpp"
#include "integral_gauss.hpp"
#include "logweight_quadrature.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

namespace parametricbem2d {
namespace single_layer {
Eigen::MatrixXd InteractionMatrix(const AbstractParametrizedCurve &pi,
                                  const AbstractParametrizedCurve &pi_p,
                                  const AbstractBEMSpace &space,
                                  const QuadRule &GaussQR) {
  double tol = std::numeric_limits<double>::epsilon();
  if (&pi == &pi_p) // Same Panels case
    return ComputeIntegralCoinciding(pi, pi_p, space, GaussQR);

  else if ((pi(1) - pi_p(-1)).norm() / 100. < tol ||
           (pi(-1) - pi_p(1)).norm() / 100. < tol) // Adjacent Panels case
    return ComputeIntegralAdjacent(pi, pi_p, space, GaussQR);

  else // Disjoint panels case
    return ComputeIntegralGeneral(pi, pi_p, space, GaussQR);
}

Eigen::MatrixXd ComputeIntegralCoinciding(const AbstractParametrizedCurve &pi,
                                          const AbstractParametrizedCurve &pi_p,
                                          const AbstractBEMSpace &space,
                                          const QuadRule &GaussQR) {
  unsigned N = GaussQR.n; // Quadrature order for the GaussQR object. Same order
                          // to be used for log weighted quadrature
  int Q = space.getQ(); // No. of Reference Shape Functions in trial/test space
  // Interaction matrix with size Q x Q
  Eigen::MatrixXd interaction_matrix(Q, Q);
  // Computing the (i,j)th matrix entry
  for (int i = 0; i < Q; ++i) {
    for (int j = 0; j < Q; ++j) {
      // Lambda expression for functions F and G in \f$\eqref{eq:Vidp}\f$
      auto F = [&](double t) { // Function associated with panel pi_p
        return space.evaluateShapeFunction(j, t) * pi_p.Derivative(t).norm();
      };

      auto G = [&](double s) { // Function associated with panel pi
        return space.evaluateShapeFunction(i, s) * pi.Derivative(s).norm();
      };

      // Lambda expression for the 1st integrand in \f$\eqref{eq:Isplit}\f$
      auto integrand1 = [&](double s, double t) {
        double sqrt_epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        double s_st;
        if (fabs(s - t) >
            sqrt_epsilon) // Away from singularity for stable evaluation as
                          // mentioned in \f$\eqref{eq:Stab}\f$
          // Simply evaluating the expression
          s_st = (pi(s) - pi_p(t)).squaredNorm() / (s - t) / (s - t);
        else // Near singularity
          // Using analytic limit for s - > t given in \f$\eqref{eq:Sdef}\f$
          s_st = pi.Derivative(0.5 * (t + s)).squaredNorm();
        return 0.5 * log(s_st) * F(t) * G(s);
      };

      double i1 = 0., i2 = 0.; // The two integrals in \f$\eqref{eq:Isplit}\f$

      // Tensor product quadrature for double 1st integral in
      // \f$\eqref{eq:Isplit}\f$
      for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
          i1 += GaussQR.w(i) * GaussQR.w(j) *
                integrand1(GaussQR.x(i), GaussQR.x(j));
        }
      }

      // Lambda expression for inner integrand in transformed coordinates in
      // \f$\eqref{eq:I21}\f$
      auto integrand2 = [&](double w, double z) {
        return F(0.5 * (w - z)) * G(0.5 * (w + z)) +
               F(0.5 * (w + z)) * G(0.5 * (w - z));
      };

      // Integral of integrand2 above w.r.t. w, as a function of z
      auto inner2_z = [&](double z) {
        // Integrand 2 as a function of w only (fixed z), to integrate w.r.t. w
        auto integrand2_w = [&](double w) { return integrand2(w, z); };
        // Computing the integral w.r.t. w
        return ComputeIntegral(integrand2_w, -2 + z, 2 - z, GaussQR);
      };

      // Calculating the second integral by logweighted integral of the function
      // 'inner2_z' above
      // i2 = ComputeLogwtIntegral(inner2_z, 2,GaussQR);
      i2 = ComputeLoogIntegral(inner2_z, 2, GaussQR);
      interaction_matrix(i, j) = -1. / (2. * M_PI) * (i1 + 0.5 * i2);
    }
  }
  return interaction_matrix;
}

Eigen::MatrixXd ComputeIntegralAdjacent(const AbstractParametrizedCurve &pi,
                                        const AbstractParametrizedCurve &pi_p,
                                        const AbstractBEMSpace &space,
                                        const QuadRule &GaussQR) {
  unsigned N = GaussQR.n; // Quadrature order for the GaussQR object. Same order
                          // to be used for log weighted quadrature
  int Q = space.getQ(); // No. of Reference Shape Functions in trial/test space
  // Interaction matrix with size Q x Q
  Eigen::MatrixXd interaction_matrix(Q, Q);
  // Computing the (i,j)th matrix entry
  for (int i = 0; i < Q; ++i) {
    for (int j = 0; j < Q; ++j) {
      // when transforming the parametrizations from [-1,1]->\Pi to local
      // arclength parametrizations [0,|\Pi|] -> \Pi, swap is used to ensure
      // that the common point between the panels corresponds to the parameter 0
      // in both arclength parametrizations
      bool swap = (pi(1) - pi_p(-1)).norm() / 100. >
                  std::numeric_limits<double>::epsilon();
      // Panel lengths for local arclength parametrization in
      // \f$\eqref{eq:ap}\f$. Actual values are not required so a length of 1 is
      // used for both the panels
      double length_pi =
          2 * pi.Derivative(swap ? -1 : 1)
                  .norm(); // Length for panel pi to ensure norm of arclength
                           // parametrization is 1 at the common point
      double length_pi_p =
          2 * pi_p.Derivative(swap ? 1 : -1)
                  .norm(); // Length for panel pi_p to ensure norm of arclength
                           // parametrization is 1 at the common point

      assert((pi(swap ? -1 : 1) - pi_p(swap ? 1 : -1)).norm() <
             10. * std::numeric_limits<double>::epsilon());
      assert(fabs(pi.Derivative(swap ? -1 : 1).norm() * 2 / length_pi - 1.) <
             10. * std::numeric_limits<double>::epsilon());
      assert(fabs(pi_p.Derivative(swap ? 1 : -1).norm() * 2 / length_pi_p -
                  1.) < 10. * std::numeric_limits<double>::epsilon());

      // Lambda expressions for the functions F,G and D(r,phi) in
      // \f$\eqref{eq:Isplitapn}\f$
      auto F = [&](double t_pr) { // Function associated with panel pi_p
        // Transforming the local arclength parameter to standard parameter
        // range [-1,1] using swap
        double t =
            swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
        return space.evaluateShapeFunction(j, t) * pi_p.Derivative(t).norm();
      };

      auto G = [&](double s_pr) { // Function associated with panel pi
        // Transforming the local arclength parameter to standard parameter
        // range [-1,1] using swap
        double s = swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
        return space.evaluateShapeFunction(i, s) * pi.Derivative(s).norm();
      };

      auto D_r_phi = [&](double r, double phi) { // \f$\eqref{eq:Ddef}\f$
        double sqrt_epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        // Transforming to local arclength parameter range
        double s_pr = r * cos(phi);
        // Transforming to standard parameter range [-1,1] using swap
        double s = swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
        // Transforming to local arclength parameter range
        double t_pr = r * sin(phi);
        // Transforming to standard parameter range [-1,1] using swap
        double t =
            swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
        if (r > sqrt_epsilon) // Away from singularity, simply use the formula
          return (pi(s) - pi_p(t)).squaredNorm() / r / r;
        else // Near singularity, use analytically evaluated limit at r -> 0 for
             // stable evaluation \f$\eqref{eq:Dstab}\f$
          return 1 + sin(2 * phi) * pi.Derivative(s).dot(pi_p.Derivative(t)) *
                         4 / length_pi / length_pi_p;
      };

      // The two integrals in \f$\eqref{eq:Isplitapn}\f$ have to be further
      // split into two parts part 1 is where phi goes from 0 to alpha part 2 is
      // where phi goes from alpha to pi/2
      double alpha = atan(length_pi_p / length_pi); // the split point

      // Integrand for the second double integral in equation
      // \f$\eqref{eq:Isplitapn}\f$, without the log weight
      auto integrand2 = [&](double r, double phi) {
        return r * F(r * sin(phi)) * G(r * cos(phi));
      };

      // Part 1 of second double integral, integrated only in r, as a function
      // of phi (phi from 0 to alpha)
      auto inner21 = [&](double phi) {
        // Integrand as a function of r only (fixed phi), to calculate the inner
        // integral
        auto in = [&](double r) { return integrand2(r, phi); };
        // Integrating the inner integral in r using log weighted quadrature
        // return ComputeLogwtIntegral(in,length_pi/cos(phi),GaussQR);
        return ComputeLoogIntegral(in, length_pi / cos(phi), GaussQR);
      };

      // Part 2 of second double integral, integrated only in r, as a function
      // of phi (phi from alpha to pi/2)
      auto inner22 = [&](double phi) {
        // Integrand as a function of r only (fixed phi), to calculate the inner
        // integral
        auto in = [&](double r) { return integrand2(r, phi); };
        // Integrating the inner integral in r using log weighted quadrature
        // return ComputeLogwtIntegral(in,length_pi_p/sin(phi),GaussQR);
        return ComputeLoogIntegral(in, length_pi_p / sin(phi), GaussQR);
      };

      // i_IJ -> Integral I, part J
      double i11 = 0., i21 = 0., i12 = 0., i22 = 0.;
      // Computing both the parts of the second integral by integrating the
      // inner integrals w.r.t. phi
      i21 = ComputeIntegral(inner21, 0, alpha, GaussQR);
      i22 = ComputeIntegral(inner22, alpha, M_PI / 2, GaussQR);
      // Computing the first integral

      // part 1 (phi from 0 to alpha)
      for (unsigned int i = 0; i < N; ++i) {
        // Transforming gauss quadrature node into phi
        double phi = alpha / 2 * (1 + GaussQR.x(i));
        // Computing inner integral with fixed phi
        // Inner integral for double integral 1, evaluated with Gauss Legendre
        // quadrature
        double inner1 = 0.;
        // Upper limit for inner 'r' integral
        double rmax = length_pi / cos(phi);
        // Evaluating the inner 'r' integral
        for (unsigned int j = 0; j < N; ++j) {
          // Evaluating inner1 using Gauss Legendre quadrature
          double r = rmax / 2 * (1 + GaussQR.x(j));
          inner1 += GaussQR.w(j) * r * log(D_r_phi(r, phi)) * F(r * sin(phi)) *
                    G(r * cos(phi));
        }
        // Multiplying the integral with appropriate constants for
        // transformation to r from Gauss Legendre nodes
        inner1 *= rmax / 2;
        // Multiplying the integrals with appropriate constants for
        // transformation to phi from Gauss Legendre nodes
        i11 += GaussQR.w(i) * inner1 * alpha / 2;
      }

      // part 2 (phi from alpha to pi/2)
      for (unsigned int i = 0; i < N; ++i) {
        // Transforming gauss quadrature node into phi (alpha to pi/2)
        double phi =
            GaussQR.x(i) * (M_PI / 2. - alpha) / 2. + (M_PI / 2. + alpha) / 2.;
        // Computing inner integral with fixed phi
        // Inner integral for double integral 1, evaluated with Gauss Legendre
        // quadrature
        double inner1 = 0.;
        // Upper limit for inner 'r' integral
        double rmax = length_pi_p / sin(phi);
        // Evaluating the inner 'r' integral
        for (unsigned int j = 0; j < N; ++j) {
          // Evaluating inner1 using Gauss Legendre quadrature
          double r = rmax / 2 * (1 + GaussQR.x(j));
          inner1 += GaussQR.w(j) * r * log(D_r_phi(r, phi)) * F(r * sin(phi)) *
                    G(r * cos(phi));
        }
        // Multiplying the integral with appropriate constants for
        // transformation to r from Gauss Legendre quadrature nodes
        inner1 *= rmax / 2;
        // Multiplying the integrals with appropriate constants for
        // transformation to phi from Gauss Legendre quadrature nodes
        i12 += GaussQR.w(i) * inner1 * (M_PI / 2. - alpha) / 2.;
      }
      // Summing up the parts to get the final integral
      double integral = 0.5 * (i11 + i12) + (i21 + i22);
      // Multiplying the integral with appropriate constants for transformation
      // to local arclength variables
      integral *= 4. / length_pi / length_pi_p;
      // Filling up the matrix entry
      interaction_matrix(i, j) = -1. / (2 * M_PI) * integral;
    }
  }
  return interaction_matrix;
}

Eigen::MatrixXd ComputeIntegralGeneral(const AbstractParametrizedCurve &pi,
                                       const AbstractParametrizedCurve &pi_p,
                                       const AbstractBEMSpace &space,
                                       const QuadRule &GaussQR) {
  unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
  // Calculating the quadrature order for stable evaluation of integrands for
  // disjoint panels as mentioned in \f$\ref{par:distpan}\f$
  unsigned n0 = 9; // Order for admissible cases
  // Admissibility criteria
  double eta = 0.5;
  // Calculating the quadrature order
  // N = n0 * std::max(1., 1. + 51 * log(rho(pi, pi_p) / eta));
  // QuadRule GaussQR = getGaussQR(N);
  // std::cout << "ComputeIntegralGeneral used with order " << N << std::endl;
  // No. of Reference Shape Functions in trial/test space
  int Q = space.getQ();
  // Interaction matrix with size Q x Q
  Eigen::MatrixXd interaction_matrix(Q, Q);
  // Computing the (i,j)th matrix entry
  for (int i = 0; i < Q; ++i) {
    for (int j = 0; j < Q; ++j) {
      // Lambda expression for functions F and G in \f$\eqref{eq:titg}\f$ for
      // Single Layer BIO
      auto F = [&](double t) { // Function associated with panel pi_p
        return space.evaluateShapeFunction(j, t) * pi_p.Derivative(t).norm();
      };

      auto G = [&](double s) { // Function associated with panel pi
        return space.evaluateShapeFunction(i, s) * pi.Derivative(s).norm();
      };

      double integral = 0.;

      // Tensor product quadrature rule
      for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
          double s = GaussQR.x(i);
          double t = GaussQR.x(j);
          integral += GaussQR.w(i) * GaussQR.w(j) *
                      log((pi(s) - pi_p(t)).norm()) * F(t) * G(s);
        }
      }
      // Filling up the matrix entry
      interaction_matrix(i, j) = -1. / (2 * M_PI) * integral;
    }
  }
  return interaction_matrix;
}

Eigen::MatrixXd GalerkinMatrix(const ParametrizedMesh mesh,
                               const AbstractBEMSpace &space,
                               const unsigned int &N) {
  // Getting the number of panels in the mesh
  unsigned int numpanels = mesh.getNumPanels();
  // Getting dimensions of trial/test space
  unsigned int dims = space.getSpaceDim(numpanels);
  // Getting the panels from the mesh
  PanelVector panels = mesh.getPanels();
  // Getting the number of local shape functions in the trial/test space
  unsigned int Q = space.getQ();
  // Initializing the Galerkin matrix with zeros
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(dims, dims);
  // Panel oriented assembly \f$\ref{pc:ass}\f$
  QuadRule LogWeightQR = getLogWeightQR(1, N);
  QuadRule GaussQR = getGaussQR(N);
  for (unsigned int i = 0; i < numpanels; ++i) {
    for (unsigned int j = 0; j < numpanels; ++j) {
      // std::cout << "For panels "<< i <<  "," <<j << " " ;
      // Getting the interaction matrix for the pair of panels i and j
      Eigen::MatrixXd interaction_matrix =
          InteractionMatrix(*panels[i], *panels[j], space, GaussQR);
      // Local to global mapping of the elements in interaction matrix
      for (unsigned int I = 0; I < Q; ++I) {
        for (unsigned int J = 0; J < Q; ++J) {
          // int II = space.LocGlobMap(I + 1, i + 1, numpanels) - 1;
          // int JJ = space.LocGlobMap(J + 1, j + 1, numpanels) - 1;
          int II = space.LocGlobMap2(I + 1, i + 1, mesh) - 1;
          int JJ = space.LocGlobMap2(J + 1, j + 1, mesh) - 1;
          // Filling the Galerkin matrix entries
          output(II, JJ) += interaction_matrix(I, J);
        }
      }
    }
  }
  return output;
}

double Potential(const Eigen::Vector2d &x, const Eigen::VectorXd &coeffs,
                 const ParametrizedMesh &mesh, const AbstractBEMSpace &space,
                 const unsigned int &N) {
  // Getting the number of panels in the mesh
  unsigned int numpanels = mesh.getNumPanels();
  // Getting dimensions of BEM space
  unsigned int dims = space.getSpaceDim(numpanels);
  // asserting that the space dimension matches with coefficients
  assert(coeffs.rows() == dims);
  // Getting the panels from the mesh
  PanelVector panels = mesh.getPanels();
  // Getting the number of local shape functions in the BEM space
  unsigned int Q = space.getQ();
  // Getting general Gauss Quadrature rule
  QuadRule GaussQR = getGaussQR(N);
  // Initializing the single layer potential vector for the bases, with zeros
  Eigen::VectorXd potentials = Eigen::VectorXd::Zero(dims);
  // Looping over all the panels
  for (unsigned panel = 0; panel < numpanels; ++panel) {
    // Looping over all reference shape functions
    for (unsigned i = 0; i < Q; ++i) {
      // The local integrand for a panel and a reference shape function
      auto integrand = [&](double t) {
        // Single Layer Potential
        return -1. / 2. / M_PI *
               log((x - panels[panel]->operator()(t)).norm()) *
               space.evaluateShapeFunction(i, t) *
               panels[panel]->Derivative(t).norm();
      };
      // Local to global mapping
      double local = ComputeIntegral(integrand, -1., 1., GaussQR);
      //unsigned ii = space.LocGlobMap(i + 1, panel + 1, numpanels) - 1;
      unsigned ii = space.LocGlobMap2(i + 1, panel + 1, mesh) - 1;
      // Filling the potentials vector
      potentials(ii) += local;
    }
  }
  return coeffs.dot(potentials);
}

} // namespace single_layer
} // namespace parametricbem2d
