/**
 * \file double_layer.cpp
 * \brief This file declares the functions to evaluate the entries of
 *        Galerkin matrices based on the bilinear form induced by the
 *        Double Layer BIO, using the transformations given in
 *        \f$\ref{ss:quadapprox}\f$ in the Lecture Notes for Advanced Numerical
 *        Methods for CSE.
 *
 * This File is a part of the 2D-Parametric BEM package
 */

#include "double_layer.hpp"

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
namespace double_layer {
Eigen::MatrixXd InteractionMatrix(const AbstractParametrizedCurve &pi,
                                  const AbstractParametrizedCurve &pi_p,
                                  const AbstractBEMSpace &trial_space,
                                  const AbstractBEMSpace &test_space,
                                  const QuadRule &GaussQR) {
  double tol = std::numeric_limits<double>::epsilon();

  if (&pi == &pi_p) // Same Panels case
    return ComputeIntegralCoinciding(pi, pi_p, trial_space, test_space,
                                     GaussQR);

  else if (fabs((pi(1) - pi_p(-1)).norm()) < tol ||
           fabs((pi(-1) - pi_p(1)).norm()) < tol) // Adjacent Panels case
    return ComputeIntegralAdjacent(pi, pi_p, trial_space, test_space, GaussQR);

  else // Disjoint panels case
    return ComputeIntegralGeneral(pi, pi_p, trial_space, test_space, GaussQR);
}

Eigen::MatrixXd ComputeIntegralCoinciding(const AbstractParametrizedCurve &pi,
                                          const AbstractParametrizedCurve &pi_p,
                                          const AbstractBEMSpace &trial_space,
                                          const AbstractBEMSpace &test_space,
                                          const QuadRule &GaussQR) {
  unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
  // The number of Reference Shape Functions in trial space
  int Qtrial = trial_space.getQ();
  // The number of Reference Shape Functions in test space
  int Qtest = test_space.getQ();

  // Interaction matrix with size Qtest x Qtrial
  Eigen::MatrixXd interaction_matrix(Qtest, Qtrial);
  // Computing the (i,j)th matrix entry
  for (int i = 0; i < Qtest; ++i) {
    for (int j = 0; j < Qtrial; ++j) {
      // Lambda expression for functions F and G in \f$\eqref{eq:Kidp}\f$
      auto F = [&](double t) {
        return trial_space.evaluateShapeFunction(j, t) *
               pi_p.Derivative(t).norm();
      };

      auto G = [&](double s) {
        return test_space.evaluateShapeFunction(i, s) * pi.Derivative(s).norm();
      };

      // Lambda expression for the integrand in \f$\eqref{eq:Kidp}\f$
      auto integrand = [&](double s, double t) {
        double sqrt_epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
        double k;
        // Finding the tangent of pi_p to get its normal
        Eigen::Vector2d tangent = pi_p.Derivative(t);
        Eigen::Vector2d normal;
        // Outward normal vector
        normal << tangent(1), -tangent(0);
        // Normalizing the normal vector
        normal = normal / normal.norm();
        if (fabs(s - t) > sqrt_epsilon) // Away from singularity
          k = (pi(s) - pi_p(t)).dot(normal) / (pi(s) - pi_p(t)).squaredNorm();
        else // Near singularity
          // Limit evaluated analytically for s -> t
          k = 0.5 * pi.DoubleDerivative(t).dot(normal) /
              pi.Derivative(t).squaredNorm();

        return k * F(t) * G(s);
      };

      double integral = 0;

      // Tensor product quadrature for double integral in \f$\eqref{eq:Kidp}\f$
      for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
          integral += GaussQR.w(i) * GaussQR.w(j) *
                      integrand(GaussQR.x(i), GaussQR.x(j));
        }
      }
      // Filling the matrix entry
      interaction_matrix(i, j) = 1. / (2. * M_PI) * integral;
    }
  }
  return interaction_matrix;
}

Eigen::MatrixXd ComputeIntegralAdjacent(const AbstractParametrizedCurve &pi,
                                        const AbstractParametrizedCurve &pi_p,
                                        const AbstractBEMSpace &trial_space,
                                        const AbstractBEMSpace &test_space,
                                        const QuadRule &GaussQR) {
  unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
  // The number of Reference Shape Functions in trial space
  int Qtrial = trial_space.getQ();
  // The number of Reference Shape Functions in test space
  int Qtest = test_space.getQ();
  // Interaction matrix with size Qtest x Qtrial
  Eigen::MatrixXd interaction_matrix(Qtest, Qtrial);
  // Computing the (i,j)th matrix entry
  for (int i = 0; i < Qtest; ++i) {
    for (int j = 0; j < Qtrial; ++j) {
      // Panel lengths for local arclength parametrization in
      // \f$\eqref{eq:ap}\f$. Actual values are not required so a length of 1 is
      // used for both the panels
      double length_pi = 1.;   // Length for panel pi
      double length_pi_p = 1.; // Length for panel pi_p

      // when transforming the parametrizations from [-1,1]->\Pi to local
      // arclength parametrizations [0,|\Pi|] -> \Pi, swap is used to ensure
      // that the common point between the panels corresponds to the parameter 0
      // in both arclength parametrizations
      bool swap = (pi(1) != pi_p(-1));

      // Lambda expressions for functions F,G and the integrand in
      // \f$\eqref{eq:Kitrf}\f$
      auto F = [&](double t_pr) { // Function associated with panel pi_p
        // Transforming the local arclength parameter to standard parameter
        // range [-1,1] using swap
        double t =
            swap ? 1 - 2 * t_pr / length_pi_p : 2 * t_pr / length_pi_p - 1;
        return trial_space.evaluateShapeFunction(j, t) *
               pi_p.Derivative(t).norm();
      };

      auto G = [&](double s_pr) { // Function associated with panel pi
        // Transforming the local arclength parameter to standard parameter
        // range [-1,1] using swap
        double s = swap ? 2 * s_pr / length_pi - 1 : 1 - 2 * s_pr / length_pi;
        return test_space.evaluateShapeFunction(i, s) * pi.Derivative(s).norm();
      };

      // Lambda expressions for the integrand in \f$\eqref{eq:Kitrf}\f$ in polar
      // coordinates
      auto integrand = [&](double r, double phi) {
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
        // Getting the tangent vector to find the normal vector
        Eigen::Vector2d tangent = pi_p.Derivative(t);
        Eigen::Vector2d normal;
        // Outward normal vector
        normal << tangent(1), -tangent(0);
        // Normalizing the normal vector
        normal = normal / normal.norm();
        if (r > sqrt_epsilon) // Away from singularity
          // Stable evaluation according to \f$\eqref{eq:Vbstab}\f$
          return r * (pi(s) - pi_p(t)).dot(normal) /
                 (pi(s) - pi_p(t)).squaredNorm();
        else { // near singularity
          // Transforming 0 to standard parameter range [-1,1] using swap
          double s0 = swap ? -1 : 1;
          double t0 = swap ? 1 : -1;
          // Using the analytic limit for r - > 0, \f$\eqref{eq:Nexp}\f$
          Eigen::Vector2d b_r_phi =
              pi.Derivative(s0) * cos(phi) - pi_p.Derivative(t0) * sin(phi);
          return b_r_phi.dot(normal) / b_r_phi.squaredNorm();
        }
      };

      // The integral is split into two parts according to eq. 1.4.178
      // part 1 is where phi goes from 0 to alpha
      // part 2 is where phi goes from alpha to pi/2
      double alpha = atan(length_pi_p / length_pi); // the split point

      // i_J -> part J of the integral
      double i1 = 0., i2 = 0.;
      // part 1 (phi from 0 to alpha)
      for (unsigned int i = 0; i < N; ++i) {
        // Transforming gauss quadrature node into phi (0 to alpha)
        double phi = alpha / 2 * (1 + GaussQR.x(i));
        // Computing inner integral with fixed phi
        double inner = 0.; // inner integral
        // Upper limit for inner 'r' integral
        double rmax = length_pi / cos(phi);
        // Evaluating the inner 'r' integral
        for (unsigned int j = 0; j < N; ++j) {
          // Transforming Gauss quadrature node into r
          double r = rmax / 2 * (1 + GaussQR.x(j));
          inner += GaussQR.w(j) * integrand(r, phi) * F(r * sin(phi)) *
                   G(r * cos(phi));
        }
        // Multiplying the integral with appropriate constants for
        // transformation to r using Gauss quadrature nodes
        inner *= rmax / 2;
        // Multiplying the integral with appropriate constants for
        // transformation to phi using Gauss quadrature nodes
        i1 += GaussQR.w(i) * inner * alpha / 2;
      }

      // part 2 (phi from alpha to pi/2)
      for (unsigned int i = 0; i < N; ++i) {
        // Transforming gauss quadrature node into phi (alpha to pi/2)
        double phi =
            GaussQR.x(i) * (M_PI / 2. - alpha) / 2. + (M_PI / 2. + alpha) / 2.;
        double inner = 0.; // inner integral
        // Upper limit for inner 'r' integral
        double rmax = length_pi_p / sin(phi);
        // Evaluating the inner 'r' integral
        for (unsigned int j = 0; j < N; ++j) {
          // Transforming Gauss quadrature node into r
          double r = rmax / 2 * (1 + GaussQR.x(j));
          inner += GaussQR.w(j) * integrand(r, phi) * F(r * sin(phi)) *
                   G(r * cos(phi));
        }
        // Multiplying the integral with appropriate constants for
        // transformation to r using Gauss quadrature nodes
        inner *= rmax / 2;
        // Multiplying the integral with appropriate constants for
        // transformation to phi using Gauss quadrature nodes
        i2 += GaussQR.w(i) * inner * (M_PI / 2. - alpha) / 2.;
      }
      // Summing up the parts to get the final integral
      double integral = i1 + i2;
      // Multiplying the integral with appropriate constants for transformation
      // to local arclength variables
      integral *= 4 / length_pi / length_pi_p;
      // Filling the matrix entry
      interaction_matrix(i, j) = 1 / (2 * M_PI) * integral;
    }
  }
  return interaction_matrix;
}

Eigen::MatrixXd ComputeIntegralGeneral(const AbstractParametrizedCurve &pi,
                                       const AbstractParametrizedCurve &pi_p,
                                       const AbstractBEMSpace &trial_space,
                                       const AbstractBEMSpace &test_space,
                                       const QuadRule &GaussQR) {
  unsigned N = GaussQR.n; // Quadrature order for the GaussQR object.
  // Calculating the quadrature order for stable evaluation of integrands for
  // disjoint panels as mentioned in \f$\ref{par:distpan}\f$
  unsigned n0 = 5; // Order for admissible cases
  // Admissibility criteria
  double eta = 0.5;
  // Calculating the quadrature order
  unsigned n = n0 * std::max(1., 1. + log(rho(pi, pi_p) / eta));
  // std::cout << "calculated rho dl " << log(rho(pi,pi_p)/eta) << std::endl;
  // The number of Reference Shape Functions in space
  int Qtrial = trial_space.getQ();
  // The number of Reference Shape Functions in space
  int Qtest = test_space.getQ();
  // Interaction matrix with size Qtest x Qtrial
  Eigen::MatrixXd interaction_matrix(Qtest, Qtrial);
  // Computing the (i,j)th matrix entry
  for (int i = 0; i < Qtest; ++i) {
    for (int j = 0; j < Qtrial; ++j) {
      // Lambda expression for functions F and G in \f$\eqref{eq:titg}\f$ for
      // Double Layer BIO
      auto F = [&](double t) { // Function associated with panel pi_p
        return trial_space.evaluateShapeFunction(j, t) *
               pi_p.Derivative(t).norm();
      };

      auto G = [&](double s) { // Function associated with panel pi
        return test_space.evaluateShapeFunction(i, s) * pi.Derivative(s).norm();
      };
      // Lambda expression for \f$\hat{K}\f$ in \f$\eqref{eq:titg}\f$ for double
      // Layer BIO
      auto integrand = [&](double s, double t) {
        // Finding the tangent of pi_p to get its normal
        Eigen::Vector2d tangent = pi_p.Derivative(t);
        Eigen::Vector2d normal;
        // Outward normal vector
        normal << tangent(1), -tangent(0);
        // Normalizing the normal vector
        normal = normal / normal.norm();
        return (pi(s) - pi_p(t)).dot(normal) / (pi(s) - pi_p(t)).squaredNorm();
      };

      double integral = 0.;

      // Getting quadrature weights and points
      Eigen::RowVectorXd weights, points;
      std::tie(points, weights) =
          gauleg(-1, 1, N, std::numeric_limits<double>::epsilon());

      // Tensor product quadrature for double integral
      for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
          double s = GaussQR.x(i);
          double t = GaussQR.x(j);
          integral +=
              GaussQR.w(i) * GaussQR.w(j) * integrand(s, t) * F(t) * G(s);
        }
      }
      // Filling the matrix entry
      interaction_matrix(i, j) = 1 / (2 * M_PI) * integral;
    }
  }
  return interaction_matrix;
}

Eigen::MatrixXd GalerkinMatrix(const ParametrizedMesh mesh,
                               const AbstractBEMSpace &trial_space,
                               const AbstractBEMSpace &test_space,
                               const unsigned int &N) {
  // Getting number of panels in the mesh
  unsigned int numpanels = mesh.getNumPanels();
  // Getting dimensions for trial and test spaces
  unsigned int rows = test_space.getSpaceDim(numpanels);
  unsigned int cols = trial_space.getSpaceDim(numpanels);
  // Getting the panels from the mesh
  PanelVector panels = mesh.getPanels();
  // Getting the number of local shape functions in the trial and test spaces
  unsigned int Qtest = test_space.getQ();
  unsigned int Qtrial = trial_space.getQ();
  // Initializing the Galerkin matrix with zeros
  Eigen::MatrixXd output = Eigen::MatrixXd::Zero(rows, cols);
  // Panel oriented assembly \f$\ref{pc:ass}\f$
  QuadRule LogWeightQR = getLogWeightQR(1, N);
  QuadRule GaussQR = getGaussQR(N);
  for (unsigned int i = 0; i < numpanels; ++i) {
    for (unsigned int j = 0; j < numpanels; ++j) {
      // Getting the interaction matrix for the pair of panels i and j
      Eigen::MatrixXd interaction_matrix = InteractionMatrix(
          *panels[i], *panels[j], trial_space, test_space, GaussQR);
      // Local to global mapping of the elements in interaction matrix
      for (unsigned int I = 0; I < Qtest; ++I) {
        for (unsigned int J = 0; J < Qtrial; ++J) {
          int II = test_space.LocGlobMap(I + 1, i + 1, numpanels) - 1;
          int JJ = trial_space.LocGlobMap(J + 1, j + 1, numpanels) - 1;
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
  // Initializing the double layer potential vector for the bases, with zeros
  Eigen::VectorXd potentials = Eigen::VectorXd::Zero(dims);
  // Looping over all the panels
  for (unsigned panel = 0; panel < numpanels; ++panel) {
    // Looping over all reference shape functions
    for (unsigned i = 0; i < Q; ++i) {
      // The local integrand for a panel and a reference shape function
      auto integrand = [&](double t) {
        Eigen::Vector2d y = panels[panel]->operator()(t);
        Eigen::Vector2d tangent = panels[panel]->Derivative(t);
        Eigen::Vector2d normal;
        // Outward normal vector
        normal << tangent(1), -tangent(0);
        // Normalizing the normal vector
        normal /= normal.norm();
        double dotprod = (x - y).dot(normal) / (x - y).squaredNorm();
        // Double Layer Potential
        return 1. / 2. / M_PI * dotprod * space.evaluateShapeFunction(i, t) *
               tangent.norm();
      };
      // Local to global mapping
      double local = ComputeIntegral(integrand, -1, 1, GaussQR);
      unsigned ii = space.LocGlobMap(i + 1, panel + 1, numpanels) - 1;
      // Filling the potentials vector
      potentials(ii) += local;
    }
  }
  // std::cout << "calculated potentials: " << std::endl << potentials <<
  // std::endl;
  // Dot product with coefficients to get the final double layer potential
  return coeffs.dot(potentials);
}

} // namespace double_layer
} // namespace parametricbem2d
