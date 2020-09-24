#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <fstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "discontinuous_space.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_mesh.hpp"
#include "single_layer.hpp"

#define _USE_MATH_DEFINES // for Pi

int main() {
  std::string filename = "sl_circle_test_eigs.txt";
  std::ofstream output(filename);
  // Getting a uniform circular mesh
  Eigen::Vector2d center;
  // Center for the circle
  center << 0, 0;
  parametricbem2d::ParametrizedCircularArc circle(center, 1., 0, 2 * M_PI);
  using PanelVector = parametricbem2d::PanelVector;
  // Number of panels
  unsigned int N = 100;
  // Splitting the whole circle to get panels
  PanelVector circle_panels = circle.split(N);
  // Creating a mesh with the obtained panels
  parametricbem2d::ParametrizedMesh uniform_mesh(circle_panels);
  // order of integration
  unsigned int order = 50;
  // BEM space for evaluating the Single Layer Galerkin matrix
  parametricbem2d::DiscontinuousSpace<0> space;
  Eigen::MatrixXd galerkin =
      parametricbem2d::single_layer::GalerkinMatrix(uniform_mesh, space, order);
  // Verifying that Galerkin matrix is circulant
  for (unsigned int i = 0; i < N - 1; ++i) {
    Eigen::VectorXd col1 = galerkin.col(i);
    Eigen::VectorXd col2(N);
    col2 << galerkin.col(i + 1).segment(1, N - 1), galerkin.col(i + 1)(0);
    assert((col1 - col2).norm() < 1e-7);
  }
  // Getting the eigenvalues
  Eigen::EigenSolver<Eigen::MatrixXd> es(galerkin);
  Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> eigs =
      es.eigenvalues();
  Eigen::VectorXd eigs_r(N);
  // Extracting the real part of eigenvalues (imaginary zero in this case)
  for (unsigned int i = 0; i < N; ++i)
    eigs_r(i) = eigs(i).real();
  // Sorting the real parts of eigenvalues
  std::sort(eigs_r.data(), eigs_r.data() + eigs_r.size());
  // Saving the real parts to a file
  for (unsigned int i = 0; i < N; ++i)
    output << eigs_r(i) << std::endl;
  return 0;
}
