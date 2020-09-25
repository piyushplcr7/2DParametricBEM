/**
 * \file annular_dvp.cpp
 * \brief This file demonstrates the solving of a Dirichlet BVP using the
 * direct first kind formulation, for a domain that is annular. Specifically,
 * the domain is bounded between two ellipses. The outer one has the
 * parametrization [3 cosx, 4 sinx] and the inner one has the parametrization
 * [2 cosx, sinx]. The potential is known beforehand and is equal to x+y.
 *
 * This File is a part of the 2D-Parametric BEM package
 */

#include "dirichlet.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

int main() {
  // Preparing the output files
  std::string fname2 = "neumann2dpbem.txt";
  std::string fname3 = "tnpbemex.txt";
  // Initializing the output streams
  std::ofstream out2(fname2);
  std::ofstream out3(fname3);
  // Setting up the annular domain
  // Setting up the outer ellipse [3 cosx, 4 sinx]
  Eigen::MatrixXd cos_list_o(2, 2);
  cos_list_o << 3, 0, 0, 0;
  Eigen::MatrixXd sin_list_o(2, 2);
  sin_list_o << 0, 0, 4, 0;
  // Setting up the inner ellipse [2 cosx, sinx]
  Eigen::MatrixXd cos_list_i(2, 2);
  cos_list_i << 2, 0, 0, 0;
  Eigen::MatrixXd sin_list_i(2, 2);
  sin_list_i << 0, 0, 1, 0;
  // Setting up the parametric curves
  parametricbem2d::ParametrizedFourierSum outer(
      Eigen::Vector2d(0, 0), cos_list_o, sin_list_o, 0, 2 * M_PI);
  // The inner curve goes from 2*Pi to 0. The reason for that is that the
  // 'outward' normal for the inner boundary should point inside
  parametricbem2d::ParametrizedFourierSum inner(
      Eigen::Vector2d(0, 0), cos_list_i, sin_list_i, 2 * M_PI, 0);

  // The known solution for the potential. Also encodes the dirichlet data
  auto potential = [&](double x, double y) { return x + y; };

  unsigned maxpanels = 200;
  // looping over number of panels
  for (unsigned numpanels = 1; numpanels < maxpanels; numpanels += 3) {
    // Number of panels for the inner curve
    unsigned numpanels_i = numpanels;
    // Number of panels for the outer curve
    unsigned numpanels_o = numpanels;
    // PanelVector for inner curve
    parametricbem2d::PanelVector panels_i = inner.split(numpanels_i);
    // PanelVector for outer curve
    parametricbem2d::PanelVector panels_o = outer.split(numpanels_o);
    // Single PanelVector to hold both PanelVectors
    parametricbem2d::PanelVector panels;
    panels.insert(panels.end(), panels_i.begin(),
                  panels_i.end()); // inner panels added first
    panels.insert(panels.end(), panels_o.begin(), panels_o.end());
    // Creating ParametrizedMesh object
    parametricbem2d::ParametrizedMesh mesh(panels);
    // Solving using direct first kind formulation
    Eigen::VectorXd sol =
        parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh,
                                                                 potential, 16);
    // Vector to store the calculated Neumann Trace. The solution is padded with
    // zeros to get a vector of fixed length (maxpanels).
    Eigen::VectorXd tn2dpbem(maxpanels);
    // the padding of zeros
    Eigen::VectorXd app =
        Eigen::VectorXd::Constant(maxpanels - 2 * numpanels, 0);
    // padding the solution with zeros
    tn2dpbem << sol, app;
    // writing the solution to the output file
    out2 << tn2dpbem.transpose() << std::endl;

    // writing the exact neumann trace to the output file
    Eigen::VectorXd v(2 * numpanels); // neumann
    for (unsigned I = 0; I < 2 * numpanels; ++I) {
      Eigen::VectorXd tangent = mesh.getPanels()[I]->Derivative(0.);
      Eigen::Vector2d normal;
      // Outward normal vector
      normal << tangent(1), -tangent(0);
      // Normalizing the normal vector
      normal = normal / normal.norm();
      // gradient of the true solution
      Eigen::VectorXd gradu(2);
      gradu << 1, 1;
      // neumann trace
      v(I) = gradu.dot(normal);
    }
    // writing the exact neumann trace to the output file using zero padding
    Eigen::VectorXd tnpbemex(maxpanels);
    tnpbemex << v, app;
    out3 << tnpbemex.transpose() << std::endl;
  }
  return 0;
}
