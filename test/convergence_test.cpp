#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <string>
#include <fstream>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "gtest/gtest.h"
#include "buildK.hpp"
#include "buildV.hpp"
#include "buildW.hpp"
#include "buildM.hpp"
#include "BoundaryMesh.hpp"
#include "doubleLayerPotential.hpp"
#include "singleLayerPotential.hpp"
#include "abstract_bem_space.hpp"
#include "continuous_space.hpp"
#include "discontinuous_space.hpp"
#include "double_layer.hpp"
#include "integral_gauss.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include "parametrized_semi_circle.hpp"
#include "single_layer.hpp"
#include "hypersingular.hpp"
#include "dirichlet.hpp"
#include "neumann.hpp"

int main() {
  std::string filename = "convergence.txt";
  std::ofstream output(filename);
  output << std::setw(15) << "#order" << std::setw(15) << "error sl" << std::setw(15) << "error dl" << std::endl;
  // Convergence test for Single Layer Galerkin Matrix
  double a = 0.01; // Side of the square
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the square
  /*Eigen::RowVectorXd x1(2);
  x1 << -a/2, -a/2; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << a/2, -a/2; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << a/2, a/2; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << -a/2, a/2; // Point (0,1.5)*/
  Eigen::RowVectorXd x1(2);
  x1 << 0, -1.5; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0.5; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 0.1, .33; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << -1, -.1; // Point (0,1.5)
  // Parametrized line segments forming the edges of the polygon
  parametricbem2d::ParametrizedLine line1(x1, x2);
  parametricbem2d::ParametrizedLine line2(x2, x3);
  parametricbem2d::ParametrizedLine line3(x3, x4);
  parametricbem2d::ParametrizedLine line4(x4, x1);
  // Splitting the parametrized lines into panels for a mesh to be used for
  // BEM (Discretization). Here Split is used with input "1" implying that the
  // original edges are used as panels in our mesh.
  PanelVector line1panels = line1.split(1);
  PanelVector line2panels = line2.split(1);
  PanelVector line3panels = line3.split(1);
  PanelVector line4panels = line4.split(1);
  PanelVector panels;
  // Storing all the panels in order so that they form a polygon
  panels.insert(panels.end(), line1panels.begin(), line1panels.end());
  panels.insert(panels.end(), line2panels.begin(), line2panels.end());
  panels.insert(panels.end(), line3panels.begin(), line3panels.end());
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh parametrizedmesh(panels);
  // BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> space;
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::ContinuousSpace<1> trial_space;
  // Matrix to store Vertices/Corners of panels in the mesh to compute Galerkin
  // Matrix using CppHilbert
  Eigen::MatrixXd coords(4, 2);
  coords << x1, x2, x3, x4;
  // Matrix to store the end points of elements/edges of the panels in our mesh
  // used to compute Galerkin Matrix using CppHilbert
  Eigen::Matrix<int, 4, 2> elems;
  elems << 0, 1, 1, 2, 2, 3, 3, 0;
  // Creating a boundarymesh object used in CppHilbert library
  BoundaryMesh boundarymesh(coords, elems);
  // Galerkin Matrix computed using CppHilbert
  Eigen::MatrixXd cpp_sl;
  computeV(cpp_sl, boundarymesh, 0);
  Eigen::MatrixXd cpp_dl;
  computeK(cpp_dl, boundarymesh, 0);
  for (unsigned order = 2 ; order < 500 ; ++order) {
    Eigen::MatrixXd sl =
        parametricbem2d::single_layer::GalerkinMatrix(parametrizedmesh, space,
                                                         order);
    double err_sl = (cpp_sl-sl).norm()/cpp_sl.norm();

    Eigen::MatrixXd dl = parametricbem2d::double_layer::GalerkinMatrix(
        parametrizedmesh, trial_space, test_space, order);
    double err_dl = (cpp_dl-dl).norm()/cpp_dl.norm();
    output << std::setw(15) << order << std::setw(15) << err_sl << std::setw(15) << err_dl << std::endl;

  }
  output.close();
  return 0;
}
