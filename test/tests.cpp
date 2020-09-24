#include <stdlib.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <chrono>

#include "gtest/gtest.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "buildK.hpp"
#include "buildM.hpp"
#include "buildV.hpp"
#include "buildW.hpp"
#include "doubleLayerPotential.hpp"
#include "singleLayerPotential.hpp"
#include "BoundaryMesh.hpp"
#include "abstract_bem_space.hpp"
#include "continuous_space.hpp"
#include "dirichlet.hpp"
#include "discontinuous_space.hpp"
#include "double_layer.hpp"
#include "hypersingular.hpp"
#include "integral_gauss.hpp"
#include "neumann.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include "parametrized_semi_circle.hpp"
#include "single_layer.hpp"

#define _USE_MATH_DEFINES // for pi

double eps = 1e-5; // A global threshold for error

// LineParametrizationTest is hieararcy name, Parametrization is a test in this
// hierarchy
TEST(LineParametrizationTest, Parametrization) {
  using Point = std::pair<double, double>;
  Eigen::Vector2d x1;
  x1 << 0, 1; // Point (0,1)
  Eigen::Vector2d x2;
  x2 << 1, 0; // Point (1,0)
  parametricbem2d::ParametrizedLine parametrization(x1, x2);
  // Test point (0.5,0.5) corresponding to t = 0.0
  Eigen::Vector2d testpoint;
  testpoint = parametrization(0.);
  EXPECT_NEAR(0.5, testpoint(0), eps);
  EXPECT_NEAR(0.5, testpoint(1), eps);
}

TEST(LineParametrizationTest, Derivative) {
  using Point = std::pair<double, double>;
  Eigen::Vector2d x1;
  x1 << 0, 1; // Point (0,1)
  Eigen::Vector2d x2;
  x2 << 1, 0; // Point (1,0)
  parametricbem2d::ParametrizedLine parametrization(x1, x2);
  // Test point (0.5,0.5) corresponding to t = 0.0
  Eigen::Vector2d testpoint;
  testpoint = parametrization(0.);
  // Derivative at the point corresponding to t = 0.3
  testpoint = parametrization.Derivative(0.3);
  EXPECT_NEAR(-1, testpoint(1) / testpoint(0), eps);
}

TEST(SemiCircleParametrizationTest, Parametrization) {
  // Parametrized semi-circle with unit radius
  parametricbem2d::ParametrizedSemiCircle parametrization;
  double tmin, tmax;
  // Getting the parameter range
  std::tie(tmin, tmax) = parametrization.ParameterRange();
  double length = 0.;
  double t1, t2;
  // Representing the semi-circular parametrized curve by Nbins number of
  // Points or Nbins -1 number of line segments and summing up the lengths
  // to get an approximation of Pi
  int Nbins = 1000;
  for (int i = 0; i < Nbins - 1; ++i) {
    t1 = tmin + i * (tmax - tmin) / (Nbins - 1);       // Current Point
    t2 = tmin + (i + 1) * (tmax - tmin) / (Nbins - 1); // Next Point
    Eigen::Vector2d start, end;
    start = parametrization(t1);
    end = parametrization(t2);
    length += (end - start).norm(); // Adding the length of current line segment
  }
  EXPECT_NEAR(M_PI, length, eps);
}

TEST(ParametrizedCircularArcTest, Parametrization) {
  // Parametrized radius = 1.2, centered at (1,-1)
  Eigen::Vector2d center(2);
  center << 1, -1;
  double r = 1.2;
  parametricbem2d::ParametrizedCircularArc parametrization(center, r, 0,
                                                           M_PI / 2);
  double tmin, tmax;
  // Getting the parameter range
  std::tie(tmin, tmax) = parametrization.ParameterRange();
  double length = 0.;
  double t1, t2;
  // Representing the circular arc parametrized curve by Nbins number of
  // Points or Nbins -1 number of line segments and summing up the lengths
  // to get an approximation of Pi
  int Nbins = 1000;
  for (int i = 0; i < Nbins - 1; ++i) {
    t1 = tmin + i * (tmax - tmin) / (Nbins - 1);       // Current Point
    t2 = tmin + (i + 1) * (tmax - tmin) / (Nbins - 1); // Next Point
    Eigen::Vector2d start, end;
    start = parametrization(t1);
    end = parametrization(t2);
    length += (end - start).norm(); // Adding the length of current line segment
  }
  EXPECT_NEAR(M_PI, length * 2 / r, eps);
}

TEST(FourierSumParametrizationTest, Parametrization) {
  Eigen::MatrixXd a(2, 1); // cosine coefficients ; N = 1
  Eigen::MatrixXd b(2, 1); // sine coefficients ; N = 1
  a << 1., 0.;
  b << 0., 1.;
  // The fourier sum parametrization using these coefficients reduces to
  // (cos(t),sin(t))
  parametricbem2d::ParametrizedFourierSum parametrization(a, b);
  // Random point in [0,1]
  double t = rand() / RAND_MAX;
  t = 2 * t - 1; // Transforming to the range [-1,1]
  Eigen::Vector2d randompoint = parametrization(t);
  EXPECT_NEAR(1, randompoint.norm(), eps);
}

TEST(InterfaceTest, ParameterRange) {
  // Checking the static functionality of the function ParameterRange() which is
  // independent of concrete implementation of a parametrized curve.
  double a, b;
  std::tie(a, b) = parametricbem2d::ParametrizedLine::ParameterRange();
  EXPECT_EQ(a, -1);
  EXPECT_EQ(b, 1);
  std::tie(a, b) = parametricbem2d::ParametrizedFourierSum::ParameterRange();
  EXPECT_EQ(a, -1);
  EXPECT_EQ(b, 1);
  std::tie(a, b) = parametricbem2d::ParametrizedSemiCircle::ParameterRange();
  EXPECT_EQ(a, -1);
  EXPECT_EQ(b, 1);
}

TEST(InterfaceTest, IsWithinParameterRange) {
  // Testing the functionality of IsWithinParameterRange()
  EXPECT_EQ(true,
            parametricbem2d::ParametrizedLine::IsWithinParameterRange(0.99));
  EXPECT_EQ(
      false,
      parametricbem2d::ParametrizedFourierSum::IsWithinParameterRange(1.01));
  EXPECT_EQ(
      true,
      parametricbem2d::ParametrizedSemiCircle::IsWithinParameterRange(-0.99));
}

// Definition of the integrand which is used in IntegralGauss test
double integrand(double x) { return x * x; }

TEST(IntegralGauss, ComputeIntegral) {
  // Using Gauss Quadrature to find the integral value
  double integral = parametricbem2d::ComputeIntegral(integrand, 0, 1, 10);
  EXPECT_NEAR(integral, 1. / 3., eps);
}

TEST(BemSpace, DiscontinuousSpace0) {
  // Test for DiscontinuousSpace<0>
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::DiscontinuousSpace<0>();
  // Number of reference shape functions in the space.
  int Q = space->getQ();
  EXPECT_EQ(Q, 1);
  // Random point between [0,1]
  double t = rand() / RAND_MAX;
  t = 2 * t - 1; // Transformation to the range [-1,1]
  // Checking evaluationShapeFunction
  EXPECT_EQ(space->evaluateShapeFunction(0, t), 1);
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(BemSpace, DiscontinuousSpace1) {
  // Test for DiscontinuousSpace<1>
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::DiscontinuousSpace<1>();
  // Number of reference shape functions in the space.
  int Q = space->getQ();
  EXPECT_EQ(Q, 2);
  // Random point between [0,1]
  double t = rand() / RAND_MAX;
  t = 2 * t - 1; // Transformation to the range [-1,1]
  // Checking evaluationShapeFunction
  EXPECT_EQ(space->evaluateShapeFunction(0, t), 0.5);
  EXPECT_EQ(space->evaluateShapeFunction(1, t), 0.5 * t);
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(BemSpace, ContinuousSpace1) {
  // Test for ContinuousSpace<1>
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::ContinuousSpace<1>();
  // Number of reference shape functions in the space.
  int Q = space->getQ();
  EXPECT_EQ(Q, 2);
  // Random point between [0,1]
  double t = rand() / RAND_MAX;
  t = 2 * t - 1; // Transformation to the range [-1,1]
  // Checking evaluationShapeFunction
  EXPECT_EQ(space->evaluateShapeFunction(0, t), 0.5 * (1 + t));
  EXPECT_EQ(space->evaluateShapeFunction(1, t), 0.5 * (1 - t));
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(BemSpace, ContinuousSpace2) {
  // Test for ContinuousSpace<2>
  const int p = 2;
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::ContinuousSpace<p>();
  // Number of reference shape functions in the space.
  int Q = space->getQ();
  EXPECT_EQ(Q, 3);
  // Random point between [0,1]
  double t = rand() / RAND_MAX;
  t = 2 * t - 1; // Transformation to the range [-1,1]
  // Checking evaluationShapeFunction
  EXPECT_EQ(space->evaluateShapeFunction(0, t), 0.5 * (1 + t));
  EXPECT_EQ(space->evaluateShapeFunction(1, t), 0.5 * (1 - t));
  EXPECT_EQ(space->evaluateShapeFunction(2, t), (1 - t * t));
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(LocGlobMap, ContinuousSpace1) {
  // Testing the local to global map for ContinuousSpace<1>
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::ContinuousSpace<1>();
  EXPECT_EQ(space->LocGlobMap(1, 1, 4), 2);
  EXPECT_EQ(space->LocGlobMap(1, 2, 4), 3);
  EXPECT_EQ(space->LocGlobMap(1, 3, 4), 4);
  EXPECT_EQ(space->LocGlobMap(2, 3, 4), 3);
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(LocGlobMap, DiscontinuousSpace0) {
  // Testing the local to global map for DiscontinuousSpace<0>
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::DiscontinuousSpace<0>();
  EXPECT_EQ(space->LocGlobMap(1, 1, 4), 1);
  EXPECT_EQ(space->LocGlobMap(1, 2, 4), 2);
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(LocGlobMap, DiscontinuousSpace1) {
  // Testing the local to global map for DiscontinuousSpace<1>
  parametricbem2d::AbstractBEMSpace *space =
      new parametricbem2d::DiscontinuousSpace<1>();
  EXPECT_EQ(space->LocGlobMap(2, 1, 4), 5);
  EXPECT_EQ(space->LocGlobMap(2, 2, 4), 6);
  EXPECT_EQ(space->LocGlobMap(1, 4, 4), 4);
  EXPECT_EQ(space->LocGlobMap(2, 4, 4), 8);
  // Dynamically allocated, freeing memory
  delete space;
}

TEST(Split, ParametrizedLine) {
  // Testing the Split functionality for a parametrized line
  using Point = std::pair<double, double>;
  using PanelVector = parametricbem2d::PanelVector;
  Eigen::Vector2d x1;
  x1 << 0, 1; // Point (0,1)
  Eigen::Vector2d x2;
  x2 << 1, 0; // Point (1,0)
  parametricbem2d::ParametrizedLine parametrization(x1, x2);
  // Number of split components
  unsigned int N = 10;
  PanelVector components = parametrization.split(N);
  // Confirming that the components form a sequence
  for (unsigned int i = 0; i < N - 1; ++i) {
    EXPECT_NEAR(
        (components[i]->operator()(1) - components[i + 1]->operator()(-1))
            .norm(),
        0, eps);
  }
}

TEST(Split, ParametrizedCircularArc) {
  // Testing the Split functionality for a parametrized line
  using Point = std::pair<double, double>;
  using PanelVector = parametricbem2d::PanelVector;
  Eigen::Vector2d center;
  center << 2.5, 3.6;
  double radius = 33;
  parametricbem2d::ParametrizedCircularArc parametrization(
      center, radius, .33 * M_PI, .99 * M_PI);
  // Number of split components
  unsigned int N = 10;
  PanelVector components = parametrization.split(N);
  // Confirming that the components form a sequence
  for (unsigned int i = 0; i < N - 1; ++i) {
    // std::cout << "i: " << i << std::endl;
    EXPECT_NEAR(
        (components[i]->operator()(1) - components[i + 1]->operator()(-1))
            .norm(),
        0, eps);
  }
}

TEST(Split, ParametrizedFourierSum) {
  // Testing the Split functionality for a parametrized line
  Eigen::MatrixXd a(2, 1); // cosine coefficients ; N = 1
  Eigen::MatrixXd b(2, 1); // sine coefficients ; N = 1
  a << 1., 0.;
  b << 0., 1.;
  parametricbem2d::ParametrizedFourierSum parametrization(a, b);
  // Number of split components
  unsigned int N = 10;
  using PanelVector = parametricbem2d::PanelVector;
  PanelVector components = parametrization.split(N);
  // Confirming that the components form a sequence
  for (unsigned int i = 0; i < N - 1; ++i)
    EXPECT_NEAR(
        (components[i]->operator()(1) - components[i + 1]->operator()(-1))
            .norm(),
        0, eps);
}

TEST(ParametrizedMeshTest, DISABLED_MemberFunctions) {
  using PanelVector = parametricbem2d::PanelVector;
  // Definition of corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
  // Parametrized line segments forming the edges of the polygon
  parametricbem2d::ParametrizedLine line1(x1, x2);
  parametricbem2d::ParametrizedLine line2(x2, x3);
  parametricbem2d::ParametrizedLine line3(x3, x4);
  parametricbem2d::ParametrizedLine line4(x4, x1);
  // Splitting the parametrized lines into panels for a mesh to be used for
  // BEM (Discretization)
  PanelVector line1panels = line1.split(2);
  PanelVector line2panels = line2.split(2);
  PanelVector line3panels = line3.split(2);
  PanelVector line4panels = line4.split(2);
  PanelVector panels;
  // Storing all the panels in order so that they form a polygon
  panels.insert(panels.end(), line1panels.begin(), line1panels.end());
  panels.insert(panels.end(), line2panels.begin(), line2panels.end());
  panels.insert(panels.end(), line3panels.begin(), line3panels.end());
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh mesh(panels);
  // Checking functionality of ParametrizedMesh
  Eigen::Vector2d vertex2 = mesh.getVertex(1);
  EXPECT_EQ(mesh.getNumPanels(), 8);
  EXPECT_NEAR(vertex2(0), 0.5, eps);
  EXPECT_NEAR(vertex2(1), 0, eps);
}

TEST(SingleLayer_0, DISABLED_PanelOrientedAssembly) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
  // Parametrized line segments forming the edges of the polygon
  parametricbem2d::ParametrizedLine line1(x1, x2);
  parametricbem2d::ParametrizedLine line2(x2, x3);
  parametricbem2d::ParametrizedLine line3(x3, x4);
  parametricbem2d::ParametrizedLine line4(x4, x1);
  // Splitting the parametrized lines into panels for a mesh to be used for
  // BEM (Discretization)
  PanelVector line1panels = line1.split(1);
  PanelVector line2panels = line2.split(2);
  PanelVector line3panels = line3.split(3);
  PanelVector line4panels = line4.split(4);
  PanelVector panels;
  // Storing all the panels in order so that they form a polygon
  panels.insert(panels.end(), line1panels.begin(), line1panels.end());
  panels.insert(panels.end(), line2panels.begin(), line2panels.end());
  panels.insert(panels.end(), line3panels.begin(), line3panels.end());
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh mesh(panels);
  // BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> space;
  Eigen::MatrixXd galerkin =
      parametricbem2d::single_layer::GalerkinMatrix(mesh, space, 32);
  unsigned int numpanels = mesh.getNumPanels();
  // Checking the size of the obtained Galerkin Matrix
  EXPECT_EQ(numpanels, galerkin.cols());
  EXPECT_EQ(numpanels, galerkin.rows());
}

TEST(SingleLayer_1, DISABLED_PanelOrientedAssembly) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  // Construction of a ParametrizedMesh object from the vector of panels
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  parametricbem2d::ParametrizedMesh mesh(panels);
  // BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<1> space;
  Eigen::MatrixXd galerkin =
      parametricbem2d::single_layer::GalerkinMatrix(mesh, space, 32);
  unsigned int numpanels = mesh.getNumPanels();
  // Checking the size of the obtained Galerkin Matrix
  EXPECT_EQ(8, galerkin.cols());
  EXPECT_EQ(8, galerkin.rows());
}

TEST(SingleLayer_0, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  Eigen::MatrixXd galerkinnew = parametricbem2d::single_layer::GalerkinMatrix(
      parametrizedmesh, space, 32);
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
  Eigen::MatrixXd galerkinold;
  computeV(galerkinold, boundarymesh, 0);
  // Finding error using matrix norm
  EXPECT_NEAR((galerkinold - galerkinnew).norm(), 0, eps);
}

TEST(DoubleLayer_1_0, DISABLED_PanelOrientedAssembly) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
  // Parametrized line segments forming the edges of the polygon
  parametricbem2d::ParametrizedLine line1(x1, x2);
  parametricbem2d::ParametrizedLine line2(x2, x3);
  parametricbem2d::ParametrizedLine line3(x3, x4);
  parametricbem2d::ParametrizedLine line4(x4, x1);
  // Splitting the parametrized lines into panels for a mesh to be used for
  // BEM (Discretization)
  PanelVector line1panels = line1.split(1);
  PanelVector line2panels = line2.split(2);
  PanelVector line3panels = line3.split(3);
  PanelVector line4panels = line4.split(4);
  PanelVector panels;
  // Storing all the panels in order so that they form a polygon
  panels.insert(panels.end(), line1panels.begin(), line1panels.end());
  panels.insert(panels.end(), line2panels.begin(), line2panels.end());
  panels.insert(panels.end(), line3panels.begin(), line3panels.end());
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh mesh(panels);
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::ContinuousSpace<1> trial_space;
  Eigen::MatrixXd galerkin = parametricbem2d::double_layer::GalerkinMatrix(
      mesh, trial_space, test_space, 32);
  // Checking the size of the obtained Galerkin Matrix
  unsigned int numpanels = mesh.getNumPanels();
  EXPECT_EQ(numpanels, galerkin.cols());
  EXPECT_EQ(numpanels, galerkin.rows());
}

TEST(DoubleLayer_1_0, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::ContinuousSpace<1> trial_space;
  Eigen::MatrixXd galerkin = parametricbem2d::double_layer::GalerkinMatrix(
      parametrizedmesh, trial_space, test_space, 32);
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
  Eigen::MatrixXd galerkinold;
  computeK(galerkinold, boundarymesh, 0);
  // Finding error using matrix norm
  EXPECT_NEAR((galerkinold - galerkin).norm(), 0, eps);
}

TEST(DoubleLayer_0_0, DISABLED_PanelOrientedAssembly) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
  // Parametrized line segments forming the edges of the polygon
  parametricbem2d::ParametrizedLine line1(x1, x2);
  parametricbem2d::ParametrizedLine line2(x2, x3);
  parametricbem2d::ParametrizedLine line3(x3, x4);
  parametricbem2d::ParametrizedLine line4(x4, x1);
  // Splitting the parametrized lines into panels for a mesh to be used for
  // BEM (Discretization).
  PanelVector line1panels = line1.split(1);
  PanelVector line2panels = line2.split(2);
  PanelVector line3panels = line3.split(3);
  PanelVector line4panels = line4.split(4);
  PanelVector panels;
  // Storing all the panels in order so that they form a polygon
  panels.insert(panels.end(), line1panels.begin(), line1panels.end());
  panels.insert(panels.end(), line2panels.begin(), line2panels.end());
  panels.insert(panels.end(), line3panels.begin(), line3panels.end());
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh mesh(panels);
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  Eigen::MatrixXd galerkin = parametricbem2d::double_layer::GalerkinMatrix(
      mesh, trial_space, test_space, 32);
  unsigned int numpanels = mesh.getNumPanels();
  // Checking the size of the obtained Galerkin Matrix
  EXPECT_EQ(numpanels, galerkin.cols());
  EXPECT_EQ(numpanels, galerkin.rows());
}

TEST(DoubleLayer_0_0, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  Eigen::MatrixXd galerkin = parametricbem2d::double_layer::GalerkinMatrix(
      parametrizedmesh, trial_space, test_space, 32);
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
  Eigen::MatrixXd galerkinold;
  computeK00(galerkinold, boundarymesh, 0);
  // Finding error using matrix norm
  EXPECT_NEAR((galerkinold - galerkin).norm(), 0, eps);
}

TEST(AbstractParametrizedCurve, DistanceTo) {
  Eigen::Vector2d center;
  center << 0, 0;
  parametricbem2d::ParametrizedCircularArc curve1(center, 1., 0, M_PI / 2);
  Eigen::Vector2d point1(1 + 0.1, 0);
  Eigen::Vector2d point2(1 + 0.1, 1);
  parametricbem2d::ParametrizedLine curve2(point1, point2);
  double distance = curve1.distanceTo(curve2);
  EXPECT_NEAR(distance, 0.1, eps);
  /*Eigen::Vector2d point3(1,0); Eigen::Vector2d point4(0,1);
  parametricbem2d::ParametrizedLine curve3(point3,point4);
  std::tie(solution,distance) = curve3.distanceTo(curve2);
  EXPECT_NEAR(distance,0.1,eps);*/
  std::cout << "Distance = " << distance << std::endl;
}

TEST(ParametrizedCircularArc, Length) {
  // Defining the center for the circular arc
  Eigen::Vector2d center;
  center << 0, 0;
  parametricbem2d::ParametrizedCircularArc curve(center, 1., 0, M_PI / 2);
  // Evaluating the length of the curve
  double length = curve.length();
  EXPECT_NEAR(length, M_PI / 2, eps);
}

TEST(ParametrizedLine, Length) {
  // Defining the endpoints for parametrized line segment
  Eigen::Vector2d x1;
  x1 << 0, 1; // Point (0,1)
  Eigen::Vector2d x2;
  x2 << 1, 1; // Point (1,0)
  parametricbem2d::ParametrizedLine curve(x1, x2);
  // Evaluating the length of the curve
  double length = curve.length();
  EXPECT_NEAR(length, 1, eps);
}

TEST(Hypersingular_1, DISABLED_PanelOrientedAssembly) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
  // Parametrized line segments forming the edges of the polygon
  parametricbem2d::ParametrizedLine line1(x1, x2);
  parametricbem2d::ParametrizedLine line2(x2, x3);
  parametricbem2d::ParametrizedLine line3(x3, x4);
  parametricbem2d::ParametrizedLine line4(x4, x1);
  // Splitting the parametrized lines into panels for a mesh to be used for
  // BEM (Discretization)
  PanelVector line1panels = line1.split(1);
  PanelVector line2panels = line2.split(2);
  PanelVector line3panels = line3.split(3);
  PanelVector line4panels = line4.split(4);
  PanelVector panels;
  // Storing all the panels in order so that they form a polygon
  panels.insert(panels.end(), line1panels.begin(), line1panels.end());
  panels.insert(panels.end(), line2panels.begin(), line2panels.end());
  panels.insert(panels.end(), line3panels.begin(), line3panels.end());
  panels.insert(panels.end(), line4panels.begin(), line4panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh mesh(panels);
  // BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::ContinuousSpace<1> space;
  Eigen::MatrixXd galerkin =
      parametricbem2d::hypersingular::GalerkinMatrix(mesh, space, 32);
  unsigned int numpanels = mesh.getNumPanels();
  // Checking the size of the obtained Galerkin Matrix
  EXPECT_EQ(numpanels, galerkin.cols());
  EXPECT_EQ(numpanels, galerkin.rows());
}

TEST(Hypersingular_1, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  parametricbem2d::ContinuousSpace<1> space;
  Eigen::MatrixXd galerkinnew = parametricbem2d::hypersingular::GalerkinMatrix(
      parametrizedmesh, space, 32);
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
  Eigen::MatrixXd galerkinold;
  computeW(galerkinold, boundarymesh, 0);
  // Finding error using matrix norm
  EXPECT_NEAR((galerkinold - galerkinnew).norm(), 0, eps);
}

TEST(MassMatrix00, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  Eigen::MatrixXd massnew =
      parametricbem2d::MassMatrix(parametrizedmesh, space, space, 32);
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
  Eigen::SparseMatrix<double> massold(4, 4);
  computeM00(massold, boundarymesh);
  Eigen::MatrixXd fullmass = massold * Eigen::MatrixXd::Identity(4, 4);
  // Finding error using matrix norm
  EXPECT_NEAR((fullmass - massnew).norm(), 0, eps);
}

TEST(MassMatrix11, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  parametricbem2d::ContinuousSpace<1> space;
  Eigen::MatrixXd massnew =
      parametricbem2d::MassMatrix(parametrizedmesh, space, space, 32);
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
  Eigen::SparseMatrix<double> massold(4, 4);
  computeM11(massold, boundarymesh);
  Eigen::MatrixXd fullmass = massold * Eigen::MatrixXd::Identity(4, 4);
  // Finding error using matrix norm
  EXPECT_NEAR((fullmass - massnew).norm(), 0, eps);
}

TEST(MassMatrix01, DISABLED_CppHilbertComparison) {
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 1, .5; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 0, 1.5; // Point (0,1.5)
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
  parametricbem2d::DiscontinuousSpace<0> space1;
  parametricbem2d::ContinuousSpace<1> space2;
  Eigen::MatrixXd massnew =
      parametricbem2d::MassMatrix(parametrizedmesh, space1, space2, 32);
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
  Eigen::SparseMatrix<double> massold(4, 4);
  computeM01(massold, boundarymesh);
  Eigen::MatrixXd fullmass = massold * Eigen::MatrixXd::Identity(4, 4);
  // Finding error using matrix norm
  EXPECT_NEAR((fullmass - massnew).norm(), 0, eps);
}

TEST(ADNUMCSE_1_9, DISABLED_DirectFirstKind) {
  // Solving the Dirichlet BVP given in problem 1.9 in ADNUMCSE problems
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Initializing coefficients to construct the kite shaped curve in 1.9 using
  // fourier sum parametrization
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25, 0.1625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375, 0;
  // Parametrization of the kite given
  parametricbem2d::ParametrizedFourierSum kite(cos_list, sin_list, -M_PI, M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 30;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(kite.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Dirichlet trace given in problem 1.9
  std::function<double(double, double)> Td = [](double x1, double x2) {
    return sin(x1 - x2) * sinh(x1 + x2);
  };
  // Lambda function for known analytic Neumann Trace
  std::function<double(double)> Tn = [&](unsigned id) {
    // id denotes the panel id where neumann trace is to be calculated
    // Getting the point lying at midpoint of the parameter interval
    Eigen::Vector2d pt = panels[id]->operator()(0.);
    double x1 = pt(0);
    double x2 = pt(1);
    Eigen::Vector2d gradu;
    // Calculating the gradient of u
    gradu << cos(x1 - x2) * sinh(x1 + x2) + sin(x1 - x2) * cosh(x1 + x2),
        -cos(x1 - x2) * sinh(x1 + x2) + sin(x1 - x2) * cosh(x1 + x2);
    // Finding the tangent to evaluate the normal vector
    Eigen::Vector2d tangent = panels[id]->Derivative(0);
    Eigen::Vector2d normal;
    // Normal vector assuming that the curve is counter clockwise
    normal << -tangent(1), tangent(0);
    // This normal vector works
    normal = -normal;
    // Normalizing the normal vector
    normal /= normal.norm();
    // Returning the Neumann trace
    return gradu.dot(normal);
  };

  std::function<double(Eigen::VectorXd &, Eigen::VectorXd &, Eigen::VectorXd &)>
      TNu_pt =
          [&](Eigen::VectorXd &point, Eigen::VectorXd &a, Eigen::VectorXd &b) {
            Eigen::VectorXd grad_u(2); // gradient of u
            double X1 = point(0), X2 = point(1);
            grad_u << std::sin(X1 - X2) * std::cosh(X1 + X2) +
                          std::cos(X1 - X2) * std::sinh(X1 + X2),
                std::sin(X1 - X2) * std::cosh(X1 + X2) -
                    std::cos(X1 - X2) * std::sinh(X1 + X2);
            Eigen::VectorXd n(2); // normal vector
            Eigen::VectorXd temp = (b - a);
            n(0) = temp(1);
            n(1) = -temp(0);
            n /= n.norm();
            // n=-n;
            double traceval = grad_u.dot(n);
            return traceval;
          };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the dirichlet bvp using direct first kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh, Td, order);
  // Trial space used in the solver mentioned above, to evaluate V matrix for
  // error calculation
  DiscontinuousSpace<0> trial_space;
  // Evaluating the V matrix for calculating av error norm
  Eigen::MatrixXd V = single_layer::GalerkinMatrix(mesh, trial_space, order);
  // Vector to store the exact Neumann traces
  Eigen::VectorXd exTn(numpanels);
  for (int i = 0; i < numpanels; ++i) {
    exTn(i) = Tn(i);
    // Eigen::VectorXd
    // a=mesh.getVertex(i%numpanels),b=mesh.getVertex((i+1)%numpanels);
    // Eigen::VectorXd point =(a+b)/2;
    // exTn(i)=TNu_pt(point,a,b);
  }
  // Getting the error coefficients
  Eigen::VectorXd err_coeff = sol - exTn;
  std::cout << "error using av norm: "
            << err_coeff.transpose().dot(V * err_coeff) << std::endl;
}

TEST(ADNUMCSE_1_9, DISABLED_DirectSecondKind) {
  // Solving the Dirichlet BVP given in problem 1.9 in ADNUMCSE problems
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Initializing coefficients to construct the kite shaped curve in 1.9 using
  // fourier sum parametrization
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25, 0.1625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375, 0;
  // Parametrization of the kite given
  parametricbem2d::ParametrizedFourierSum kite(cos_list, sin_list, -M_PI, M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 10;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(kite.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Dirichlet trace given in problem 1.9
  std::function<double(double, double)> Td = [](double x1, double x2) {
    return sin(x1 - x2) * sinh(x1 + x2);
  };
  // Lambda function for known analytic Neumann Trace
  std::function<double(double)> Tn = [&](unsigned id) {
    // id denotes the panel id where neumann trace is to be calculated
    // Getting the point lying at midpoint of the parameter interval
    Eigen::Vector2d pt = panels[id]->operator()(0.);
    double x1 = pt(0);
    double x2 = pt(1);
    Eigen::Vector2d gradu;
    // Calculating the gradient of u
    gradu << cos(x1 - x2) * sinh(x1 + x2) + sin(x1 - x2) * cosh(x1 + x2),
        -cos(x1 - x2) * sinh(x1 + x2) + sin(x1 - x2) * cosh(x1 + x2);
    // Finding the tangent to evaluate the normal vector
    Eigen::Vector2d tangent = panels[id]->Derivative(0);
    Eigen::Vector2d normal;
    // Normal vector assuming that the curve is counter clockwise
    normal << -tangent(1), tangent(0);
    // This normal vector works
    normal = -normal;
    // Normalizing the normal vector
    normal /= normal.norm();
    // Returning the Neumann trace
    return gradu.dot(normal);
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the dirichlet bvp using direct second kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::dirichlet_bvp::direct_second_kind::solve(mesh, Td,
                                                                order);
  Eigen::VectorXd soldfk =
      parametricbem2d::dirichlet_bvp::direct_first_kind::solve(mesh, Td, order);

  std::cout << "dfk Tn: \n" << soldfk << std::endl;
  std::cout << "dsk Tn: \n" <<  sol << std::endl;

  // Same trial and test spaces
  DiscontinuousSpace<0> trial_space;
  Eigen::MatrixXd V = single_layer::GalerkinMatrix(mesh, trial_space, order);
  Eigen::VectorXd exTn(numpanels);
  for (int i = 0; i < numpanels; ++i) {
    exTn(i) = Tn(i);
  }
  /*
  Eigen::VectorXd err_coeff = sol - exTn;
  std::cout << "dsk error using av norm: "
            << err_coeff.transpose().dot(V * err_coeff) << std::endl;
  std::cout << "solution error with direct first kind " << (sol - soldfk).norm()
            << std::endl;*/
}

TEST(ADNUMCSE_1_9, DISABLED_IndirectFirstKind) {
  // Solving the Dirichlet BVP given in problem 1.9 in ADNUMCSE problems
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Initializing coefficients to construct the kite shaped curve in 1.9 using
  // fourier sum parametrization
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25, 0.1625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375, 0;
  // Parametrization of the kite given
  parametricbem2d::ParametrizedFourierSum kite(cos_list, sin_list, -M_PI, M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 50;
  // Getting the mesh
  parametricbem2d::ParametrizedMesh mesh(kite.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Dirichlet trace given in problem 1.9
  std::function<double(double, double)> Td = [](double x1, double x2) {
    return sin(x1 - x2) * sinh(x1 + x2);
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the dirichlet bvp using indirect first kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::dirichlet_bvp::indirect_first_kind::solve(mesh, Td,
                                                                 order);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0, 0.3;
  // Trial space used in the solver mentioned above
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Constructing the solution by calculating the single layer potential
  double u = single_layer::Potential(x, sol, mesh, trial_space, order);
  // Exact value
  double u_ex = Td(x(0), x(1));
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(ADNUMCSE_1_9, DISABLED_IndirectSecondKind) {
  // Solving the Dirichlet BVP given in problem 1.9 in ADNUMCSE problems
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Initializing coefficients to construct the kite shaped curve in 1.9 using
  // fourier sum parametrization
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25, 0.1625, 0, 0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375, 0;
  // Parametrization of the kite given
  parametricbem2d::ParametrizedFourierSum kite(cos_list, sin_list, -M_PI, M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 50;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(kite.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Dirichlet trace given in problem 1.9
  std::function<double(double, double)> Td = [](double x1, double x2) {
    return sin(x1 - x2) * sinh(x1 + x2);
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the dirichlet bvp using indirect second kind method in
  // parametricbem library
  Eigen::VectorXd sol =
      parametricbem2d::dirichlet_bvp::indirect_second_kind::solve(mesh, Td,
                                                                  order);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0, 0.3;
  // Trial space used in the solver mentioned above
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Constructing the solution by calculating the double layer potential
  double u = double_layer::Potential(x, sol, mesh, trial_space, order);
  // Exact value
  double u_ex = Td(x(0), x(1));
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(NEUMANNBVP, DISABLED_DirectFirstKind) {
  // Solving Neumann BVP on a unit disk. Exact u = r cos(phi)
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 80;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Neumann Trace
  std::function<double(double, double)> Tn = [&](double x1, double x2) {
    return x1/R; //cos(phi)
  };
  // Lambda function for Dirichlet Trace
  std::function<double(double, double)> Td = [&](double x, double y) {
    return x; // R cos(phi)
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the Neumann bvp using direct first kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::neumann_bvp::direct_first_kind::solve(mesh, Tn, order);
  // Getting exact dirichlet trace vector
  Eigen::VectorXd Td_ex(numpanels);
  for (unsigned i = 0; i < numpanels; ++i) {
    Eigen::Vector2d pt = mesh.getVertex(i);
    double x1 = pt(0);
    double x2 = pt(1);
    Td_ex(i) = Td(x1, x2);
  }
  Eigen::VectorXd error_coeffs = Td_ex - sol;
  std::cout << "error_coeffs.norm() " << error_coeffs.norm() << std::endl;
}

TEST(NEUMANNBVP, DISABLED_DirectSecondKind) {
  // Solving Neumann BVP on a unit disk. Exact u = r cos(phi)
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 80;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Neumann Trace
  std::function<double(double, double)> Tn = [&](double x1, double x2) {
    return x1/R; //cos(phi)
  };
  // Lambda function for Dirichlet Trace
  std::function<double(double, double)> Td = [&](double x, double y) {
    return x; // R cos(phi)
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the Neumann bvp using direct second kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::neumann_bvp::direct_second_kind::solve(mesh, Tn, order);
  // Getting exact dirichlet trace vector
  Eigen::VectorXd Td_ex(numpanels);
  for (unsigned i = 0; i < numpanels; ++i) {
    Eigen::Vector2d pt = panels[i]->operator()(0.);
    double x1 = pt(0);
    double x2 = pt(1);
    Td_ex(i) = Td(x1, x2);
  }

  Eigen::VectorXd error_coeffs = Td_ex - sol;
  std::cout << "error_coeffs.norm() " << error_coeffs.norm() << std::endl;
}

TEST(NEUMANNBVP, DISABLED_IndirectFirstKind) {
  // Solving Neumann BVP on a unit disk. Exact u = r cos(phi)
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 20;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Neumann Trace
  std::function<double(double, double)> Tn = [&](double x1, double x2) {
    return x1/R; //cos(phi)
  };
  // Lambda function for Dirichlet Trace
  std::function<double(double, double)> Td = [&](double x, double y) {
    return x; // R cos(phi)
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the Neumann bvp using indirect first kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::neumann_bvp::indirect_first_kind::solve(mesh, Tn, order);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0.3, 0;
  // Trial space used in the solver mentioned above
  parametricbem2d::ContinuousSpace<1> trial_space;
  // Constructing the solution by calculating the double layer potential
  double u = double_layer::Potential(x, sol, mesh, trial_space, order);
  // Exact value
  double u_ex = Td(x(0), x(1));
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(NEUMANNBVP, DISABLED_IndirectSecondKind) {
  // Solving Neumann BVP on a unit disk. Exact u = r cos(phi)
  using namespace parametricbem2d;
  using PanelVector = parametricbem2d::PanelVector;
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The number of panels for the mesh
  unsigned numpanels = 40;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  PanelVector panels = mesh.getPanels();
  // Lambda function for Neumann Trace
  std::function<double(double, double)> Tn = [&](double x1, double x2) {
    return x1/R; //cos(phi)
  };
  // Lambda function for Dirichlet Trace
  std::function<double(double, double)> Td = [&](double x, double y) {
    return x; // R cos(phi)
  };
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Solving the Neumann bvp using indirect first kind method in parametricbem
  // library
  Eigen::VectorXd sol =
      parametricbem2d::neumann_bvp::indirect_second_kind::solve(mesh, Tn,
                                                                order);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0.3, 0;
  // Trial space used in the solver mentioned above
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Constructing the solution by calculating the single layer potential
  double u = single_layer::Potential(x, sol, mesh, trial_space, order);
  // Exact value
  double u_ex = Td(x(0), x(1));
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(Potential, DISABLED_SingleLayer_0) {
  // Test for the Single Layer Potential function. Calculating the Single Layer
  // Potential on a circle with center (0,0) and radius R, for a constant
  // function phi. Comparing with the known analytic result.
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The constant function
  double phi = 1.;
  // The number of panels for the mesh
  unsigned numpanels = 20;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Vector for storing the coefficients required for calculation of the single
  // layer potential
  Eigen::VectorXd sol = Eigen::VectorXd::Constant(numpanels, phi);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0, 0;
  // Trial space for calculating the Single Layer Potential
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Constructing the solution by calculating the single layer potential
  double u = parametricbem2d::single_layer::Potential(x, sol, mesh, trial_space,
                                                      order);
  // Exact value
  double u_ex = -R * log(R);
  std::cout << "u,uex: " << u << " ," << u_ex << std::endl;
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(Potential, DISABLED_SingleLayer_1) {
  // Test for the Single Layer Potential function. Calculating the Single Layer
  // Potential on a circle with center (0,0) and radius R, for a constant
  // function phi. Comparing with the known analytic result.
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The constant function
  double phi = 1.;
  // The number of panels for the mesh
  unsigned numpanels = 20;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Vector for storing the coefficients required for calculation of the single
  // layer potential
  Eigen::VectorXd sol = Eigen::VectorXd::Constant(numpanels, phi);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0, 0;
  // Trial space for calculating the Single Layer Potential
  parametricbem2d::ContinuousSpace<1> trial_space;
  // Constructing the solution by calculating the single layer potential
  double u = parametricbem2d::single_layer::Potential(x, sol, mesh, trial_space,
                                                      order);
  // Exact value
  double u_ex = -R * log(R);
  std::cout << "u,uex: " << u << " ," << u_ex << std::endl;
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(Potential, DISABLED_DoubleLayer_0) {
  // Test for the Double Layer Potential function. Calculating the Double Layer
  // Potential on a circle with center (0,0) and radius R, for a constant
  // function phi. Comparing with the known analytic result.
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The constant function
  double phi = 1.;
  // The number of panels for the mesh
  unsigned numpanels = 40;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Vector for storing the coefficients required for calculation of the double
  // layer potential
  Eigen::VectorXd sol = Eigen::VectorXd::Constant(numpanels, phi);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0, 0;
  // Trial space used fir calculating the Double Layer Potential
  parametricbem2d::DiscontinuousSpace<0> trial_space;
  // Constructing the solution by calculating the single layer potential
  double u = parametricbem2d::double_layer::Potential(x, sol, mesh, trial_space,
                                                      order);
  // Exact value
  double u_ex = -1.;
  std::cout << "u,uex: " << u << " ," << u_ex << std::endl;
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

TEST(Potential, DISABLED_DoubleLayer_1) {
  // Test for the Double Layer Potential function. Calculating the Double Layer
  // Potential on a circle with center (0,0) and radius R, for a constant
  // function phi. Comparing with the known analytic result.
  // Center for the unit circle
  Eigen::Vector2d center;
  center << 0, 0;
  // Radius of the circle
  double R = 1.5;
  // Parametrization of unit circle centered at 0,0
  parametricbem2d::ParametrizedCircularArc curve(center, R, 0, 2 * M_PI);
  // The constant function
  double phi = 1.;
  // The number of panels for the mesh
  unsigned numpanels = 40;
  // Getting the mesh by splitting the fourier parametrization
  parametricbem2d::ParametrizedMesh mesh(curve.split(numpanels));
  // Order of quadrature to be used
  unsigned order = 16;
  std::cout << "numpanels, order = " << numpanels << " ," << order << std::endl;
  // Vector for storing the coefficients required for calculation of the double
  // layer potential
  Eigen::VectorXd sol = Eigen::VectorXd::Constant(numpanels, phi);
  // Evaluation point
  Eigen::Vector2d x;
  x << 0, 0;
  // Trial space used fir calculating the Double Layer Potential
  parametricbem2d::ContinuousSpace<1> trial_space;
  // Constructing the solution by calculating the single layer potential
  double u = parametricbem2d::double_layer::Potential(x, sol, mesh, trial_space,
                                                      order);
  // Exact value
  double u_ex = -1.;
  std::cout << "u,uex: " << u << " ," << u_ex << std::endl;
  std::cout << "error: " << fabs(u - u_ex) << std::endl;
}

/*TEST(StableEvaluationOfIntegrands, DISABLED_SingleLayer) {
  // Checking stable evaluation by using two very close but disjoint panels
  using PanelVector = parametricbem2d::PanelVector;
  double eps = 0.001;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 0, 1; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << eps, 0; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 2 * eps, 1; // Point (0,1.5)
  // Creating the two panels
  parametricbem2d::ParametrizedLine panel1(x1, x2);
  parametricbem2d::ParametrizedLine panel2(x3, x4);
  // BEM space used for calculating the interaction matrix
  parametricbem2d::DiscontinuousSpace<0> space;
  Eigen::MatrixXd new_lib =
      parametricbem2d::single_layer::ComputeIntegralGeneral(panel1, panel2,
                                                            space, 32);
  // Interaction Matrix using CppHilbert
  double old_lib = computeVij(x1, x2, x3, x4, 0.);
  std::cout << "exact: " << old_lib << std::endl;
  std::cout << "stable evaluation: " << new_lib << std::endl;
  // EXPECT_NEAR((sol_old - solnew).norm(), 0, eps);
}*/

/*TEST(StableEvaluationOfIntegrands, DISABLED_DoubleLayer) {
  // Checking stable evaluation by using two very close but disjoint panels
  using PanelVector = parametricbem2d::PanelVector;
  double eps = 0.001;
  // Corner points for the polygon
  Eigen::RowVectorXd x1(2);
  x1 << 0, 0; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 0, 1; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << eps, 0; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << 2 * eps, 1; // Point (0,1.5)
  // Creating the two panels
  parametricbem2d::ParametrizedLine panel1(x1, x2);
  parametricbem2d::ParametrizedLine panel2(x3, x4);
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::ContinuousSpace<1> trial_space;
  Eigen::MatrixXd new_lib =
      parametricbem2d::double_layer::ComputeIntegralGeneral(
          panel1, panel2, trial_space, test_space, 1000);
  // Interaction Matrix using CppHilbert
  double i0, i1;
  computeKij(&i0, &i1, 0., x1, x2, x3, x4);
  std::cout << "stable evaluation: " << new_lib << std::endl;
  std::cout << "exact: " << i0 - i1 << " " << i0 + i1 << std::endl;
  // EXPECT_NEAR((sol_old - solnew).norm(), 0, eps);
}*/

int main(int argc, char **argv) {
  srand(time(NULL));
  // run tests
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
