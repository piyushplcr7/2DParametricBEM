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

BoundaryMesh createfrom(const parametricbem2d::ParametrizedMesh& pmesh) {

  unsigned nV = pmesh.getNumPanels();
  Eigen::MatrixXd coords(nV,2);
  Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic> elems(nV,2);
  for (unsigned i = 0 ; i < nV ; ++i) {
    coords.row(i) = pmesh.getVertex(i);
    elems(i,0) = i;
    elems(i,1) = (i+1)%nV;
  }
  BoundaryMesh bmesh(coords,elems);
  return bmesh;
}

parametricbem2d::ParametrizedMesh
convert_to_linear(const parametricbem2d::ParametrizedMesh &pmesh) {
  unsigned N = pmesh.getNumPanels();
  parametricbem2d::PanelVector lmesh;
  for (unsigned i = 0; i < N; ++i) {
    // Making linear panels using the end points of the original panel
    parametricbem2d::ParametrizedLine lpanel(pmesh.getVertex(i),
                                             pmesh.getVertex((i + 1) % N));
    parametricbem2d::PanelVector tmp = lpanel.split(1);
    lmesh.insert(lmesh.end(), tmp.begin(), tmp.end());
  }
  parametricbem2d::ParametrizedMesh plmesh(lmesh);
  return plmesh;
}

int main() {
  std::string filename = "convergence.txt";
  //std::string filename = "5.txt";
  std::ofstream output(filename);
  output << std::setw(15) << "#order" << std::setw(15) << "error sl" << std::setw(15) << "error dl" << std::setw(15) << "error hyp" << std::endl;
  // Convergence test for Single Layer Galerkin Matrix
  // Coefficients for cosine terms
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25,0.1625,0,0;
  //cos_list << 0.5,0,0,0;
  // Coefficients for sine terms
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375,0;
  //sin_list << 0, 0, 0.5,0;
  //sin_list << 0, 1;
  // Parametrized curve for kite shaped boundary
  parametricbem2d::ParametrizedFourierSum curve(Eigen::Vector2d(0,0),cos_list, sin_list, 0,
                                                2 * M_PI);
  /*double a = 0.01; // Side of the square
  using PanelVector = parametricbem2d::PanelVector;
  // Corner points for the square
  Eigen::RowVectorXd x1(2);
  x1 << -a/2, -a/2; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << a/2, -a/2; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << a/2, a/2; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << -a/2, a/2; // Point (0,1.5)
  /*Eigen::RowVectorXd x1(2);
  x1 << 0, -1.5; // Point (0,0)
  Eigen::RowVectorXd x2(2);
  x2 << 1, 0.5; // Point (1,0)
  Eigen::RowVectorXd x3(2);
  x3 << 0.1, .33; // Point (1,0.5)
  Eigen::RowVectorXd x4(2);
  x4 << -1, -.1; // Point (0,1.5)*/
  // Parametrized line segments forming the edges of the polygon
  /*parametricbem2d::ParametrizedLine line1(x1, x2);
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
  PanelVector square;
  // Storing all the panels in order so that they form a polygon
  square.insert(square.end(), line1panels.begin(), line1panels.end());
  square.insert(square.end(), line2panels.begin(), line2panels.end());
  square.insert(square.end(), line3panels.begin(), line3panels.end());
  square.insert(square.end(), line4panels.begin(), line4panels.end());

  // Constructing a triangle
  Eigen::RowVectorXd X1(2);
  X1 << 0.5,0; // Point (0,0)
  Eigen::RowVectorXd X2(2);
  X2 << 0,std::sqrt(3)/2; // Point (1,0)
  Eigen::RowVectorXd X3(2);
  X3 << -0.5,0; // Point (1,0.5)
  parametricbem2d::ParametrizedLine edge1(X1, X2);
  parametricbem2d::ParametrizedLine edge2(X2, X3);
  parametricbem2d::ParametrizedLine edge3(X3, X1);
  PanelVector e1panels = edge1.split(1);
  PanelVector e2panels = edge2.split(1);
  PanelVector e3panels = edge3.split(1);
  PanelVector triangle;
  // Storing all the panels in order so that they form a polygon
  triangle.insert(triangle.end(), e1panels.begin(), e1panels.end());
  triangle.insert(triangle.end(), e2panels.begin(), e2panels.end());
  triangle.insert(triangle.end(), e3panels.begin(), e3panels.end());
  // Construction of a ParametrizedMesh object from the vector of panels
  parametricbem2d::ParametrizedMesh parametrizedmesh(curve.split(5));
  //parametricbem2d::ParametrizedMesh parametrizedmesh(panels);*/
  // BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> space;
  // Test BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::DiscontinuousSpace<0> test_space;
  // Trial BEM space to be used for computing the Galerkin Matrix
  parametricbem2d::ContinuousSpace<1> trial_space;
  // Matrix to store Vertices/Corners of panels in the mesh to compute Galerkin
  // Matrix using CppHilbert
  /*Eigen::MatrixXd coords(4, 2);
  coords << x1, x2, x3, x4;
  // Matrix to store the end points of elements/edges of the panels in our mesh
  // used to compute Galerkin Matrix using CppHilbert
  Eigen::Matrix<int, 4, 2> elems;
  elems << 0, 1, 1, 2, 2, 3, 3, 0;
  // Creating a boundarymesh object used in CppHilbert library
  BoundaryMesh boundarymesh(coords, elems);*/
  // Galerkin Matrix computed using CppHilbert
  /*Eigen::MatrixXd final_sl =
      parametricbem2d::single_layer::GalerkinMatrix(parametrizedmesh, space,
                                                       256);
  Eigen::MatrixXd fcoinciding = final_sl.diagonal().asDiagonal();

  Eigen::MatrixXd fadjacent = Eigen::MatrixXd::Zero(5,5);
  fadjacent.diagonal(+1) = final_sl.diagonal(1);
  fadjacent.diagonal(-1) = final_sl.diagonal(-1);
  fadjacent.diagonal(+4) = final_sl.diagonal(4);
  fadjacent.diagonal(-4) = final_sl.diagonal(-4);

  Eigen::MatrixXd farbitrary(final_sl);
  farbitrary -= fadjacent+fcoinciding;*/
  //Eigen::MatrixXd final_sl;
  //computeV(final_sl, boundarymesh, 0);

  //Eigen::MatrixXd final_dl = parametricbem2d::double_layer::GalerkinMatrix(
  //    parametrizedmesh, trial_space, test_space, 256);
  //Eigen::MatrixXd final_dl;
  //computeK(final_dl, boundarymesh, 0);

  //Eigen::MatrixXd final_hyp = parametricbem2d::hypersingular::GalerkinMatrix(
  //    parametrizedmesh, trial_space, 256);
  //Eigen::MatrixXd final_hyp;
  //computeW(final_hyp, boundarymesh, 0);
  //std::cout << "computed overkill!" << std::endl;
  /*for (unsigned order = 2 ; order <= 100 ; order++) {
    Eigen::MatrixXd sl =
        parametricbem2d::single_layer::GalerkinMatrix(parametrizedmesh, space,
                                                         order);

     Eigen::MatrixXd coinciding = sl.diagonal().asDiagonal();

     Eigen::MatrixXd adjacent = Eigen::MatrixXd::Zero(5,5);
     adjacent.diagonal(+1) = sl.diagonal(1);
     adjacent.diagonal(-1) = sl.diagonal(-1);
     adjacent.diagonal(+4) = sl.diagonal(4);
     adjacent.diagonal(-4) = sl.diagonal(-4);

     Eigen::MatrixXd arbitrary(final_sl);
     arbitrary -= adjacent+coinciding;

    double err_sl = (fcoinciding-coinciding).norm()/fcoinciding.norm();//(final_sl-sl).norm()/final_sl.norm();

    //Eigen::MatrixXd dl = parametricbem2d::double_layer::GalerkinMatrix(
    //    parametrizedmesh, trial_space, test_space, order);
    double err_dl = (fadjacent-adjacent).norm()/fadjacent.norm();//(final_dl-dl).norm()/final_dl.norm();

    //Eigen::MatrixXd hyp = parametricbem2d::hypersingular::GalerkinMatrix(
    //    parametrizedmesh, trial_space, order);
    double err_hyp = (farbitrary-arbitrary).norm()/farbitrary.norm();//(final_hyp-hyp).norm()/final_hyp.norm();
    output << std::setw(15) << order << std::setw(15) << err_sl << std::setw(15) << err_dl << std::setw(15) << err_hyp << std::endl;
    std::cout << std::setw(15) << order << std::setw(15) << err_sl << std::setw(15) << err_dl << std::setw(15) << err_hyp << std::endl;

  }*/

  for (unsigned numpanels = 2 ; numpanels <= 500 ; numpanels+=10) {
    parametricbem2d::ParametrizedMesh pmesh(curve.split(numpanels));
    parametricbem2d::ParametrizedMesh plmesh = convert_to_linear(pmesh);
    BoundaryMesh bmesh = createfrom(pmesh);

    Eigen::MatrixXd sl = parametricbem2d::single_layer::GalerkinMatrix(plmesh, space, 16);
    Eigen::MatrixXd slcpp;

    Eigen::MatrixXd dl = parametricbem2d::double_layer::GalerkinMatrix(
        plmesh, trial_space, test_space, 16);
    Eigen::MatrixXd dlcpp;

    //Eigen::MatrixXd hyp = parametricbem2d::hypersingular::GalerkinMatrix(
    //    plmesh, trial_space, 16);
    //Eigen::MatrixXd hypcpp;

    Eigen::MatrixXd M00 = parametricbem2d::MassMatrix(plmesh,space,trial_space,16);
    Eigen::SparseMatrix<double> M00cpp(numpanels,numpanels);

    computeV(slcpp,bmesh,0);
    computeK(dlcpp,bmesh,0);
    //computeW(hypcpp,bmesh,0);
    computeM01(M00cpp,bmesh);

    double err_sl = (sl-slcpp).norm()/slcpp.norm();
    double err_dl = (dl-dlcpp).norm()/dlcpp.norm();
    double err_hyp = (M00-M00cpp).norm()/M00cpp.norm();//(hyp-hypcpp).norm()/hypcpp.norm();
    //double err_m00 = (M00-M00cpp).norm()/M00cpp.norm();
    output << std::setw(15) << numpanels << std::setw(15) << err_sl << std::setw(15) << err_dl << std::setw(15) << err_hyp << std::endl;
    std::cout << std::setw(15) << numpanels << std::setw(15) << err_sl << std::setw(15) << err_dl << std::setw(15) << err_hyp << std::endl;
    //output << std::setw(15) << numpanels << std::setw(15) << err_m00 << std::endl;
    //std::cout << std::setw(15) << numpanels << std::setw(15) << err_m00 << std::endl;

  }
  output.close();
  return 0;
}
