#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <utility>

#include "BoundaryMesh.hpp"
#include "abstract_bem_space.hpp"
#include "buildK.hpp"
#include "buildM.hpp"
#include "buildV.hpp"
#include "buildW.hpp"
#include "continuous_space.hpp"
#include "dirichlet.hpp"
#include "discontinuous_space.hpp"
#include "doubleLayerPotential.hpp"
#include "double_layer.hpp"
#include "hypersingular.hpp"
#include "integral_gauss.hpp"
#include "neumann.hpp"
#include "parametrized_circular_arc.hpp"
#include "parametrized_fourier_sum.hpp"
#include "parametrized_line.hpp"
#include "parametrized_mesh.hpp"
#include "parametrized_polynomial.hpp"
#include "parametrized_semi_circle.hpp"
#include "singleLayerPotential.hpp"
#include "single_layer.hpp"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>

int main() {
  // std::string filename = "convergence_panels.txt";
  std::string filename = "convergence_panels2.txt";
  std::ofstream output(filename);
  // output << std::setw(15) << "#order" << std::setw(15) << "error sl" <<
  // std::endl;
  output << std::setw(15) << "#order" << std::setw(15) << "error1"
         << std::setw(15) << "error2" << std::endl;

  // arbitrary lines
  /*Eigen::RowVectorXd x1(2);
  x1 << 1, 1;
  Eigen::RowVectorXd x2(2);
  x2 << 3, 4;
  Eigen::RowVectorXd x3(2);
  x3 << 1.5, 0.5;
  Eigen::RowVectorXd x4(2);
  x4 << 5, 2.1;
  parametricbem2d::ParametrizedLine pi(x1,x2);
  parametricbem2d::ParametrizedLine pi_p(x3,x4);*/

  // coinciding lines
  /*Eigen::RowVectorXd x1(2);
  x1 << 1.5, 0.5;
  Eigen::RowVectorXd x2(2);
  x2 << 5, 2.1;
  parametricbem2d::ParametrizedLine pi(x1,x2);*/

  // Adjacent Lines
  /*Eigen::RowVectorXd x1(2);
  x1 << 1, 1;
  Eigen::RowVectorXd x2(2);
  x2 << 3, 4;
  Eigen::RowVectorXd x3(2);
  x3 << 3, 4;
  Eigen::RowVectorXd x4(2);
  x4 << 2, 10;
  parametricbem2d::ParametrizedLine pi(x1,x2);
  parametricbem2d::ParametrizedLine pi_p(x3,x4);*/

  // Quadratic parametrizations
  // Arbitrary
  /*Eigen::MatrixXd coeffs1(2,3);
  coeffs1 << 0,1,0,1,0,1;
  parametricbem2d::ParametrizedPolynomial pi(coeffs1);
  Eigen::MatrixXd coeffs2(2,3);
  coeffs2 << 1,1,0,1.15,1,-1;
  parametricbem2d::ParametrizedPolynomial pi_p(coeffs2);*/

  // Coinciding case
  /*Eigen::MatrixXd coeffs1(2,3);
  coeffs1 << 0,1,0,1,0,1;
  parametricbem2d::ParametrizedPolynomial pi(coeffs1);*/

  // Adjacent case
  /*Eigen::MatrixXd coeffs1(2, 3);
  coeffs1 << 0, 1, 0, 1, 0, 1;
  parametricbem2d::ParametrizedPolynomial pi(coeffs1);
  Eigen::MatrixXd coeffs2(2, 3);
  coeffs2 << 2, 1, 0, 5, 4, 1;
  parametricbem2d::ParametrizedPolynomial pi_p(coeffs2);*/

  // Fourier parametrizations
  // Arbitrary
  double step = 2*M_PI/5;
  Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25,0.1625,0,0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375,0;
  parametricbem2d::ParametrizedFourierSum pi(Eigen::Vector2d(0,0),cos_list, sin_list, 0, step);//M_PI);
  Eigen::MatrixXd cos_lis(2, 2);
  //cos_lis << 1,0.1,0,0;
  cos_lis << 0.25,0.1625,0,0;
  Eigen::MatrixXd sin_lis(2, 2);
  //sin_lis << 0, 0, 0.5,0.1;
  sin_lis << 0, 0, 0.375,0;
  parametricbem2d::ParametrizedFourierSum pi_p(Eigen::Vector2d(0,0),cos_lis, sin_lis, 2*step,3*step);//0.1, M_PI-0.1);

  // Coinciding
  /*Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25,0.1625,0,0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375,0;
  parametricbem2d::ParametrizedFourierSum pi(Eigen::Vector2d(0,0),cos_list, sin_list, 1, 2);*/

  // Adjacent
  /*Eigen::MatrixXd cos_list(2, 2);
  cos_list << 0.25,0.1625,0,0;
  Eigen::MatrixXd sin_list(2, 2);
  sin_list << 0, 0, 0.375,0;
  parametricbem2d::ParametrizedFourierSum pi(Eigen::Vector2d(0,0),cos_list, sin_list, 1, 2);
  Eigen::MatrixXd cos_lis(2, 2);
  cos_lis << 0.25,0.1625,0,0;
  Eigen::MatrixXd sin_lis(2, 2);
  sin_lis << 0, 0, 0.375,0;
  parametricbem2d::ParametrizedFourierSum pi_p(Eigen::Vector2d(0,0),cos_lis, sin_lis, 2, M_PI);*/

  // Circular panels
  // arbitrary
  /*parametricbem2d::ParametrizedCircularArc
  pi(Eigen::Vector2d(0,0),1.,-M_PI/2,M_PI/2);
  parametricbem2d::ParametrizedCircularArc
  pi_p(Eigen::Vector2d(3,0),1.,M_PI/4,0.77*M_PI);*/

  // Coinciding
  // parametricbem2d::ParametrizedCircularArc
  // pi(Eigen::Vector2d(3,2),1.,M_PI*0.169,0.77*M_PI);

  // Adjacent
  // parametricbem2d::ParametrizedCircularArc pi(Eigen::Vector2d(3,0),1.,0,1.7);
  // parametricbem2d::ParametrizedCircularArc
  // pi_p(Eigen::Vector2d(3,0),1.,1.7,3.33);

  parametricbem2d::DiscontinuousSpace<0> space;
  QuadRule gauss = getGaussQR(256); // Overkill quadrature rule
                                    // Galerkin Matrix computed using CppHilbert
                                    /*double final_val =
                                        parametricbem2d::single_layer::InteractionMatrix(pi, pi_p, space,
                                       gauss)(0,0);*/
  /*Eigen::MatrixXd test_matrix =
      parametricbem2d::single_layer::InteractionMatrix(pi, pi_p, space, gauss);
  double final_val4 = test_matrix(3, 0);
  double final_val3 = test_matrix(2, 0);
  double final_val2 = test_matrix(1, 0);
  double final_val1 = test_matrix(0, 0);*/
   double final_val =
      parametricbem2d::single_layer::InteractionMatrix(pi, pi_p, space,
      gauss)(0,0);

  std::cout << "overkill evaluated" << std::endl;

  for (unsigned order = 2; order <= 100; order+=1) {
    QuadRule quad = getGaussQR(order);
    double val =
        parametricbem2d::single_layer::InteractionMatrix(pi, pi_p, space,
       quad)(0,0);
    /*Eigen::MatrixXd lol =
        parametricbem2d::single_layer::InteractionMatrix(pi, pi_p, space, quad);
    double val4 = lol(3, 0);
    double val3 = lol(2, 0);
    double val2 = lol(1, 0);
    double val1 = lol(0, 0);*/

    // double val =
    //    parametricbem2d::single_layer::InteractionMatrix(pi, pi, space,
    //    quad)(0,0);

    // double err_sl = fabs((final_val-val)/final_val);
     double err_sl = fabs((final_val-val));
     output << std::setw(15) << order << std::setw(15) << err_sl << std::endl;
     std::cout << std::setw(15) << order << std::setw(15) << val <<
     std::endl;

    /*output << std::setw(15) << order << std::setw(15) << fabs(final_val1 - val1)
           << std::setw(15) << fabs(final_val2 - val2) << std::setw(15)
           << fabs(final_val3 - val3) << std::setw(15)
           << fabs(final_val4 - val4) << std::endl;
    std::cout << std::setw(15) << order << std::setw(15)
              << fabs(final_val1 - val1) << std::setw(15)
              << fabs(final_val2 - val2) << std::setw(15)
              << fabs(final_val3 - val3) << std::setw(15)
              << fabs(final_val4 - val4) << std::endl;*/
  }
  output.close();
  return 0;
}
