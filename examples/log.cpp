#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/Dense>
#include "integral_gauss.hpp"

int main() {
  unsigned int num_pts = 100;
  // File containing Analytic Values
  std::string filename1 = "log_functions.txt";
  // File containing integration errors
  std::string filename2 = "log_integrals.txt";
  std::ofstream output1(filename1);
  std::ofstream output2(filename2);
  output1 << std::setw(15) << "#y1[-1,1]" << std::setw(15) << "fy1"
          << std::setw(15) << "y2[1,2]" << std::setw(15) << "fy2" << std::endl;
  output2 << std::setw(15) << "#order" << std::setw(15) << "error1"
          << std::setw(15) << "error2" << std::endl;
  Eigen::VectorXd y1 = Eigen::VectorXd::LinSpaced(num_pts, -1, 1);
  Eigen::VectorXd y2 = Eigen::VectorXd::LinSpaced(num_pts, 1, 2);
  Eigen::VectorXd fy1_computed(num_pts);
  Eigen::VectorXd fy2_computed(num_pts);
  Eigen::VectorXd fy1_analytic(num_pts);
  Eigen::VectorXd fy2_analytic(num_pts);
  // integral as a function of y for the range [-1,1]
  std::function<double(double)> f1 = [](double y) {
    if (y == -1 || y == 1)
      return 2 * (log(2) - 1);
    else
      return ((1 - y) * (log(1 - y) - 1) + (1 + y) * (log(1 + y) - 1));
  };
  // integral as a function of y for the range [1,2]
  std::function<double(double)> f2 = [](double y) {
    if (y == 1)
      return 2 * (log(2) - 1);
    else
      return ((1 - y) * (log(y - 1) - 1) + (1 + y) * (log(1 + y) - 1));
  };
  // Exact integral values computed analytically
  double int_f1_exact = 4 * log(2) - 6;
  double int_f2_exact = 4.5 * log(3) - 2 * log(2) - 3;
  // evalutating functions
  for (unsigned int i = 0; i < num_pts; ++i) {
    fy1_analytic(i) = f1(y1(i));
    fy2_analytic(i) = f2(y2(i));
    output1 << std::setw(15) << y1(i) << std::setw(15) << fy1_analytic(i)
            << std::setw(15) << y2(i) << std::setw(15) << fy2_analytic(i)
            << std::endl;
  }

  // evaluating the errors for quadrature order
  for (unsigned int order = 2; order < 500; order += 2) {
    double int_1 = parametricbem2d::ComputeIntegral(f1, -1, 1, order);
    double int_2 = parametricbem2d::ComputeIntegral(f2, 1, 2, order);
    output2 << std::setw(15) << order << std::setw(15)
            << fabs(int_1 - int_f1_exact) << std::setw(15)
            << fabs(int_2 - int_f2_exact) << std::setw(15) << std::endl;
  }
  return 0;
}
