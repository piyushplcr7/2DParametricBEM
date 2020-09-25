#include "integral_gauss.hpp"
#include "logweight_quadrature.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <string>
#include <fstream>
#include <Eigen/Dense>

using namespace parametricbem2d;

int main() {
  std::string filename = "convergence_panels2.txt";
  std::ofstream output(filename);
  output << std::setw(15) << "#order" << std::setw(15) << "error" << std::endl;
  double a = 1;
  unsigned deg = 32;
  Eigen::VectorXd coeffs = Eigen::VectorXd::Random(deg);

  auto F = [&](double spr) {
    //return std::sqrt(1+std::pow(spr,2));
    //return (2+std::sin(spr));
    return std::sin(spr);
    //return 1.;
    //return std::sqrt(1+std::pow(0.05*spr,2));
    //return std::pow(spr,deg);
    double polynomial = 0.;
    for (unsigned i = 0 ; i < deg ; ++i) {
      polynomial += coeffs(i) * std::pow(spr,i);
    }
    //return polynomial;
  };

  auto integrand = [&] (double s) {
    //double spr = s;
    //double spr = 2*s/a-1;
    //return s * F(spr);
    return F(s);
    //return F(std::exp(-s));
  };

  bool logweighted = true;
  bool laguerre = !true;
  bool simple = !true;
  double overkill;
  unsigned oorder = 100; QuadRule gqr = getGaussQR(oorder);

  if (logweighted) {
    //overkill = -1.05149769058874936228882;
    overkill = ComputeLoogIntegral(integrand,oorder);
    //overkill = ComputeLoogIntegral(integrand,a,gqr);
    std::cout << "logweighted!" << std::endl;
  }
  if (simple) {
    overkill = ComputeIntegral(integrand,0,a,oorder);
    std::cout << "simple!" << std::endl;
  }
  if (laguerre) {
    overkill = ComputeLaguerreIntegral(integrand,oorder);
    std::cout << "laguerre!" << std::endl;
  }

  if(!true) {
    overkill = 0.;
    for (unsigned i = 0 ; i < deg ; ++i)
      overkill -= coeffs(i) / std::pow(i+1,2);
  }

  if(!true) {
    overkill = 1.;
    for (unsigned i = 1 ; i <= deg ; ++i)
      overkill *= i;
  }
std::cout << overkill << std::endl;
for (unsigned order = 2 ; order <= oorder ; ++order) {
  QuadRule qr = getGaussQR(order);
  double val;
  if (logweighted)
    //val = ComputeLogIntegral(integrand,a,order);
    //val = ComputeLogwtIntegral(integrand,2,qr);
    //val = ComputeLoogIntegral(integrand,a,qr);
    val = ComputeLoogIntegral(integrand,order);
  if (simple)
    val = ComputeIntegral(integrand,0,a,order);
  if (laguerre)
    val = ComputeLaguerreIntegral(integrand,order);

  std::cout << std::setw(15) << order << std::setw(15) << val  << std::endl;
  output << std::setw(15) << order << std::setw(15) << fabs(val-overkill)/fabs(overkill) << std::endl;
}
test::init();
std::vector<double> testvector = test::returntestvec();
std::cout << testvector[1] << std::endl;
return 0;
}
