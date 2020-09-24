/**
 * \file continuous_space.hpp
 * \brief This file declares a templated class inherited from AbstractBEMSpace
 *        and represents discontinuous spaces of the form \f$S^{0}_{p}\f$
 *        as defined in \f$\eqref{eq:Sp}\f$, using full template specialization.
 *
 * This File is a part of the 2D-Parametric BEM package
 */

#ifndef CONTSPACEHPP
#define CONTSPACEHPP

#include "abstract_bem_space.hpp"

#include <utility>
#include <vector>

#include <Eigen/Dense>
#include "abstract_parametrized_curve.hpp"
#include "parametrized_mesh.hpp"

namespace parametricbem2d {
/**
 * \class ContinuousSpace
 * \brief This templated class inherits from the class AbstractBEMSpace
 *        . This class implements the BEM spaces of the form \f$S^{0}_{p}\f$
 *        defined in \f$\eqref{eq:Sp}\f$. The class is implemented through full
 * template specialization for different values of p.
 */
template <unsigned int p> class ContinuousSpace : public AbstractBEMSpace {
public:
  ContinuousSpace() {
    std::cout << "Error! No specialization defined for p = " << p << std::endl;
  }
};

/**
 * \brief This is a specialization of the templated class for p = 1.
 *        This class represents the space \f$S^{0}_{1}\f$
 */
template <> class ContinuousSpace<1> : public AbstractBEMSpace {
public:
  // Local to Global Map
  unsigned int LocGlobMap(unsigned int q, unsigned int n,
                          unsigned int N) const {
    // Asserting the index of local shape function and the panel number are
    // within limits
    assert(q <= q_ && n <= N);
    if (q == 2)
      return n;
    else
      return (n % N == 0) ? 1 : (n + 1);
  }

  // Space Dimensions as defined in \f$\ref{T:thm:dimbe}\f$
  unsigned int getSpaceDim(unsigned int numpanels) const {
    return numpanels * (q_ - 1);
  }

  // Function for interpolating the input function
  Eigen::VectorXd Interpolate(const std::function<double(double, double)> &func,
                              const ParametrizedMesh &mesh) const {
    // The output vector
    unsigned numpanels = mesh.getNumPanels();
    unsigned coeffs_size = getSpaceDim(numpanels);
    Eigen::VectorXd coeffs(coeffs_size);
    // using PanelVector = parametricbem2d::PanelVector;
    // Filling the coefficients
    for (unsigned i = 0; i < coeffs_size; ++i) {
      Eigen::Vector2d pt = mesh.getVertex(i);
      coeffs(i) = func(pt(0), pt(1));
    }
    return coeffs;
  }

  // Constructor
  ContinuousSpace() {
    // Number of reference shape functions for the space
    q_ = 2;
    // Reference shape function 1, defined using a lambda expression
    BasisFunctionType b1 = [&](double t) { return 0.5 * (t + 1); };
    // Reference shape function 2, defined using a lambda expression
    BasisFunctionType b2 = [&](double t) { return 0.5 * (1 - t); };
    // Adding the reference shape functions to the vector
    referenceshapefunctions_.push_back(b1);
    referenceshapefunctions_.push_back(b2);
    // Reference shape function 1 derivative, defined using a lambda expression
    BasisFunctionType b1dot = [&](double t) { return 0.5; };
    // Reference shape function 2 derivative, defined using a lambda expression
    BasisFunctionType b2dot = [&](double t) { return -0.5; };
    // Adding the reference shape function derivatives to the vector
    referenceshapefunctiondots_.push_back(b1dot);
    referenceshapefunctiondots_.push_back(b2dot);
  }
};

/**
 * \brief This is a specialization of the templated class for p = 2.
 *        This class represents the space \f$S^{0}_{2}\f$
 */
template <> class ContinuousSpace<2> : public AbstractBEMSpace {
public:
  unsigned int LocGlobMap(unsigned int q, unsigned int n,
                          unsigned int N) const {
    // Asserting the index of local shape function and the panel number are
    // within limits
    assert(q <= q_ && n <= N);
    if (q == 2)
      return n;
    else if (q == 1)
      return (n % N == 0) ? 1 : (n + 1);
    else
      return N + n;
  }

  // Space Dimensions as defined in \f$\ref{T:thm:dimbe}\f$
  unsigned int getSpaceDim(unsigned int numpanels) const {
    return numpanels * (q_ - 1);
  }

  // Function for interpolating the input function
  Eigen::VectorXd Interpolate(const std::function<double(double, double)> &func,
                              const ParametrizedMesh &mesh) const {
    // The output vector
    unsigned numpanels = mesh.getNumPanels();
    unsigned coeffs_size = getSpaceDim(numpanels);
    Eigen::VectorXd coeffs(coeffs_size);
    // using PanelVector = parametricbem2d::PanelVector;
    // Filling the coefficients
    for (unsigned i = 0; i < coeffs_size; ++i) {
      // TODO
      coeffs(i) = 0;
    }
    return coeffs;
  }

  // Constructor
  ContinuousSpace() {
    // Number of reference shape functions for the space
    q_ = 3;
    // Reference shape function 1, defined using a lambda expression
    BasisFunctionType b1 = [&](double t) { return 0.5 * (t + 1); };
    // Reference shape function 2, defined using a lambda expression
    BasisFunctionType b2 = [&](double t) { return 0.5 * (1 - t); };
    // Reference shape function 3, defined using a lambda expression
    BasisFunctionType b3 = [&](double t) { return 1 - t * t; };
    // Adding the reference shape functions to the vector
    referenceshapefunctions_.push_back(b1);
    referenceshapefunctions_.push_back(b2);
    referenceshapefunctions_.push_back(b3);
    // Reference shape function 1 derivative, defined using a lambda expression
    BasisFunctionType b1dot = [&](double t) { return 0.5; };
    // Reference shape function 2 derivative, defined using a lambda expression
    BasisFunctionType b2dot = [&](double t) { return -0.5; };
    // Reference shape function 3 derivative, defined using a lambda expression
    BasisFunctionType b3dot = [&](double t) { return -2 * t; };
    // Adding the reference shape function derivatives to the vector
    referenceshapefunctiondots_.push_back(b1dot);
    referenceshapefunctiondots_.push_back(b2dot);
    referenceshapefunctiondots_.push_back(b3dot);
  }
};
} // namespace parametricbem2d

#endif // CONTSPACEHPP
