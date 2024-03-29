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

#include <exception>
#include <utility>
#include <vector>

#include "abstract_parametrized_curve.hpp"
#include "parametrized_mesh.hpp"
#include <Eigen/Dense>

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
    throw std::invalid_argument("Class specialization not defined!");
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
    if (!(q <= q_ && n <= N)) {
      throw std::out_of_range("Panel/RSF index out of range!");
    }

    if (q == 2)
      return n;
    else
      return (n % N == 0) ? 1 : (n + 1);
  }

  // Local to Global Map
  unsigned int LocGlobMap2(unsigned int q, unsigned int n,
                           const ParametrizedMesh &mesh) const {
    // Getting the mesh information
    unsigned split = mesh.getSplit();
    unsigned N = mesh.getNumPanels();
    // Mapping depends whether domain is annular or not
    if (split != 0) {
      if (n <= split) // we are within the first boundary in the mesh
        return LocGlobMap(q, n, split);
      else // for the second boundary in the mesh
        return LocGlobMap(q, n - split, N - split) + split;
    } else { // simply connected domain
      return LocGlobMap(q, n, N);
    }
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
    if (!(q <= q_ && n <= N)) {
      throw std::out_of_range("Panel/RSF index out of range!");
    }

    if (q == 2)
      return n;
    else if (q == 1)
      return (n % N == 0) ? 1 : (n + 1);
    else
      return N + n;
  }

  // Local to Global Map
  unsigned int LocGlobMap2(unsigned int q, unsigned int n,
                           const ParametrizedMesh &mesh) const {
    // Getting the mesh information
    unsigned split = mesh.getSplit();
    unsigned N = mesh.getNumPanels();
    // Mapping depends whether domain is annular or not
    if (split != 0) {
      if (n <= split) // we are within the first boundary in the mesh
        return LocGlobMap(q, n, split);
      else // for the second boundary in the mesh
        return LocGlobMap(q, n - split, N - split) + split;
    } else { // simply connected domain
      return LocGlobMap(q, n, N);
    }
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
    PanelVector panels = mesh.getPanels();
    // Filling the coefficients
    for (unsigned i = 0; i < numpanels; ++i) {
      Eigen::Vector2d lvertex = mesh.getVertex(i);
      Eigen::Vector2d rvertex = mesh.getVertex((i + 1) % numpanels);
      coeffs(i) = func(lvertex(0), lvertex(1));
      Eigen::Vector2d cvertex = panels[i]->operator()(0);
      coeffs(numpanels + i) = func(cvertex(0), cvertex(1)) -
                              0.5 * (coeffs(i) + func(rvertex(0), rvertex(1)));
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
