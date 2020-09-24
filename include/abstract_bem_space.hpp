/**
 * \file abstract_bem_space.hpp
 * \brief This file declares an abstract class representing a BEM Space
 *
 * This File is a part of the 2D-Parametric BEM package
 */

#ifndef ABSTRACTBEMSPACEHPP
#define ABSTRACTBEMSPACEHPP

#include <cassert>
#include <functional>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include "parametrized_mesh.hpp"

namespace parametricbem2d {
/**
 * \class AbstractBEMSpace
 * \brief This abstract class declares the interface for a BEM Space
 */
class AbstractBEMSpace {
public:
  /**
   * This function maps a local shape function to the corresponding global shape
   * function for a BEM space on the given number of panels, defined in
   * \f$\eqref{eq:lgm}\f$. This is a pure virtual function and has to be
   * implemented in the derived classes.
   *
   * @param q Index of the local/reference shape function
   * @param n Index of the panel for which this map is applied
   * @param N Total number of panels in the mesh
   * @return Integer denoting the global shape function number corresponding
   *         to the local shape function indexed by q for the panel no. n
   */
  virtual unsigned int LocGlobMap(unsigned int q, unsigned int n,
                                  unsigned int N) const = 0;

  /**
   * This function is used for querying the parameter interval.
   * The interval is fixed to be [-1,1] by declaring it as a non
   * virtual function, preventing overriding of this function.
   * The function is declared static as it is independent
   * of the concrete object.
   *
   * @return A std::pair<double,double> object containing
   *          the valid parameter range for parametrization
   *          that is [-1,1]
   */
  static std::pair<double, double> ParameterRange(void) {
    // Parameter range : [-1,1]
    return std::make_pair(-1., 1.);
  }

  /**
   * This function is used to get the number of local shape functions for
   * the BEM space.
   *
   * @return Integer q_, which is the number of local shape functions
   */
  int getQ() const { return q_; }

  /**
   * This function is used to evaluate a particular reference shape function of
   * the BEM space at a given point.
   *
   * @return Value of the local shape function at the given point
   */
  double evaluateShapeFunction(unsigned int q, double t) const {
    // Asserting that the requested shape function index is valid
    assert(q < q_);
    // Asserting that the evaluation point is within parameter domain
    assert(IsWithinParameterRange(t));
    // Evaluating the requested reference shape function which is stored in a
    // vector with others
    return referenceshapefunctions_[q](t);
  }

  /**
   * This function is used to evaluate the derivative of a particular reference
   * shape function of the BEM space at a given point.
   *
   * @return Value of the local shape function at the given point
   */
  double evaluateShapeFunctionDot(unsigned int q, double t) const {
    // Asserting that the requested shape function index is valid
    assert(q < q_);
    // Asserting that the evaluation point is within parameter domain
    assert(IsWithinParameterRange(t));
    // Evaluating the requested reference shape function which is stored in a
    // vector with others
    return referenceshapefunctiondots_[q](t);
  }

  /**
   * This function is used for checking whether a value t is within the
   * valid parameter range. This function is non virtual to prevent it
   * from being overriden as the parameter interval is fixed. It is
   * declared static because the check is independent of the concrete
   * implementation.
   *
   * @param t The value to be checked
   * @return boolean indicating result of the performed check
   */
  static bool IsWithinParameterRange(double t) {
    double a, b;
    // Getting the parameter range
    std::tie(a, b) = ParameterRange();
    // Checking if the value is within parameter range
    return (t >= a && t <= b);
  }

  /**
   * This function returns the dimension of the given BEM space on the given
   * number of panels. It is a pure virtual function and has to be implemented
   * by the inherited class.
   *
   * @param numpanels Number of panels in the mesh
   * @return Dimension of the BEM space
   */
  virtual unsigned int getSpaceDim(unsigned int numpanels) const = 0;

  /**
   * This function interpolates the function func, provided as an input of the
   * form std::function<double(double,double)>, on the given mesh. It is a pure
   * virtual function which uses the BEM space implementation for the
   * interpolation. The output is a vector \f$c_{i}\f$ which containes the
   * interpolation coefficients such that \f$func =
   * \sum_{i=1}^{N}c_{i}b_{i}^{N}\f$ on the mesh.
   *
   * @param func Function to be interpolated with the signature mentioned above
   * @param mesh Parametrized mesh object on which interpolation is done
   * @return An Eigen::VectorXd type containing the interpolation coefficients
   */
  virtual Eigen::VectorXd
  Interpolate(const std::function<double(double, double)> &func,
              const ParametrizedMesh &mesh) const = 0;

protected:
  /**
   * This typedef defines the basis function type. This is helpful in
   * storing the reference shape functions for a BEM space in a vector.
   */
  typedef std::function<double(double)> BasisFunctionType;
  /**
   * Protected constructor ensures this base class is non instantiable.
   */
  AbstractBEMSpace() : q_(0){};
  /**
   * A protected vector containing all the reference shape functions.
   */
  std::vector<BasisFunctionType> referenceshapefunctions_;
  /**
   * A protected vector containing all the reference shape function derivatives.
   */
  std::vector<BasisFunctionType> referenceshapefunctiondots_;
  /**
   * A protected integer which stores the number of reference shape
   * functions in the derived concrete classes
   */
  int q_;
}; // class AbstractBEMSpace
} // namespace parametricbem2d

#endif // ABSTRACTBEMSPACEHPP
