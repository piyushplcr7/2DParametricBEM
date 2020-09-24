/**
 * \file parametrized_mesh.cpp
 * \brief This file defines the class for representing a mesh comprised
 *        of panels represented by parametrized curves.
 *
 *  This File is a part of the 2D-Parametric BEM package
 */

#include "parametrized_mesh.hpp"

#include <utility>

#include <Eigen/Dense>

namespace parametricbem2d {

ParametrizedMesh::ParametrizedMesh(PanelVector panels) : panels_(panels) {
  unsigned int N = getNumPanels();
  // Enforcing the constraint for the mesh where end point of a panel has to be
  // the starting point of the next such that the panels form a curved polygon
  for (unsigned int i = 0; i < N; ++i)
    assert(fabs((panels_[i % N]->operator()(1) -
                 panels_[(i + 1) % N]->operator()(-1))
                    .norm()) < 1e-5);
}

PanelVector ParametrizedMesh::getPanels() const {
  // Returning a copy of the stored panels
  return panels_;
}

unsigned int ParametrizedMesh::getNumPanels() const {
  // Returning the size of the PanelVector panels_
  return panels_.size();
}

Eigen::Vector2d ParametrizedMesh::getVertex(unsigned int i) const {
  assert(i < getNumPanels()); // Asserting requested index is within limits
  return panels_[i]->operator()(-1);
}

} // namespace parametricbem2d
