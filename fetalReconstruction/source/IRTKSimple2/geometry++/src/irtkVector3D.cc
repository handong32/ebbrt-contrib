/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkVector3D.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#include <irtkVector3D.h>

template <typename T> irtkVector3D<T> irtkVector3D<T>::operator/(const irtkVector3D<T>& v)
{
  irtkVector3D<T> val(0, 0, 0);

  if (v._x != 0) {
    val._x = _x/v._x;
  }

  if (v._y != 0) {
    val._y = _y/v._y;
  }

  if (v._z != 0) {
    val._z = _z/v._z;
  }

  return val;
}

template <typename T> irtkVector3D<T>& irtkVector3D<T>::operator/=(const irtkVector3D<T>& v)
{
  if (v._x != 0) {
    _x /= v._x;
  }

  if (v._y != 0) {
    _y /= v._y;
  }

  if (v._z != 0) {
    _z /= v._z;
  }

  return *this;
}

template class irtkVector3D<char>;
template class irtkVector3D<short>;
template class irtkVector3D<float>;
template class irtkVector3D<double>;
