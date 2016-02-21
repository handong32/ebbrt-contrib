/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkScalarGaussianDyDy.h 62 2009-05-28 13:19:03Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2009-05-28 14:19:03 +0100 (Thu, 28 May 2009) $
  Version   : $Revision: 62 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKSCALARGAUSSIANDYDY_H

#define _IRTKSCALARGAUSSIANDYDY_H

/**

  Class for computing Gaussian second derivative with respect to y
  
*/

class irtkScalarGaussianDyDy : public irtkScalarGaussian {
  
protected:

  /// Normalization of the Gaussian function
  double _Factor;
  double _Exp;
  double _VarX;
  double _VarY;
  double _VarZ;

public:

  /// Constructor with anisotropic sigma and specific center
  irtkScalarGaussianDyDy(double sigma_x, double sigma_y, double sigma_z, double x_0, double y_0, double z_0);

  /// Virtual destructor
  virtual ~irtkScalarGaussianDyDy();

  /// Virtual local evaluation function
  double Evaluate(double, double, double);
  
};

#endif
