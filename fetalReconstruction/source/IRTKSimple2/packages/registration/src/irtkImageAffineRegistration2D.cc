/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageAffineRegistration2D.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#include <irtkRegistration.h>

void irtkImageAffineRegistration2D::SetOutput(irtkTransformation *transformation)
{
  if (strcmp(transformation->NameOfClass(), "irtkAffineTransformation") != 0) {
    cerr << "irtkImageAffineRegistration::SetOutput: Transformation must be affine"
         << endl;
    exit(0);
  }
  _transformation = transformation;
}
