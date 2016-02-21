/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkImageRigidRegistration.h 509 2012-01-17 10:45:53Z mm3 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2012-01-17 10:45:53 +0000 (Tue, 17 Jan 2012) $
  Version   : $Revision: 509 $
  Changes   : $Author: mm3 $

=========================================================================*/

#ifndef _IRTKIMAGERIGIDREGISTRATION_H

#define _IRTKIMAGERIGIDREGISTRATION_H

/**
 * Filter for rigid registration based on voxel similarity measures.
 *
 * This class implements a registration filter for the rigid registration of
 * two images. The basic algorithm is described in Studholme, Medical Image
 * Analysis, Vol. 1, No. 2, 1996.
 *
 */

class irtkImageRigidRegistration : public irtkImageRegistration
{

protected:

  /// Evaluate the similarity measure for a given transformation.
  virtual double Evaluate();

  /// Initial set up for the registration
  virtual void Initialize();

  /// Final set up for the registration
  virtual void Finalize();

public:

  /** Sets the output for the registration filter. The output must be a rigid
   *  transformation. The current parameters of the rigid transformation are
   *  used as initial guess for the rigid registration. After execution of the
   *  filter the parameters of the rigid transformation are updated with the
   *  optimal transformation parameters.
   */
  virtual void SetOutput(irtkTransformation *);

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Print information about the progress of the registration
  virtual void Print();

  /// Guess parameters
  virtual void GuessParameter();
  /// Guess parameters for slice to volume registration 
  virtual void GuessParameterSliceToVolume();
  /// Guess parameters volumes with thick slices
  virtual void GuessParameterThickSlices();
};

inline void irtkImageRigidRegistration::SetOutput(irtkTransformation *transformation)
{
  if (strcmp(transformation->NameOfClass(), "irtkRigidTransformation") != 0) {
    cerr << "irtkImageRigidRegistration::SetOutput: Transformation must be rigid"
         << endl;
    exit(0);
  }
  _transformation = transformation;
}

inline const char *irtkImageRigidRegistration::NameOfClass()
{
  return "irtkImageRigidRegistration";
}

inline void irtkImageRigidRegistration::Print()
{
  _transformation->Print();
}

#endif
