/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkConvolution_3D.h 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKCONVOLUTION_3D_H

#define _IRTKCONVOLUTION_3D_H

/**
 * Class for three-dimensional convolution.
 *
 * This class defines and implements three-dimensional convolutions of an
 * image with a filter kernel. The convolution is computed along all axis.
 */

template <class VoxelType> class irtkConvolution_3D : public irtkConvolution<VoxelType>
{

protected:

  /// Second input, i.e. the filter kernel
  irtkGenericImage<irtkRealPixel> *_input2;

  /** Returns whether the filter requires buffering. This filter requires
   *  buffering and returns 0.
   */
  virtual bool RequiresBuffering();

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /** Run the convolution filter. This method is protected and should only
   *  be called from within public member function Run().
   */
  virtual double Run(int, int, int, int);

public:

  /// Constructor
  irtkConvolution_3D(bool = false);

  /// Set second input, i.e. the filter kernel
  virtual void SetInput2(irtkGenericImage<irtkRealPixel> *);

  /// Run the convolution filter
  virtual void Initialize();
};

#endif
