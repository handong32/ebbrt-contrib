/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkScalarFunctionToImage.h 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKSCALARFUNCTIONTOIMAGE_H

#define _IRTKSCALARFUNCTIONTOIMAGE_H

/**
 * Class for scalar function to image filter.
 *
 * This class uses a scalar function to produce an image as output. The
 * filter loops through each voxel of the output image and calculates its
 * intensity as the value of the scalar function as a function of spatial
 * location.
 */

template <class VoxelType> class irtkScalarFunctionToImage : public irtkObject
{

private:

  /// Debugging flag
  bool _DebugFlag;

  /// Flag to use world or image coordinates for scalar function evaluation
  bool _UseWorldCoordinates;

protected:

  /// Input for the filter
  irtkScalarFunction *_input;

  /// Output for the filter
  irtkGenericImage<VoxelType> *_output;

public:

  /// Constructor (using world coordinates by default)
  irtkScalarFunctionToImage(bool = true);

  /// Deconstuctor
  virtual ~irtkScalarFunctionToImage();

  /// Set input
  virtual void SetInput (irtkScalarFunction *);

  /// Set output
  virtual void SetOutput(irtkGenericImage<VoxelType> *);

  /// Run the filter on entire image
  virtual void   Run();

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Set debugging flag
  SetMacro(DebugFlag, bool);

  /// Get debugging flag
  GetMacro(DebugFlag, bool);

  /// Print debugging messages if debugging is enabled
  virtual void Debug(char *);

};

#endif
