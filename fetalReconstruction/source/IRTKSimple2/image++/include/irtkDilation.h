/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkDilation.h 358 2011-06-28 07:10:17Z pa100 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2011-06-28 08:10:17 +0100 (Tue, 28 Jun 2011) $
  Version   : $Revision: 358 $
  Changes   : $Author: pa100 $

=========================================================================*/

#ifndef _IRTKDILATION_H

#define _IRTKDILATION_H

#include <irtkImageToImage.h>

/**
 * Class for dilation of images
 *
 * This class defines and implements the morphological dilation of images.
 *
 */

template <class VoxelType> class irtkDilation : public irtkImageToImage<VoxelType>
{

protected:

  /// Returns whether the filter requires buffering
  virtual bool RequiresBuffering();

  /// Returns the name of the class
  virtual const char *NameOfClass();

  /// Initialize the filter
  virtual void Initialize();

  /// What connectivity to assume when running the filter.
  irtkConnectivityType _Connectivity;

  // List of voxel offsets of the neighbourhood.
  irtkNeighbourhoodOffsets _offsets;

public:

  /// Constructor
  irtkDilation();

  /// Destructor
  ~irtkDilation();

  /// Run dilation
  virtual void Run();

  SetMacro(Connectivity, irtkConnectivityType);

  GetMacro(Connectivity, irtkConnectivityType);
};

#endif
