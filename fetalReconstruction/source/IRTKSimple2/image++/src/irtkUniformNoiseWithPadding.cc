/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkUniformNoiseWithPadding.cc 2 2008-12-23 12:40:14Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2008-12-23 12:40:14 +0000 (Tue, 23 Dec 2008) $
  Version   : $Revision: 2 $
  Changes   : $Author: dr $

=========================================================================*/

#include <irtkImage.h>
#include <irtkNoise.h>

template <class VoxelType> irtkUniformNoiseWithPadding<VoxelType>::irtkUniformNoiseWithPadding() : irtkUniformNoise<VoxelType>()
{
  _PaddingValue = std::numeric_limits<VoxelType>::min();
}

template <class VoxelType> irtkUniformNoiseWithPadding<VoxelType>::irtkUniformNoiseWithPadding(double Amplitude, VoxelType PaddingValue) : irtkUniformNoise<VoxelType>(Amplitude)
{
  _PaddingValue = PaddingValue;
}

template <class VoxelType> const char *irtkUniformNoiseWithPadding<VoxelType>::NameOfClass()
{
  return "irtkUniformNoiseWithPadding";
}

template <class VoxelType> double irtkUniformNoiseWithPadding<VoxelType>::Run(int x, int y, int z, int t)
{
  if (this->_input->Get(x, y, z, t) > this->_PaddingValue) {
    return this->irtkUniformNoise<VoxelType>::Run(x, y, z, t);
  } else {
    return this->_PaddingValue;
  }
}

template class irtkUniformNoiseWithPadding<irtkBytePixel>;
template class irtkUniformNoiseWithPadding<irtkGreyPixel>;
template class irtkUniformNoiseWithPadding<irtkRealPixel>;
