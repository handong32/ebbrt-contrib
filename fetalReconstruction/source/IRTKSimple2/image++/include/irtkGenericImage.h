/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkGenericImage.h 968 2013-08-15 08:48:21Z kpk09 $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2013-08-15 09:48:21 +0100 (Thu, 15 Aug 2013) $
  Version   : $Revision: 968 $
  Changes   : $Author: kpk09 $

=========================================================================*/

#ifndef _IRTKGENERICIMAGE_H

#define _IRTKGENERICIMAGE_H

#ifdef HAS_VTK
i
#include <vtkStructuredPoints.h>

#endif

#include <vector>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/tmpdir.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/serialization/base_object.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>

#define _1D_

//convert 4D to 1D indexing
#define TO1D(t, z, y, x, dT, dZ, dY, dX)                                       \
  (t * dZ * dY * dX + z * dY * dX + y * dX + x)

/**
 * Generic class for 2D or 3D images
 *
 * This class implements generic 2D and 3D images. It provides functions
 * for accessing, reading, writing and manipulating images. This class can
 * be used for images with arbitrary voxel types using templates.
 */

template <typename T> class irtkGenericImage : public irtkBaseImage {
public:
  /// Voxel type
  typedef T VoxelType;

protected:
/// Pointer to image data
#ifdef _1D_
  VoxelType *_matrix;
#else
  VoxelType ****_matrix;
#endif

  /// Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    int i, n;
    VoxelType *ptr1;

    ar &_attr;
    ar &_matI2W;
    ar &_matW2I;

    /*if(_matrix == NULL)
    {
        irtkImageAttributes attr = _attr;
        _matrix = Allocate(_matrix, attr._x, attr._y, attr._z, attr._t);
        }*/

    n = this->GetNumberOfVoxels();
    ptr1 = this->GetPointerToVoxels();

    for (i = 0; i < n; i++) {
      ar &ptr1[i];
    }
  }

public:
  /// Default constructor
  irtkGenericImage();

  /// Constructor from image file
  irtkGenericImage(char *);

  /// Constructor for given image size
  irtkGenericImage(int, int, int, int = 1);

  /// Copy constructor for image
  irtkGenericImage(const irtkGenericImage &);

  /// Constructor for given image attributes
  irtkGenericImage(const irtkImageAttributes &);

  irtkGenericImage(const irtkImageAttributes &, VoxelType[], irtkMatrix, irtkMatrix);

  /// Copy constructor for image of different type
  template <class TVoxel2> irtkGenericImage(const irtkGenericImage<TVoxel2> &);

  /// Destructor
  ~irtkGenericImage(void);

  /// Initialize an image
  void Initialize(const irtkImageAttributes &);

  /// Initialize an image
  //void Initialize(const irtkImageAttributes &, VoxelType[], irtkMatrix, irtkMatrix);

  /// Clear an image
  void Clear();

  /// Read image from file
  void Read(const char *);

  /// Write image to file
  void Write(const char *);

  /// Minimum and maximum pixel values get accessor
  void GetMinMax(VoxelType *, VoxelType *) const;

  /// Average pixel values get accessor
  VoxelType GetAverage(int = 1) const;

  /// Standard Deviation of the pixels
  VoxelType GetSD(int = 1) const;

  /// Get Max Intensity position around the point
  void GetMaxPosition(irtkPoint &, int = 1, int = 0) const;

  /// Get Gravity center position of a given window
  void GravityCenter(irtkPoint &, int = 1, int = 0) const;

  /// Minimum and maximum pixel values get accessor with padding
  void GetMinMaxPad(VoxelType *, VoxelType *, VoxelType) const;

  /// Minimum and maximum pixel values put accessor
  void PutMinMax(VoxelType, VoxelType);

  /// Saturation
  void Saturate(double q0 = 0.01, double q1 = 0.99);

  /// Function for pixel access via pointers
  VoxelType *GetPointerToVoxels(int = 0, int = 0, int = 0, int = 0) const;

  /// Function to convert pixel to index
  int VoxelToIndex(int, int, int, int = 0) const;

  /// Function for pixel get access
  VoxelType Get(int, int, int, int = 0) const;

  /// Function for pixel put access
  void Put(int, int, int, VoxelType);

  /// Function for pixel put access
  void Put(int, int, int, int, VoxelType);

  /// Function for pixel access from via operators
  VoxelType &operator()(int, int, int, int = 0);

  /// Function for image slice get access
  irtkGenericImage GetRegion(int z, int t) const;

  /// Function for image slice get access in certain region
  irtkGenericImage GetRegion(int x1, int y1, int z1, int x2, int y2,
                             int z2) const;

  /// Function for image slice get access in certain region
  irtkGenericImage GetRegion(int x1, int y1, int z1, int t1, int x2, int y2,
                             int z2, int t2) const;

  /// Function for image frame get access
  irtkGenericImage GetFrame(int t) const;

  double Sum();
  double AttrSum();
  int GetSizeMat();
  VoxelType *GetMat();
  double I2WSum();
  double W2ISum();
  void SumVec(VoxelType* mat);

  //
  // Operators for image arithmetics
  //

  /// Copy operator for image
  irtkGenericImage<VoxelType> &operator=(const irtkGenericImage &);

  /// Copy operator for image
  template <class TVoxel2>
  irtkGenericImage<VoxelType> &operator=(const irtkGenericImage<TVoxel2> &);

  /// Addition operator
  irtkGenericImage operator+(const irtkGenericImage &);

  /// Addition operator (stores result)
  irtkGenericImage &operator+=(const irtkGenericImage &);

  /// Subtraction operator
  irtkGenericImage operator-(const irtkGenericImage &);

  /// Subtraction operator (stores result)
  irtkGenericImage &operator-=(const irtkGenericImage &);

  /// Multiplication operator
  irtkGenericImage operator*(const irtkGenericImage &);

  /// Multiplication operator (stores result)
  irtkGenericImage &operator*=(const irtkGenericImage &);

  /// Division operator
  irtkGenericImage operator/(const irtkGenericImage &);

  /// Division operator (stores result)
  irtkGenericImage &operator/=(const irtkGenericImage &);

  //
  // Operators for image and Type arithmetics
  //

  /// Set all pixels to a constant value
  irtkGenericImage &operator=(VoxelType);
  /// Addition operator for type
  irtkGenericImage operator+(VoxelType);
  /// Addition operator for type (stores result)
  irtkGenericImage &operator+=(VoxelType);
  /// Subtraction operator for type
  irtkGenericImage operator-(VoxelType);
  /// Subtraction operator for type (stores result)
  irtkGenericImage &operator-=(VoxelType);
  /// Multiplication operator for type
  irtkGenericImage operator*(VoxelType);
  /// Multiplication operator for type (stores result)
  irtkGenericImage &operator*=(VoxelType);
  /// Division operator for type
  irtkGenericImage operator/(VoxelType);
  /// Division operator for type (stores result)
  irtkGenericImage &operator/=(VoxelType);

  //
  // Operators for image thresholding
  //

  /// Threshold operator >  (sets all values >  given value to that value)
  irtkGenericImage operator>(VoxelType);
  /// Threshold operator >= (sets all values >= given value to that value)
  irtkGenericImage &operator>=(VoxelType);
  /// Threshold operator <  (sets all values <  given value to that value)
  irtkGenericImage operator<(VoxelType);
  /// Threshold operator <= (sets all values <= given value to that value)
  irtkGenericImage &operator<=(VoxelType);

  /// Comparison operators == (explicit negation yields != operator)
  bool operator==(const irtkGenericImage &);

  /// Comparison operator != (if _HAS_STL is defined, negate == operator)
  ///  bool operator!=(const irtkGenericImage &);

  irtkGenericImage operator!=(VoxelType);

  //
  // Reflections and axis flipping
  //

  /// Reflect image around x
  void ReflectX();
  /// Reflect image around y
  void ReflectY();
  /// Reflect image around z
  void ReflectZ();
  /// Flip x and y axis
  void FlipXY(int);
  /// Flip x and z axis
  void FlipXZ(int);
  /// Flip y and z axis
  void FlipYZ(int);
  /// Flip x and t axis
  void FlipXT(int);
  /// Flip y and t axis
  void FlipYT(int);
  /// Flip z and t axis
  void FlipZT(int);

//
// Conversions from and to VTK
//

#ifdef HAS_VTK

  /// Return the VTK scalar image type of an IRTK image
  int ImageToVTKScalarType();

  /// Conversion to VTK structured points
  void ImageToVTK(vtkStructuredPoints *);

  /// Conversion from VTK structured points
  void VTKToImage(vtkStructuredPoints *);
#endif

  /// Function for pixel get access as double
  double GetAsDouble(int, int, int, int = 0) const;

  /// Function for pixel put access
  void PutAsDouble(int, int, int, double);

  /// Function for pixel put access
  void PutAsDouble(int, int, int, int, double);

  /// Returns the name of the image class
  const char *NameOfClass();

  /// Function for pixel access via pointers
  void *GetScalarPointer(int = 0, int = 0, int = 0, int = 0) const;

  /// Function which returns pixel scalar type
  virtual int GetScalarType() const;

  /// Function which returns the minimum value the pixel can hold without
  /// overflowing
  virtual double GetScalarTypeMin() const;

  /// Function which returns the minimum value the pixel can hold without
  /// overflowing
  virtual double GetScalarTypeMax() const;
};

template <class VoxelType>
inline void irtkGenericImage<VoxelType>::Put(int x, int y, int z,
                                             VoxelType val) {
#ifdef NO_BOUNDS
#ifdef _1D_
  _matrix[TO1D(0, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] = val;
#else
  _matrix[0][z][y][x] = val;
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0)) {
    cout << "irtkGenericImage<VoxelType>::Put: parameter out of range\n";
    exit(-1);
  } else {
#ifdef _1D_
    _matrix[TO1D(0, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] =
        static_cast<VoxelType>(val);
#else
    _matrix[0][z][y][x] = static_cast<VoxelType>(val);
#endif
  }
#endif
}

template <class VoxelType>
inline void irtkGenericImage<VoxelType>::Put(int x, int y, int z, int t,
                                             VoxelType val) {
#ifdef NO_BOUNDS
#ifdef _1D_
  _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] = val;
#else
  _matrix[t][z][y][x] = val;
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<VoxelType>::Put: parameter out of range\n";
    exit(-1);
  } else {
#ifdef _1D_
    _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] = val;
#else
    _matrix[t][z][y][x] = val;
#endif
  }
#endif
}

template <class VoxelType>
inline void irtkGenericImage<VoxelType>::PutAsDouble(int x, int y, int z,
                                                     double val) {
  if (val > voxel_limits<VoxelType>::max())
    val = voxel_limits<VoxelType>::max();
  if (val < voxel_limits<VoxelType>::min())
    val = voxel_limits<VoxelType>::min();

#ifdef NO_BOUNDS
#ifdef _1D_
  _matrix[TO1D(0, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] =
      static_cast<VoxelType>(val);
#else
  _matrix[0][z][y][x] = static_cast<VoxelType>(val);
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (_attr._t > 0)) {
    cout << "irtkGenericImage<Type>::PutAsDouble: parameter out of range\n";
    exit(-1);
  } else {
#ifdef _1D_
    _matrix[TO1D(0, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] =
        static_cast<VoxelType>(val);
#else
    _matrix[0][z][y][x] = static_cast<VoxelType>(val);
#endif
  }
#endif
}

template <class VoxelType>
inline void irtkGenericImage<VoxelType>::PutAsDouble(int x, int y, int z, int t,
                                                     double val) {
  if (val > voxel_limits<VoxelType>::max())
    val = voxel_limits<VoxelType>::max();
  if (val < voxel_limits<VoxelType>::min())
    val = voxel_limits<VoxelType>::min();

#ifdef NO_BOUNDS
#ifdef _1D_
  _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] =
      static_cast<VoxelType>(val);
#else
  _matrix[t][z][y][x] = static_cast<VoxelType>(val);
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<Type>::PutAsDouble: parameter out of range\n";
    exit(-1);
  } else {
#ifdef _1D_
    _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)] =
        static_cast<VoxelType>(val);
#else
    _matrix[t][z][y][x] = static_cast<VoxelType>(val);
#endif
  }
#endif
}

template <class VoxelType>
inline VoxelType irtkGenericImage<VoxelType>::Get(int x, int y, int z,
                                                  int t) const {
// cout << "Get _attr._t=" << _attr._t << " _attr._z=" << _attr._z << "
// _attr._y=" << _attr._y << " _attr._x=" << _attr._x;
// cout << " t=" << t << " z=" << z << " y=" << y << " x=" << x << endl;

#ifdef NO_BOUNDS
#ifdef _1D_
  return _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)];
#else
  return (_matrix[t][z][y][x]);
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<Type>::Get: parameter out of range\n";
    return -1;
  } else {
#ifdef _1D_
    // cout << "get = " << _matrix[TO1D(t, z, y, x, _attr._t, _attr._z,
    // _attr._y, _attr._x)] << endl;
    return _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)];
#else
    // cout << "get = " << (_matrix[t][z][y][x]) << endl;
    return (_matrix[t][z][y][x]);
#endif
  }
#endif
}

template <class VoxelType>
inline double irtkGenericImage<VoxelType>::GetAsDouble(int x, int y, int z,
                                                       int t) const {

// cout << "GetAsDouble t = " << t << " z = " << z << " y = " << y << " x = " <<
// x;
// cout << " _attr._t = " << _attr._t << " _attr._z =" << _attr._z << " _attr._y
// = " << _attr._y << " _attr._x = " << _attr._x << endl;

// cout << " TO1D " << TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y);
// cout << " total = " << _attr._t * _attr._z * _attr._y * _attr._x  << "\n" <<
// endl;

#ifdef NO_BOUNDS
#ifdef _1D_
  return (static_cast<double>(
      _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]));
#else
  return (static_cast<double>(_matrix[t][z][y][x]));
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<Type>::GetAsDouble: parameter out of range\n";
    return -1;
  } else {
#ifdef _1D_
    return (static_cast<double>(
        _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]));
#else
    return (static_cast<double>(_matrix[t][z][y][x]));
#endif
  }
#endif
}

template <class VoxelType>
inline VoxelType &irtkGenericImage<VoxelType>::operator()(int x, int y, int z,
                                                          int t) {
#ifdef NO_BOUNDS
#ifdef _1D_
  return _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)];
#else
  return (_matrix[t][z][y][x]);
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<Type>::(): parameter out of range\n";
    exit(-1);
  } else {
#ifdef _1D_
    return _matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)];
#else
    return (_matrix[t][z][y][x]);
#endif
  }
#endif
}

template <class VoxelType>
inline int irtkGenericImage<VoxelType>::VoxelToIndex(int x, int y, int z,
                                                     int t) const {
#ifdef NO_BOUNDS
#ifdef _1D_
  return (&(_matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]) -
          &(_matrix[0]));
#else
  return (&(_matrix[t][z][y][x]) - &(_matrix[0][0][0][0]));
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<Type>::VoxelToIndex: parameter out of range\n";
    return -1;
  } else {
#ifdef _1D_
    return (
        &(_matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]) -
        &(_matrix[0]));
#else
    return (&(_matrix[t][z][y][x]) - &(_matrix[0][0][0][0]));
#endif
  }
#endif
}

template <class VoxelType>
inline VoxelType *irtkGenericImage<VoxelType>::GetPointerToVoxels(int x, int y,
                                                                  int z,
                                                                  int t) const {
#ifdef NO_BOUNDS
#ifdef _1D_
  return &(_matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]);
#else
  return &(_matrix[t][z][y][x]);
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout << "irtkGenericImage<Type>::GetPointerToVoxels: parameter out of "
            "range\n"
         << endl;
    return NULL;
  } else {
#ifdef _1D_
    return &(_matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]);
#else
    return &(_matrix[t][z][y][x]);
#endif
  }
#endif
}

template <class VoxelType>
inline void *irtkGenericImage<VoxelType>::GetScalarPointer(int x, int y, int z,
                                                           int t) const {
#ifdef NO_BOUNDS
#ifdef _1D_
  return &(_matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]);
#else
  return &(_matrix[t][z][y][x]);
#endif
#else
  if ((x >= _attr._x) || (x < 0) || (y >= _attr._y) || (y < 0) ||
      (z >= _attr._z) || (z < 0) || (t >= _attr._t) || (t < 0)) {
    cout
        << "irtkGenericImage<Type>::GetScalarPointer: parameter out of range\n";
    return NULL;
  } else {
#ifdef _1D_
    return &(_matrix[TO1D(t, z, y, x, _attr._t, _attr._z, _attr._y, _attr._x)]);
#else
    return &(_matrix[t][z][y][x]);
#endif
  }
#endif
}

template <> inline int irtkGenericImage<char>::GetScalarType() const {
  return IRTK_VOXEL_CHAR;
}

template <> inline int irtkGenericImage<unsigned char>::GetScalarType() const {
  return IRTK_VOXEL_UNSIGNED_CHAR;
}

template <> inline int irtkGenericImage<unsigned short>::GetScalarType() const {
  return IRTK_VOXEL_UNSIGNED_SHORT;
}

template <> inline int irtkGenericImage<short>::GetScalarType() const {
  return IRTK_VOXEL_SHORT;
}

template <> inline int irtkGenericImage<int>::GetScalarType() const {
  return IRTK_VOXEL_INT;
}

template <> inline int irtkGenericImage<unsigned int>::GetScalarType() const {
  return IRTK_VOXEL_UNSIGNED_INT;
}

template <> inline int irtkGenericImage<float>::GetScalarType() const {
  return IRTK_VOXEL_FLOAT;
}

template <> inline int irtkGenericImage<double>::GetScalarType() const {
  return IRTK_VOXEL_DOUBLE;
}

template <class VoxelType>
inline double irtkGenericImage<VoxelType>::GetScalarTypeMin() const {
  return std::numeric_limits<VoxelType>::min();
}

template <class VoxelType>
inline double irtkGenericImage<VoxelType>::GetScalarTypeMax() const {
  return std::numeric_limits<VoxelType>::max();
}

template <class VoxelType> inline double irtkGenericImage<VoxelType>::Sum() {
  int i, n;
  double sum = 0;

  n = this->GetNumberOfVoxels();
  auto ptr1 = this->GetPointerToVoxels();

  for (i = 0; i < n; i++) {
    sum += (double)ptr1[i];
  }
  return sum;
}

template <class VoxelType>
inline double irtkGenericImage<VoxelType>::AttrSum() {
  return _attr.Sum();
}

template <class VoxelType>
inline VoxelType *irtkGenericImage<VoxelType>::GetMat() {
  return _matrix;
}

template <class VoxelType>
inline int irtkGenericImage<VoxelType>::GetSizeMat() {
  return _attr._t * _attr._z * _attr._y * _attr._x;
}

template <class VoxelType>
inline double irtkGenericImage<VoxelType>::I2WSum() {
    return _matI2W.Sum();
}

template <class VoxelType>
inline double irtkGenericImage<VoxelType>::W2ISum() {
    return _matW2I.Sum();
}

template <class VoxelType>
inline void irtkGenericImage<VoxelType>::SumVec(VoxelType* mat)
{
    int i, n;
    
    n = this->GetNumberOfVoxels();
    for (i = 0; i < n; i++) {
        _matrix[i] += mat[i];
    }
}

#endif
