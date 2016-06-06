/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : $Id: irtkRigidTransformation.h 235 2010-10-18 09:25:20Z dr $
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2008 onwards
  Date      : $Date: 2010-10-18 10:25:20 +0100 (Mon, 18 Oct 2010) $
  Version   : $Revision: 235 $
  Changes   : $Author: dr $

=========================================================================*/

#ifndef _IRTKRIGIDTRANSFORMATION_H

#define _IRTKRIGIDTRANSFORMATION_H

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/list.hpp>
#include <boost/serialization/assume_abstract.hpp>
#include <boost/archive/tmpdir.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/**
 * Class for rigid transformations.
 *
 * This class defines and implements rigid body transformations. The rigid
 * body transformations are parameterized by three rotations around the axes
 * of the coordinate system followed by three translations along the axes of
 * the coordinate system. Note that the order of rotations is defined as a
 * rotation around the z-axis, the y-axis and finally around the x-axis. In
 * total, the transformation is parameterized by six degrees of freedom.
 *
 */

class irtkRigidTransformation : public irtkHomogeneousTransformation {

protected:
  /// Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) {
    ar & _tx & _ty & _tz & _rx & _ry & _rz & _cosrx
	& _cosry & _cosrz & _sinrx & _sinry & _sinrz; 

//    if(_matrix == NULL)
    //  {
//	_matrix = irtkMatrix(4,4);
//	_matrix.Ident();
//	this->UpdateMatrix();
    //  }

    ar & _matrix;
  }

  /// Construct a matrix based on parameters passed in the array.
  virtual irtkMatrix Parameters2Matrix(double *) const;

  /// Return an array with parameters corresponding to a given matrix.
  virtual void Matrix2Parameters(irtkMatrix, double *) const;

  /// Assign the parameters passed to the current object and update the
  /// matrix.
  virtual void SetParameters(double *params);

public:
    /// Translation along the x-axis (in mm)
  double _tx;

  /// Translation along the y-axis (in mm)
  double _ty;

  /// Translation along the z-axis (in mm)
  double _tz;

  /// Rotation around the x-axis (in degrees)
  double _rx;

  /// Rotation around the y-axis (in degrees)
  double _ry;

  /// Rotation around the z-axis (in degrees)
  double _rz;

  /// Cosine of rotation angle rx
  double _cosrx;

  /// Cosine of rotation angle ry
  double _cosry;

  /// Cosine of rotation angle rz
  double _cosrz;

  /// Sine of rotation angle rx
  double _sinrx;

  /// Sine of rotation angle ry
  double _sinry;

  /// Sine of rotation angle rz
  double _sinrz;


  /// Constructor (default)
  irtkRigidTransformation();

  /// Constructor (copy)
  irtkRigidTransformation(const irtkRigidTransformation &);
  
  irtkRigidTransformation(double, double, double, double, double, double, double, double, double, double, double, double, int, int, int, int, int, int, irtkMatrix);

  /// Destructor
  virtual ~irtkRigidTransformation();

  /// Reset transformation
  virtual void Reset();

  /// Puts translation along the x-axis (transformation matrix is updated)
  void PutTranslationX(double);

  /// Gets translation along the x-axis
  virtual double GetTranslationX();

  /// Puts translation along the y-axis (transformation matrix is updated)
  virtual void PutTranslationY(double);

  /// Gets translation along the y-axis
  virtual double GetTranslationY();

  /// Puts translation along the z-axis (transformation matrix is updated)
  virtual void PutTranslationZ(double);

  /// Gets translation along the z-axis
  virtual double GetTranslationZ();

  /// Puts rotation angle around the x-axis (transformation matrix is updated)
  virtual void PutRotationX(double);

  /// Gets rotation angle around the x-axis
  virtual double GetRotationX();

  /// Puts rotation angle around the y-axis (transformation matrix is updated)
  virtual void PutRotationY(double);

  /// Gets rotation angle around the y-axis
  virtual double GetRotationY();

  /// Puts rotation angle around the z-axis (transformation matrix is updated)
  virtual void PutRotationZ(double);

  /// Gets rotation angle around the z-axis
  virtual double GetRotationZ();

  /// Returns the number of parameters of the transformation
  virtual int NumberOfDOFs() const;

  /// Puts a transformation parameter (transformation matrix is updated)
  virtual void Put(int, double);

  /// Gets a transformation parameter
  virtual double Get(int) const;

  /// Transforms a point by the rotation part of the rigid transformation.
  virtual void Rotate(double &x, double &y, double &z);

  /// Calculate the Jacobian of the transformation with respect to the
  /// transformation parameters
  virtual void JacobianDOFs(double[3], int, double, double, double, double = 0);

  /// Checks whether transformation is an identity mapping
  virtual bool IsIdentity();

  /// Prints the parameters of the transformation
  virtual void Print();

  void Print2();
  
  void Print3();
  
  double Sum();
  
  /// Check file header
  static int CheckHeader(char *);

  /// Returns a string with the name of the instantiated class
  virtual const char *NameOfClass();

  /// Reads a transformation from a file
  virtual irtkCifstream &Read(irtkCifstream &);

  /// Writes a transformation to a file
  virtual irtkCofstream &Write(irtkCofstream &);

  /// Imports a transformation from a file
  virtual istream &Import(istream &);

  /// Exports a transformation to a file
  virtual ostream &Export(ostream &);

  /// Puts the transformation matrix (transformation parameters are updated)
  virtual void PutMatrix(const irtkMatrix &);

  /// Updates transformation matrix
  virtual void UpdateMatrix();

  /// Updates transformation parameters
  virtual void UpdateParameter();
};

inline int irtkRigidTransformation::NumberOfDOFs() const { return 6; }

inline irtkRigidTransformation::irtkRigidTransformation() {
  int i;

  _tx = _ty = _tz = 0;
  _rx = _ry = _rz = 0;

  // Free memory allocated for DOF status by base class
  delete[] _status;

  // Allocate memory for DOF status
  _status = new _Status[this->NumberOfDOFs()];

  // Initialize memory for DOF status
  for (i = 0; i < this->NumberOfDOFs(); i++) {
    _status[i] = _Active;
  }

  // Update transformation matrix
  this->UpdateMatrix();
}

inline irtkRigidTransformation::irtkRigidTransformation(
    const irtkRigidTransformation &t)
    : irtkHomogeneousTransformation(t) {
  int i;

  _tx = t._tx;
  _ty = t._ty;
  _tz = t._tz;
  _rx = t._rx;
  _ry = t._ry;
  _rz = t._rz;

  // Free memory allocated for DOF status by base class
  delete[] _status;

  // Allocate memory for DOF status
  _status = new _Status[this->NumberOfDOFs()];

  // Initialize memory for DOF status
  for (i = 0; i < this->NumberOfDOFs(); i++) {
    _status[i] = t._status[i];
  }

  // Update transformation matrix
  this->UpdateMatrix();
}

inline irtkRigidTransformation::irtkRigidTransformation(double tx, double ty, double tz, double rx, double ry, double rz, double cosrx, double cosry, double cosrz, double sinrx, double sinry, double sinrz, int s0, int s1, int s2, int s3, int s4, int s5, irtkMatrix m) {
    
    _tx = tx;
    _ty = ty;
    _tz = tz;

    _rx = rx;
    _ry = ry;
    _rz = rz;

    _cosrx = cosrx;
    _cosry = cosry;
    _cosrz = cosrz;

    _sinrx = sinrx;
    _sinry = sinry;
    _sinrz = sinrz;

    if (_status != NULL) delete[] _status;
    
    _status = new _Status[this->NumberOfDOFs()];
    _status[0] = (_Status)s0;
    _status[1] = (_Status)s1;
    _status[2] = (_Status)s2;
    _status[3] = (_Status)s3;
    _status[4] = (_Status)s4;
    _status[5] = (_Status)s5;

    _matrix = std::move(m);
}

inline irtkRigidTransformation::~irtkRigidTransformation() {}

inline void irtkRigidTransformation::Reset() {
  // Initialize rotations and translations
  _tx = _ty = _tz = 0;
  _rx = _ry = _rz = 0;

  // Update transformation matrix
  this->UpdateMatrix();
}

inline void irtkRigidTransformation::PutRotationX(double rx) {
  _rx = rx;
  this->UpdateMatrix();
}

inline double irtkRigidTransformation::GetRotationX() { return _rx; }

inline void irtkRigidTransformation::PutRotationY(double ry) {
  _ry = ry;
  this->UpdateMatrix();
}

inline double irtkRigidTransformation::GetRotationY() { return _ry; }

inline void irtkRigidTransformation::PutRotationZ(double rz) {
  _rz = rz;
  this->UpdateMatrix();
}

inline double irtkRigidTransformation::GetRotationZ() { return _rz; }

inline void irtkRigidTransformation::PutTranslationX(double tx) {
  _tx = tx;
  this->UpdateMatrix();
}

inline double irtkRigidTransformation::GetTranslationX() { return _tx; }

inline void irtkRigidTransformation::PutTranslationY(double ty) {
  _ty = ty;
  this->UpdateMatrix();
}

inline double irtkRigidTransformation::GetTranslationY() { return _ty; }

inline void irtkRigidTransformation::PutTranslationZ(double tz) {
  _tz = tz;
  this->UpdateMatrix();
}

inline double irtkRigidTransformation::GetTranslationZ() { return _tz; }

inline void irtkRigidTransformation::PutMatrix(const irtkMatrix &matrix) {
  _matrix = matrix;
  this->UpdateParameter();
}

inline const char *irtkRigidTransformation::NameOfClass() {
  return "irtkRigidTransformation";
}

inline void irtkRigidTransformation::SetParameters(double *params) {
  _tx = params[TX];
  _ty = params[TY];
  _tz = params[TZ];

  _rx = params[RX];
  _ry = params[RY];
  _rz = params[RZ];

  this->UpdateMatrix();
}

inline void irtkRigidTransformation::Print2() {
    cout << _tx << " " << _ty << " " << _tz << " " << _rx << " " << _ry << " " << _rz << " " << _cosrx << " " << _cosry << " " << _cosrz << " " << _sinrx << " " << _sinry << " " << _sinrz << endl; 
    
    _matrix.Print();
}

inline void irtkRigidTransformation::Print3() {
    cout << _tx << " " << _ty << " " << _tz << " " << _rx << " " << _ry << " " << _rz << " " << _cosrx << " " << _cosry << " " << _cosrz << " " << _sinrx << " " << _sinry << " " << _sinrz << " _status: " << _status[0] << " " << _status[1] <<" " << _status[2] <<" " << _status[3] <<" " << _status[4] <<" " << _status[5]; 
    
    cout << " _matrix = " << _matrix.Sum() << endl;
}

inline double irtkRigidTransformation::Sum() {
    double sum;
    sum = _tx +_ty +_tz +_rx +_ry +_rz +_cosrx +_cosry +_cosrz +_sinrx +_sinry +_sinrz + _status[0] + _status[1] + _status[2] + _status[3] + _status[4] + _status[5]; 

    return sum;
}
#endif
