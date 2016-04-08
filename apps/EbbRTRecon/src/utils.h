#ifndef UTILS_H
#define UTILS_H

#include <vector>

typedef struct unsigned_three
{
    unsigned int x, y, z;
} uint3;

struct POINT3D
{
  short x;
  short y;
  short z;
  float value;

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version)
  {
        ar & x & y & z & value;
  }  

};

typedef std::vector<POINT3D> VOXELCOEFFS;
typedef std::vector<std::vector<VOXELCOEFFS> > SLICECOEFFS;

#define PSF_SIZE 128

#endif
