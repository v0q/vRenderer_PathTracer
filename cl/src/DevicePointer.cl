#include "cl/include/PathTracer.h"

__kernel void initDevicePointer(__global vMesh *_mesh, __global vTriangle *_triangles)
{
  _mesh->m_mesh = _triangles;
}
