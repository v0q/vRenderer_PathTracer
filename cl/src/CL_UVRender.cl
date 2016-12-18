__kernel void render(__global float3 *_texture, unsigned int _w, unsigned int _h)
{
  const unsigned int index = get_global_id(0);
  unsigned int x = _w%index;
  unsigned int y = _w/index;

//  _texture[index] = make_float3((float)x/_w, (float)y/_h, 0.5f);
}
