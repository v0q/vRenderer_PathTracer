__kernel void render(__write_only image2d_t _texture, unsigned int _w, unsigned int _h)
{
  unsigned int x = get_global_id(0);
  unsigned int y = get_global_id(1);
  int2 imgCoords = (int2)(x, _h-y);
  if(x < _w && y < _h)
  {
    write_imagef(_texture, imgCoords, make_float4((float)x/_w, (float)y/_h, 0.f, 1.f));
  }
}
