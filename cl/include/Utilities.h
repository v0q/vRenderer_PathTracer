#pragma once

int floatAsInt(const float _a)
{
  union
  {
    float a;
    int b;
  } c;
  c.a = _a;

  return c.b;
}
