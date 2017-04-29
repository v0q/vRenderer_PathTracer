#version 410 core

#define FXAA_SUBPIXEL_SHIFT 1.f/4.f

uniform vec2 u_invScreenDim;

in vec2 a_Position;
in vec2 a_FragCoord;

out vec4 o_fxaaConsolePosPos;
out vec2 o_fxaaPos;
out vec2 o_FragCoord;

void main()
{
  o_FragCoord = a_FragCoord;

  // {xy} = center of pixel
  o_fxaaPos = a_FragCoord;

  // {xy__} = upper left of pixel
  // {__zw} = lower right of pixel
  o_fxaaConsolePosPos.xy = a_FragCoord;
  o_fxaaConsolePosPos.zw = a_FragCoord - u_invScreenDim * (0.5 + FXAA_SUBPIXEL_SHIFT);

  gl_Position = vec4(a_Position, 0.0, 1.0);
}
