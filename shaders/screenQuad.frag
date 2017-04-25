#version 410 core

uniform sampler2D u_ptResult;
uniform sampler2D u_ptDepth;
uniform int u_channel;

in vec2 o_FragCoord;
out vec4 o_FragColor;

void main()
{
  vec4 colour;
  if(u_channel == 0)
  {
    colour = texture(u_ptResult, o_FragCoord);
  }
  else
  {
    colour = texture(u_ptDepth, o_FragCoord);
  }

  o_FragColor = colour;
}
