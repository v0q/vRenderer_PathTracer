#version 410 core

uniform sampler2D u_ptResult;

in vec2 o_FragCoord;
out vec4 o_FragColor;

void main()
{
  o_FragColor = texture(u_ptResult, o_FragCoord);
}
