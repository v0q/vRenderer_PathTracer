#version 410 core

uniform sampler2D u_ptResult;
uniform sampler2D u_ptDepth;

in vec2 o_FragCoord;
out vec4 o_FragColor;

void main()
{
  vec4 colour = texture(u_ptDepth, o_FragCoord);
  colour *= 1.f;
  o_FragColor = texture(u_ptResult, o_FragCoord);

//  vec4 colour = texture(u_ptResult, o_FragCoord);
//  colour *= 1.f;
//  o_FragColor = texture(u_ptDepth, o_FragCoord);
}
