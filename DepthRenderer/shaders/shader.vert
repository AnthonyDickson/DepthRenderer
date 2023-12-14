#version 330

uniform mat4 mvp; // The model, view and projection matrices as one matrix.

in vec3 in_vert;   // Vertex position
in vec2 in_texcoord;   // Vertex texture coordinates

out vec2 frag_texcoord; // Interpolated fragment texture coordinates (out)

void main()
{
  gl_Position = mvp * vec4(in_vert, 1.0);
  frag_texcoord = in_texcoord;
}