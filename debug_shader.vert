#version 460

uniform mat4      mvp;           // The model, view and projection matrices as one matrix.
uniform sampler2D colourSampler; // Texture

attribute vec3 position;   // Vertex position
attribute vec2 texcoord;   // Vertex texture coordinates

varying vec2 v_texcoord; // Interpolated fragment texture coordinates (out)
out     vec3 v_position; // Vertex coordinates (out)


void main()
{
  gl_Position = mvp * vec4(position.x, position.y, position.z, 1.0);

  v_position = position;
  v_texcoord = texcoord;
}