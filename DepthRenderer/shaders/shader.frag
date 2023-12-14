#version 330

uniform sampler2D Texture;

in vec2 frag_texcoord; // Interpolated fragment texture coordinates (in)

void main()
{
  gl_FragColor = texture2D(Texture, frag_texcoord);
}