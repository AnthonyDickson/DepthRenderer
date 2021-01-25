uniform sampler2D colourSampler; // Texture
uniform sampler2D depthSampler; // Depth Texture
varying vec2 v_texcoord;   // Interpolated fragment texture coordinates (in)

void main()
{
  gl_FragColor = texture2D(colourSampler, v_texcoord);
}