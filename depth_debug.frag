uniform sampler2D colourSampler; // Texture
uniform sampler2D depthSampler; // Depth Texture
uniform float     displacementFactor;
varying vec2 v_texcoord;   // Interpolated fragment texture coordinates (in)

void main()
{
  float displacement = displacementFactor * texture2D(depthSampler, v_texcoord).r;
  gl_FragColor = vec4(displacement, displacement, displacement, 1.0);
}