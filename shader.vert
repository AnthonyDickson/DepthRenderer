uniform mat4   mvp;           // The model, view and projection matrices as one matrix.
attribute vec3 position;      // Vertex position
attribute vec2 texcoord;   // Vertex texture coordinates
varying vec2   v_texcoord;       // Interpolated fragment texture coordinates (out)

uniform sampler2D colourSampler; // Texture
uniform sampler2D depthSampler; // Depth Texture
uniform float     displacementFactor;
// TODO: Add debug shader that visualises the z-coordinate and/or the displacement figure as the fragment colour.
void main()
{
  float displacement = displacementFactor * tex2D(depthSampler, v_texcoord).r / 255.0;
  gl_Position = mvp * vec4(position.x, position.y, position.z + displacement, 1.0);
  v_texcoord = texcoord;
}