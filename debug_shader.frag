#version 460

uniform sampler2D colourSampler; // Texture

varying vec2 v_texcoord; // Interpolated fragment texture coordinates (in)
in      vec3 v_position; // Vertex coordinates (in)

void main()
{
    vec4 colour = texture2D(colourSampler, v_texcoord);
    gl_FragColor = vec4(vec3(v_position.z), colour.a);
}