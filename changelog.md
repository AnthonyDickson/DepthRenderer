# 2021-02-04
- Improve frame capture speed by using pixel buffer object.
- Add AsyncVideoWriter class to do video rendering in separate thread and improve renderer frame rate.
- Move code into package. 

- Fix bug with depth rendering by displacing mesh vertices rather than through shaders.
- Add basic animations.
- Change debug shaders to show the z coordinates of the mesh vertices.
- Add async image writer and basic video writer.
- Abstract textures, meshes and cameras into their own classes.