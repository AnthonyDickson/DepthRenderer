# 2021-02-11
- Migrate window code from GLUT to GLFW.

# 2021-02-10
- Add a script for automatically creating video datasets for a given image and series of depth maps.
- Add back in displacement factor command line argument.
- Add function for getting the frame pixel data by value rather than reference which fixes a bug with async image/video 
  writers that are slower than the frame rate.
- Add function for resetting animations, making reusing the same animations easier.
- Add functions for copying textures and meshes.
- Add mode for rendering as fast as possible, while simulating fixed time step updates.
- The mesh in a renderer can be changed on the fly.
- Add a frame timer util.

# 2021-02-08
- Add callback to handle the program exiting.
- Load images as RGBA and add option to mask pixels.
- Fix bug with mesh generation to make sure that the mesh is the correct aspect ratio and is centered in the window.
- Improve documentation.
- Change sample images to ones from NYU dataset.

# 2021-02-04
- Improve frame capture speed by using pixel buffer object.
- Add AsyncVideoWriter class to do video rendering in separate thread and improve renderer frame rate.
- Move code into package. 

- Fix bug with depth rendering by displacing mesh vertices rather than through shaders.
- Add basic animations.
- Change debug shaders to show the z coordinates of the mesh vertices.
- Add async image writer and basic video writer.
- Abstract textures, meshes and cameras into their own classes.