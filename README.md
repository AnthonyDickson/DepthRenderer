# Getting Started
## Setting Up Your Python Environment.
If you are using Conda, you can set up an environment called 'QuadRenderer' by using the provided `environment.yml`:
```shell
conda env create -f environment.yml
```
Otherwise, please ensure you have the packages listed in `environment.yml` installed in your Python environment of choice.

**Important:** Please see the notes below for installing the OpenGL packages for your operating system.

## Issues With Installing GLUT
The official Python package for OpenGL available via Pip is missing files needed for GLUT to correctly function.

### Windows
For Windows, you can install PyOpenGL using the wheel file from here: 
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl.
Download the appropriate version and then point Pip to the downloaded file for installation:
```shell
pip install <path/to/wheel/file.whl>
```

### OpenGL For Python on Linux
For Linux, you can install the GLUT package available via the `apt` command:
```shell
sudo apt install freeglut3-dev
```

## Running
To run the program you will need a colour image and the corresponding depth map.
You can then call the main script via command:
```shell
python -m DepthRenderer <path/to/colour_image> <path/to/depth_map> -mesh-density 8
```
The argument `-mesh-density` controls the resolution of the generated mesh, each increase of 1 increases the number of 
triangles in the mesh by about 4x.
Run `python -m DepthRender -h` for more details about the available command line arguments.
