# Getting Started
## Setting Up Your Python Environment.
If you are using Conda, you can set up an environment called 'QuadRenderer' by using the provided `environment.yml`:
```shell
conda env create -f environment.yml
```
Otherwise, please ensure you have the packages listed in `environment.yml` installed in your Python environment of choice.

## Running
To run the program you will need a colour image and the corresponding depth map.
You can then call the main script via command:
```shell
python -m DepthRenderer <path/to/colour_image> <path/to/depth_map> -mesh-density 8
```
The argument `-mesh-density` controls the resolution of the generated mesh, each increase of 1 increases the number of 
triangles in the mesh by about 4x.
Run `python -m DepthRender -h` for more details about the available command line arguments.
