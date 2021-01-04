# Getting Started
## Setting Up Your Python Environment.
If you are using Conda, you can set up an environment called 'QuadRenderer' by using the provided `environment.yml`:
```shell
conda env create -f environment.yml
```
Otherwise, please ensure you have the packages listed in `environment.yml` installed in your Python environment of choice.

**Important:** For Windows users, please see the notes below for installing the OpenGL packages. 
## OpenGL For Python on Windows
The official Python package for OpenGL is broken. You need to install PyOpenGL from the wheel file from here: 
https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl.