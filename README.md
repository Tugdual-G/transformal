![example of flow](animation_crop.gif)

Basic conformal transformation script using 
_CRANE, Keenan, PINKALL, Ulrich, and SCHRÖDER, Peter. Spin transformations of discrete surfaces. ACM SCRUFF 2011 papers, 2011, p. 1-10._

It is possible to apply a curvature flow, such as shown in the script *flow_example.py* .
## Dependencies
- matplotlib
- networkx
- numba
- numpy
- scipy
- setuptools
- trimesh

        pip install matplotlib networkx numba numpy scipy setuptools trimesh


## Usage
The project use numba cc to compile the modules into a dynamic .so library, 
so before running the scripts for the first time you should compile with the command :
        
        python compile_modules.py

![example](ballfig.png)

