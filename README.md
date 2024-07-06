![example of flow](animation_crop.gif)
# Basic conformal transformation python module  

The module provides methods to find the nearest conformal transformation of a surface mesh for a given mean curvature change.
Curvature flow/fairing can be applied, such as shown in the script *flow_example.py* .

## References
The method applied here is really well explained in the two papers below,

CRANE, Keenan, PINKALL, Ulrich, and SCHRÖDER, Peter. Spin transformations of discrete surfaces. ACM SCRUFF 2011 papers, 2011, p. 1-10.

CRANE, Keenan, PINKALL, Ulrich, and SCHRÖDER, Peter. Robust fairing via conformal curvature flow. ACM Transactions on Graphics (TOG), 2013, vol. 32, no 4, p. 1-10.

## Presentation
### Conformal transformations
Conformal transformations locally preserves the angles of the surfaces they are applied on. 
For example if two intersecting lines are drawn onto a surface, those lines will keep the same angle of intersection after the transformation.
This propriety is desirable when one wants to fair a mesh without altering the mesh quality.

### Dirac operator and quaternionic transform
These transformations can be determined using the eigenvectors of the Dirac operator, or more precisely,
using the eigenvectors of $D-\rho$ with $\rho$ the given curvature change over the surface.
The Dirac operator, $D$, is defined as $D^2 = \Delta$ , with $\Delta$ being in our case the Laplace-Beltrami operator. 
The components of the eigenvectors are quaternions defining the transformation (scaling and rotation) which is applied to the tangent vectors of the surface.
Then the vertices position can be computed using the transformed tangent vectors (up to a constant position).

## Computation methods
### Eigensolver
The Dirac operator is computed over the mesh and stored as a compressed sparse column matrix.
The eigensolver use a LU decomposition of $D-\rho$ and the iteration of a linear solver to find the eigenvector corresponding to the smallest amplitude eigenvalue :

```py
while norm(residual) > tolerance :
    eigenvector = solve_LU(L, U, eigenvector) # solve the linear equation LU x = eigenvector
    eigenvector /= norm(eigenvector) # ensure the vector does not grow or shrink to much
```

This process is the main bottleneck of the code, and could be improved by implementing a solver using the quaternionic multiplication operator rather than traditional multiplication.
Implementing a real quaternionic solver would reduce the amount of memory transaction, since for now, quaternions are represented by 4x4 matrix to work with traditional solvers. 

Otherwise, a matrix-free methods such as LOBPCG could be used... with a quaternionic matrix.

__Stability:__

When the mesh's triangles areas gets smaller, the eigenvalue problem becomes ill-conditioned. 
However, the iteration method seems to withstand this problem pretty well in comparison to other algorithms.


### Finding vertices positions
After applying the transform to the tangent vectors of the mesh (the edges of the mesh in practice),
we solve the following equation for $x$, 
$$\Delta x = \nabla \cdot e$$
where, $x$ stands for the vertices position and $e$ for the tangents.

Since we operate on closed surfaces, the system of equation is singular.
To solve this issue we simply choose two vertices to constrain the position and orientation of our surface.
This result in a fully constrained problem, which is solved using sparse matrix representation.

## Implementation
This project use a procedural style which seem better suited for these kinds of problems. 

## Requirements
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
so before running the scripts for the first time, you should compile with the command :
        
        cd transformal/; python operators_core.py

code example:

```python
import trimesh
from transformal.transform import (
    transform,
    get_oriented_one_ring,
)
mesh: trimesh.Trimesh = trimesh.load("mesh.ply")

# define the change in curvature
rho = define_rho()

# Get the one-ring of the mesh 
one_ring = get_oriented_one_ring(mesh)

# Apply the transformation to the mesh
transform(mesh, rho, one_ring)

```

![example](ballfig.png)

