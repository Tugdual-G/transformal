#!/usr/bin/env python3
"""
This script uses a prescribed change in curvature to get
the nearest conformal transformation of a surface mesh
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.colors import LightSource
import trimesh
import numpy as np
from transformal.transform import (
    transform,
    get_oriented_one_ring,
    scalar_curvature,
    apply_constraints,
    integrate,
)
from transformal.operators import mean_curvature


def vertex2face(trimesh, vval):
    """
    Interpolate values defined at the vertices to the
    face center.
    """
    assert vval.shape[0] == trimesh.vertices.shape[0]
    return np.mean(vval[trimesh.faces], axis=1)


# Type hint to supress warnings of the editor since load can
# return different subclass of geometry
mesh: trimesh.Trimesh = trimesh.load("meshes/sphere.ply")

vertices = mesh.vertices
nv = vertices.shape[0]

# ++++++++++++++++++++++++++++++++++++++++
#
#      Definition of the curvature change
#
#   rho > 0  create a concave deformation
#   rho < 0 create a convex deformation
#
#   kN, the mean curvature is defined as the
#   local average of the principal curvatures,
#   not the average curvature over the mesh !
#
# ++++++++++++++++++++++++++++++++++++++++
# Use the mean curvature as a reference value to choose
# resonable curvature change values
one_ring = get_oriented_one_ring(mesh)
kN = mean_curvature(vertices, one_ring["data"], one_ring["max_length"])
k = scalar_curvature(mesh, kN)
# Average mean curvature over the mesh
mk = integrate(mesh, k) / integrate(mesh, np.ones(nv))


# Defines positions on the sphere where the curvature will change
R = 50
pts = []
pts += [[np.pi / 4, 2 * np.pi / 6]]
pts += [[-3 * np.pi / 4, 0]]
pts += [[-np.pi / 4, -np.pi / 6]]
pts += [[np.pi / 4, -np.pi / 6]]
ampl = np.array([-18, -18, -18, -18]) * abs(mk)
rad = [50.0, 50.0, 50.0, 50]

rho = np.zeros(nv, dtype=np.float64)  # rho is the curvature change
for i, pt in enumerate(pts):
    pt_cart = np.array(
        [
            R * np.cos(pt[1]) * np.cos(pt[0]),
            R * np.cos(pt[1]) * np.sin(pt[0]),
            R * np.sin(pt[1]),
        ]
    )
    dist = np.linalg.norm(vertices - pt_cart, axis=1)
    rho += ampl[i] * np.exp(-(dist**2) / rad[i])

# just Ensure that the change in curvature is compatible
# with a few invariants.
apply_constraints(mesh, rho)


# ++++++++++++++++++++++++++++++++++++++++++++
# Plots the before/after mesh
# ++++++++++++++++++++++++++++++++++++++++++++

fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={"projection": "3d"})
ax1, ax2 = axs

# colormap
rho_fc = vertex2face(mesh, rho)
rho_norm = (rho_fc - rho_fc.min()) / np.ptp(rho_fc)
cm = plt.cm.plasma(rho_norm)

# Mesh and lights (before the transform)
triangles = vertices[mesh.faces]
light = LightSource(90, 100)
pc = art3d.Poly3DCollection(
    triangles,
    facecolors=cm,
    shade=True,
    edgecolors=(1, 1, 1, 0.2),
    alpha=1,
    lightsource=light,
)
ax1.add_collection(pc)

# to keep a nice 3D aspect-ratio
axis_lim = [(vertices.min(), vertices.max())]

# ++++++++++++++++++++++++++++++
# Applying the Transform
# +++++++++++++++++++++++++++++
transform(vertices, rho, one_ring)

# Plots the transform
kN = mean_curvature(vertices, one_ring["data"], one_ring["max_length"])
k = scalar_curvature(mesh, kN)

k_fc = vertex2face(mesh, k)
k_norm = (k_fc - k_fc.min()) / np.ptp(k_fc)
cm = plt.cm.plasma(k_norm)

triangles = vertices[mesh.faces]
pc = art3d.Poly3DCollection(
    triangles,
    facecolors=cm,
    shade=True,
    edgecolors=(1, 1, 1, 0.2),
    alpha=1,
    lightsource=light,
)
ax2.add_collection(pc)


# aspect ratio
axis_lim += [(vertices.min(), vertices.max())]
for i, ax in enumerate(axs):
    ax.set_xlim(*axis_lim[i])
    ax.set_ylim(*axis_lim[i])
    ax.set_zlim(*axis_lim[i])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

fig.tight_layout(pad=0)
plt.show()
