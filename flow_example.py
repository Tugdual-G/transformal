#!/usr/bin/env python3
import numpy as np
from functools import partial
import trimesh
from operators import mean_curvature
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.colors import LightSource
import matplotlib.animation as animation
from transform import flow, scalar_curvature, get_oriented_one_ring, apply_constraints


def plot_normals(ax, vertices, normals, length=10, color="r"):
    for i in range(normals.shape[0]):
        normalsegm = np.stack((vertices[i], vertices[i, :] + length * normals[i]))
        ax.plot(normalsegm[0, 0], normalsegm[0, 1], normalsegm[0, 2], color + "o")
        ax.plot(normalsegm[:, 0], normalsegm[:, 1], normalsegm[:, 2], color)


def vertex2face(trimesh, vval):
    assert vval.shape[0] == trimesh.vertices.shape[0]
    return np.mean(vval[trimesh.faces], axis=1)


def to_cmap(v, v_min=None, v_max=None):
    if v_min is None:
        v_min = v.min()

    if v_max is None:
        v_max = v.max()
    return plt.cm.plasma((v - v_min) / (v_max - v_min))


# Type hint to supress warnings of the editor since load can
# return different subclass of geometry
mesh: trimesh.Trimesh = trimesh.load("meshes/deform.ply")
vertices = mesh.vertices
nv = vertices.shape[0]
nf = mesh.faces.shape[0]
print("number of vertices", nv)

# center the object
centerofmass = np.mean(vertices, axis=0)
vertices[:] = vertices - centerofmass
# vol0 = mesh.volume

# The one ring is needed for many subsequent routines
one_ring = get_oriented_one_ring(mesh)

# Defines the plot
fig, ax = plt.subplots(1, 1, figsize=(8, 7), subplot_kw=dict(projection="3d"))
ax.axis("off")

# Use cuvature to define the coloration of the mesh's faces
k = scalar_curvature(mesh, mean_curvature(vertices, one_ring))
k_min = k.min()
k_max = k.max()
# Define the colors of the mesh
cm = to_cmap(vertex2face(mesh, k), k_min, k_max)

# Defines illumination
light = LightSource(90, 100)
fraction = 0.9
intensity = light.shade_normals(mesh.face_normals, fraction=fraction)
intensity.shape = (nf, 1, 1)
cm.shape = (nf, 1, 4)
cm = light.blend_overlay(cm, intensity=intensity)
cm.shape = (nf, 4)

# creates the poligon collection (mesh faces)
triangles = vertices[mesh.faces]
polycol = art3d.Poly3DCollection(
    triangles,
    facecolors=cm,
    edgecolors=(1, 1, 1, 0.2),
    alpha=1,
)


def init_func(ax, polycol, vertices, fig):
    """Intialize the plot for animation."""
    ax.add_collection(polycol)
    lim = (vertices.min(), vertices.max())
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*lim)
    ax.set_box_aspect([1, 1, 1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    fig.tight_layout(pad=0)


def animate(i):
    """Animation funcion"""
    dt = 0.5
    if i > 5:
        rho = -dt * scalar_curvature(mesh, mean_curvature(vertices, one_ring))
        flow(vertices, rho, one_ring)
        centerofmass = np.mean(vertices, axis=0)
        vertices[:] = vertices - centerofmass
        cm = to_cmap(vertex2face(mesh, k), k_min, k_max)
        intensity = light.shade_normals(mesh.face_normals, fraction=fraction)
        intensity.shape = (nf, 1, 1)
        cm.shape = (nf, 1, 4)
        cm = light.blend_overlay(cm, intensity=intensity)
        cm.shape = (nf, 4)
        polycol.set_facecolor(cm)
        polycol.set_verts(vertices[mesh.faces])
        return (polycol,)


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=60,
    interval=50,
    blit=False,
    repeat=False,
    init_func=partial(init_func, ax, polycol, vertices, fig),
)
plt.show()
# ani.save("animation.gif")
