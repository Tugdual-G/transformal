#!/usr/bin/env python3
import numpy as np
from functools import partial
import trimesh
from operators import mean_curvature
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.colors import LightSource
import matplotlib.animation as animation
from transform import flow, scalar_curvature, get_oriented_one_ring


def plot_normals(ax, vertices, normals, length=10, color="r"):
    for i in range(normals.shape[0]):
        normalsegm = np.stack((vertices[i], vertices[i, :] + length * normals[i]))
        ax.plot(normalsegm[0, 0], normalsegm[0, 1], normalsegm[0, 2], color + "o")
        ax.plot(normalsegm[:, 0], normalsegm[:, 1], normalsegm[:, 2], color)


def vertex2face(trimesh, vval):
    assert vval.shape[0] == trimesh.vertices.shape[0]
    return np.mean(vval[trimesh.faces], axis=1)


def to_cmap(v, v_min=None, v_max=None):
    if not v_min:
        v_min = v.min()

    if not v_max:
        v_max = v.max()
    return plt.cm.plasma((v - v_min) / (v_max - v_min))


trimesh = trimesh.load("meshes/deform.ply")
vertices = trimesh.vertices
nv = vertices.shape[0]
nf = trimesh.faces.shape[0]
print("number of vertices", nv)

centerofmass = np.mean(vertices, axis=0)
vertices[:] = vertices - centerofmass
vol0 = trimesh.volume
one_ring = get_oriented_one_ring(trimesh)

k = scalar_curvature(trimesh, mean_curvature(vertices, one_ring))
k_min = k.min()
k_max = k.max()

fig, ax = plt.subplots(1, 1, figsize=(8, 7), subplot_kw=dict(projection="3d"))
ax.axis("off")


fraction = 0.8
cm = to_cmap(vertex2face(trimesh, k), k_min, k_max)
light = LightSource(90, 100)
intensity = light.shade_normals(trimesh.face_normals, fraction=fraction)
intensity.shape = (nf, 1, 1)
cm.shape = (nf, 1, 4)
cm = light.blend_overlay(cm, intensity=intensity)
cm.shape = (nf, 4)

fig.tight_layout(pad=0)
triangles = vertices[trimesh.faces]
pc = art3d.Poly3DCollection(
    triangles,
    facecolors=cm,
    edgecolors=(1, 1, 1, 0.2),
    alpha=1,
    lightsource=light,
)


def init_func(ax, pc, trimesh, fig):

    ax.add_collection(pc)
    xlim = (vertices[:, 0].min(), vertices[:, 0].max())
    ylim = (vertices[:, 1].min(), vertices[:, 1].max())
    zlim = (vertices[:, 2].min(), vertices[:, 2].max())
    lim = (vertices.min(), vertices.max())
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_zlim(*lim)
    ax.set_box_aspect([1, 1, 1])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    fig.tight_layout(pad=0)


dt = 0.4

polycol = pc


def animate(i):
    if i > 5:
        k = scalar_curvature(trimesh, mean_curvature(vertices, one_ring))
        flow(vertices, -dt * k, one_ring)
        dist = (vol0 / trimesh.volume) ** (1 / 3)
        centerofmass = np.mean(vertices, axis=0)
        vertices[:] = dist * (vertices - centerofmass)
        cm = to_cmap(vertex2face(trimesh, k), k_min, k_max)
        intensity = light.shade_normals(trimesh.face_normals, fraction=fraction)
        intensity.shape = (nf, 1, 1)
        cm.shape = (nf, 1, 4)
        cm = light.blend_overlay(cm, intensity=intensity)
        cm.shape = (nf, 4)
        # cm[:] *= cm * intensity[:, np.newaxis]
        polycol.set_facecolor(cm)
        polycol.set_verts(vertices[trimesh.faces])
        return (polycol,)


ani = animation.FuncAnimation(
    fig,
    animate,
    frames=60,
    interval=60,
    blit=False,
    repeat=False,
    init_func=partial(init_func, ax, pc, trimesh, fig),
)
plt.show()
# ani.save("animation.gif")
