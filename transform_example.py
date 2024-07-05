#!/usr/bin/env python3
"""
TODO
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
    assert vval.shape[0] == trimesh.vertices.shape[0]
    return np.mean(vval[trimesh.faces], axis=1)


# Type hint to supress warnings of the editor since load can
# return different subclass of geometry
mesh: trimesh.Trimesh = trimesh.load("meshes/sphere.ply")

vertices = mesh.vertices
nv = vertices.shape[0]
print("number of vertices", nv)


one_ring = get_oriented_one_ring(mesh)
kN = mean_curvature(vertices, one_ring)
k = scalar_curvature(mesh, kN)
mk = integrate(mesh, k) / integrate(mesh, np.ones(nv))
print("mean curvature =", mk)


R = 50
pts = []
pts += [[np.pi / 4, 2 * np.pi / 6]]
pts += [[-3 * np.pi / 4, 0]]
pts += [[-np.pi / 4, -np.pi / 6]]
pts += [[np.pi / 4, -np.pi / 6]]
ampl = np.array([-18, -18, -18, -18]) * abs(mk)
rad = [50.0, 50.0, 50.0, 50]

rho = np.zeros(nv, dtype=np.float64)
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

rho0 = rho.copy()
apply_constraints(mesh, rho)
print(f"{np.abs(rho0 - rho).max()/abs(mk) = }")

rho_fc = vertex2face(mesh, rho)
rho_norm = (rho_fc - rho_fc.min()) / np.ptp(rho_fc)
cm = plt.cm.plasma(rho_norm)
fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw=dict(projection="3d"))
ax1, ax2 = axs


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
# plot_normals(ax, face_center, mesh.face_normals)
xlim = [(vertices[:, 0].min(), vertices[:, 0].max())]
ylim = [(vertices[:, 1].min(), vertices[:, 1].max())]
zlim = [(vertices[:, 2].min(), vertices[:, 2].max())]

transform(mesh, rho, one_ring)

kN = mean_curvature(vertices, one_ring)
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


xlim += [(vertices[:, 0].min(), vertices[:, 0].max())]
ylim += [(vertices[:, 1].min(), vertices[:, 1].max())]
zlim += [(vertices[:, 2].min(), vertices[:, 2].max())]

for i, ax in enumerate(axs):
    ax.set_xlim(*xlim[i])
    ax.set_ylim(*ylim[i])
    ax.set_zlim(*zlim[i])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

# mesh.visual.face_colors = cm
# mesh.export("deform.ply")
fig.tight_layout(pad=0)
# fig.savefig("ballfig.png")
plt.show()
