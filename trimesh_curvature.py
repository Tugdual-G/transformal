#!/usr/bin/env python3
import numpy as np
import trimesh
import networkx as nx
from operators import mean_curvature
from transform import get_oriented_one_ring


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from mpl_toolkits.mplot3d import Axes3D, art3d

    def plot_normals(ax, vertices, normals, length=10, color="r"):
        for i in range(normals.shape[0]):
            normalsegm = np.stack((vertices[i], vertices[i, :] + length * normals[i]))
            ax.plot(normalsegm[0, 0], normalsegm[0, 1], normalsegm[0, 2], color + "o")
            ax.plot(normalsegm[:, 0], normalsegm[:, 1], normalsegm[:, 2], color)

    def vertex2face(trimesh, vval):
        assert vval.shape[0] == trimesh.vertices.shape[0]
        return np.mean(vval[trimesh.faces], axis=1)

    trimesh = trimesh.load("meshes/knobitle.ply")
    vertices = trimesh.vertices

    one_ring = get_oriented_one_ring(trimesh)
    kN = mean_curvature(vertices, one_ring)
    normals = trimesh.vertex_normals
    k = np.sum(normals * kN, axis=1)
    print(f"{ k.min() =} , { k.max() = }")

    face_k_norm = vertex2face(trimesh, k)
    face_k_norm = (face_k_norm - face_k_norm.min()) / (np.ptp(face_k_norm))
    start = 0.15
    face_k_norm = start + (1 - start - 0.001) * face_k_norm
    cm = plt.cm.inferno(face_k_norm)

    triangles = vertices[trimesh.faces]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    pc = art3d.Poly3DCollection(
        triangles, facecolors=cm, alpha=1, shade=True, edgecolors=(1, 1, 1, 0.2)
    )
    ax.add_collection(pc)

    plot_normals(ax, vertices, -kN, 2000)

    xlim = vertices[:, 0].min(), vertices[:, 0].max()
    ylim = vertices[:, 1].min(), vertices[:, 1].max()
    zlim = vertices[:, 2].min(), vertices[:, 2].max()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    plt.show()
