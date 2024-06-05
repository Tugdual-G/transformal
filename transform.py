#!/usr/bin/env python3
from __future__ import annotations
from enum import verify
import numpy as np
from numpy.linalg import norm
import trimesh
import networkx as nx
from scipy import sparse
from scipy.linalg import issymmetric
from operators import (
    dirac_op,
    set_rings_order,
    mul_quatern,
    edges_div,
    quaternionic_laplacian_matrix,
    mean_curvature,
    symetric_delete,
)


def get_oriented_one_ring(mesh: trimesh.Trimesh) -> np.ndarray:
    nv = mesh.vertices.shape[0]
    # vertices = np.zeros((nv, 3), dtype=np.float64)
    vertices = mesh.vertices

    g = nx.from_edgelist(mesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    print(f"{len(one_ring) = }")

    n_entries = nv
    n_entries += sum(map(len, one_ring))
    one_ordered = np.zeros(n_entries, dtype=np.int64)

    k = 0
    for i, ring in enumerate(one_ring):
        one_ordered[k] = len(ring)
        k += 1
        cycle = nx.cycle_basis(g.subgraph(ring))
        if len(cycle) != 1:
            raise ValueError(f"vertex {i} has more than one cycle in its one-ring.")
        for ring_vert_idx in cycle[0]:
            one_ordered[k] = ring_vert_idx
            k += 1

    # one_ordered = [
    #     [len(ring)] + nx.cycle_basis(g.subgraph(ring))[0] for ring in one_ring
    # ]

    # one_ordered = np.array([x for r in one_ordered for x in r])

    # the values must be extracted before use, otherwise we get trash
    normals = np.zeros((nv, 3), dtype=np.float64)
    normals[:] = mesh.vertex_normals[:]

    set_rings_order(one_ordered, normals, vertices)
    return one_ordered


def scalar_curvature(mesh: trimesh.Trimesh, kN: np.ndarray) -> np.ndarray:
    return np.sum(mesh.vertex_normals * kN, axis=1)


def eigensolve(M: sparse.csc_matrix, v: np.ndarray):
    nv = v.shape[0]
    v[:] = 0
    v[::4] = 1.0
    # eigval, eigvect= sparse.linalg.eigsh(M, k=1, sigma=0, which='LM', tol=0)
    # v = eigvect[:, 0]
    for i in range(4):
        v[:] = sparse.linalg.spsolve(M, v)
        v.shape = (nv // 4, 4)
        v /= np.sqrt(np.sum(v**2))
        v.shape = (nv,)

    # v_res = M @ v
    # v_res.shape = (nv // 4, 4)
    # v_res /= np.sqrt(np.sum(v_res**2))
    # v_res.shape = (nv,)
    # print("eigen vector residual inf norm :", norm(v_res - v, ord=np.inf))
    v.shape = (nv // 4, 4)
    v /= np.mean(np.sqrt(np.sum(v**2, axis=1)))
    v.shape = (nv,)


# def symetric_delete2(i_del, L):
#     L_wp = np.delete(L,i_del, axis=0)
#     return np.delete(L_wp, i_del, axis=1)


def flow(vertices: np.ndarray, rho: np.ndarray, one_ring: np.ndarray):
    nv = vertices.shape[0]

    # building the operator (D - rho)
    d_i, d_j, d_data = dirac_op(vertices, one_ring, rho)
    X = sparse.csc_matrix((d_data, (d_i, d_j)), shape=(nv * 4, nv * 4))

    # finding the eigenvector lambd for the minimum eigenvalue
    lambd = np.zeros(4 * nv)
    eigensolve(X, lambd)

    # Applying the transform defined by lambd on the edge vectors
    div_e, constridx, constrpos = edges_div(vertices, lambd, one_ring)
    i_sorted = np.argsort(constridx)
    constridx = constridx[i_sorted]
    constrpos = constrpos[i_sorted]
    i1 = constridx[0]
    i2 = constridx[1]

    # Building the laplacian matrix to solve L x = div_e
    idx_i, idx_j, data = quaternionic_laplacian_matrix(vertices, one_ring)
    # csc for column slicing
    L = sparse.csc_matrix((data, (idx_i, idx_j)), shape=(nv * 4, nv * 4))

    # Applying constraint to the system
    # TODO make it work for bounded shapes
    for i, c in zip(constridx, constrpos):
        div_e[:] -= L[:, i * 4 : i * 4 + 4] @ c.T

    i_del = np.array([ic * 4 + i for ic in [i1, i2] for i in range(4)], dtype=np.int_)

    # Specific rows and columns need to get deleted to get a well posed problem,
    # this is specific to mesh without boundaries.
    idx_i, idx_j, data = symetric_delete(i_del, idx_i, idx_j, data, nv * 4)
    div_e = np.delete(div_e, i_del, axis=0)

    # Rebuild the csr matrix
    L = sparse.csc_matrix((data, (idx_i, idx_j)), shape=((nv - 2) * 4, (nv - 2) * 4))

    new_vertices = sparse.linalg.spsolve(L, div_e)

    new_vertices.shape = (nv - 2, 4)
    vertices[:i1, :] = new_vertices[:i1, 1:]
    vertices[i1, :] = constrpos[0, 1:]
    vertices[i1 + 1 : i2, :] = new_vertices[i1 : i2 - 1, 1:]
    vertices[i2, :] = constrpos[1, 1:]
    vertices[i2 + 1 :, :] = new_vertices[i2 - 1 :, 1:]


def transform(mesh: trimesh.Trimesh, rho: np.ndarray, one_ring: np.ndarray):
    vertices = mesh.vertices
    nv = vertices.shape[0]

    # building the operator (D - rho)
    d_i, d_j, d_data = dirac_op(vertices, one_ring, rho)
    X = sparse.csc_matrix((d_data, (d_i, d_j)), shape=(nv * 4, nv * 4))
    print(f"{np.abs(X-X.T).max() = }")

    # finding the eigenvector lambd for the minimum eigenvalue
    lambd = np.zeros(4 * nv)
    eigensolve(X, lambd)

    # Applying the transform defined by lambd on the edge vectors
    div_e, constridx, constrpos = edges_div(vertices, lambd, one_ring)
    i_sorted = np.argsort(constridx)
    constridx = constridx[i_sorted]
    constrpos = constrpos[i_sorted]
    i1 = constridx[0]
    i2 = constridx[1]

    # Building the laplacian matrix to solve L x = div_e
    idx_i, idx_j, data = quaternionic_laplacian_matrix(vertices, one_ring)
    # csc for row slicing
    L = sparse.csc_matrix((data, (idx_i, idx_j)), shape=(nv * 4, nv * 4))

    # Applying constraint to the system
    # TODO make it work for bounded shapes
    for i, c in zip(constridx, constrpos):
        div_e[:] -= L[:, i * 4 : i * 4 + 4] @ c.T

    i_del = np.array([ic * 4 + i for ic in [i1, i2] for i in range(4)], dtype=np.int_)

    # Specific rows and columns need to get deleted to get a well posed problem,
    # this is specific to mesh without boundaries.
    idx_i, idx_j, data = symetric_delete(i_del, idx_i, idx_j, data, nv * 4)
    div_e = np.delete(div_e, i_del, axis=0)

    # Rebuild the csr matrix
    L = sparse.csc_matrix((data, (idx_i, idx_j)), shape=((nv - 2) * 4, (nv - 2) * 4))
    print(f"{np.abs(L-L.T).max() = }")

    # norm_L = norm(L)
    # inv_L = inv(L)
    # norm_invA = norm(inv_L)
    # cond = norm_L*norm_invA
    # print(f"L condition number = {cond}")

    new_vertices = sparse.linalg.spsolve(L, div_e)
    residual = norm(L @ new_vertices - div_e)
    if residual > 1e-10:
        print(f"WARNING : {residual =}")

    new_vertices.shape = (nv - 2, 4)
    vertices[:i1, :] = new_vertices[:i1, 1:]
    vertices[i1, :] = constrpos[0, 1:]
    vertices[i1 + 1 : i2, :] = new_vertices[i1 : i2 - 1, 1:]
    vertices[i2, :] = constrpos[1, 1:]
    vertices[i2 + 1 :, :] = new_vertices[i2 - 1 :, 1:]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    from matplotlib.colors import LightSource

    def plot_normals(ax, vertices, normals, length=10, color="r"):
        for i in range(normals.shape[0]):
            normalsegm = np.stack((vertices[i], vertices[i, :] + length * normals[i]))
            ax.plot(normalsegm[0, 0], normalsegm[0, 1], normalsegm[0, 2], color + "o")
            ax.plot(normalsegm[:, 0], normalsegm[:, 1], normalsegm[:, 2], color)

    def vertex2face(mesh, vval):
        assert vval.shape[0] == mesh.vertices.shape[0]
        return np.mean(vval[mesh.faces], axis=1)

    def to_cmap(v, v_min=None, v_max=None):
        if not v_min:
            v_min = v.min()

        if not v_max:
            v_max = v.max()
        return plt.cm.plasma((v - v_min) / (v_max - v_min))

    # mesh = mesh.mesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0],[0,0,1]],
    #                     faces=[[0, 1, 3],[1,2,3],[2,0,3],[0,2,1]])
    mesh = trimesh.load("meshes/deform.ply")
    vertices = mesh.vertices
    nv = vertices.shape[0]
    print("number of vertices", nv)

    one_ring = get_oriented_one_ring(mesh)

    dt = 1.0
    n_iter = 1
    k = scalar_curvature(mesh, mean_curvature(vertices, one_ring))
    k_min = k.min()
    k_max = k.max()

    for it in range(n_iter):
        flow(vertices, -dt * k, one_ring)
        k = scalar_curvature(mesh, mean_curvature(vertices, one_ring))

    cm = to_cmap(vertex2face(mesh, k), k_min, k_max)
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), subplot_kw=dict(projection="3d"))

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
    ax.add_collection(pc)
    # plot_normals(ax, face_center, mesh.face_normals)
    xlim = (vertices[:, 0].min(), vertices[:, 0].max())
    ylim = (vertices[:, 1].min(), vertices[:, 1].max())
    zlim = (vertices[:, 2].min(), vertices[:, 2].max())

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1, 1, 1])

    fig.tight_layout(pad=0)
    plt.show()
