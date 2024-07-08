#!/usr/bin/env python3
"""
This module define the higher level functions:
The most relevant one is,

    transform()  : find the nearest conformal transform

"""
from __future__ import annotations
import numpy as np
from numpy.linalg import norm
import trimesh
import networkx as nx
from scipy import sparse
from transformal.operators import (
    dirac_op,
    set_rings_order,
    edges_div,
    quaternionic_laplacian_matrix,
    symetric_delete,
)


def get_oriented_one_ring(mesh: trimesh.Trimesh) -> dict:
    """
    Returns :
    - per vertices one-ring list,
    - length of the longest one-ring,
    - total number of one-ring elements.

    The oriented one ring is stored as a contiguous list of variable size sublists.
    The normals of the mesh are used to order the one-ring in
    a counter-clockwise order.
    The one-ring is stored as follow,
    the first element of the list is the lenght of the one ring for each vertice,
    followed by the vertex indices.

    |vertice 0                   |vertice 1             |vertice 2        ...
    [n_0, i_00, i_01, i_02, i_03, n_1, i_10, i_11, i_12, n_2, i_20, i_21, ...]

    """

    one_ring = {}
    nv = mesh.vertices.shape[0]
    vertices = mesh.vertices

    g = nx.from_edgelist(mesh.edges_unique)
    one_ring_data = [list(g[i].keys()) for i in range(nv)]
    one_ring["max_length"] = max(map(len, one_ring_data))

    # total number of one-ring elements
    one_ring["n_total_elements"] = sum(map(len, one_ring_data))
    one_ordered = np.zeros(one_ring["n_total_elements"] + nv, dtype=np.int64)

    k = 0
    for i, ring in enumerate(one_ring_data):
        one_ordered[k] = len(ring)
        k += 1
        cycle = nx.cycle_basis(g.subgraph(ring))
        if len(cycle) > 1:
            raise ValueError(f"vertex {i} has more than one cycle in its one-ring.")

        if len(cycle) < 1:
            # In case we encounter an edge
            raise ValueError(f"The one-ring of vertex {i} is empty.")

        for ring_vert_idx in cycle[0]:
            one_ordered[k] = ring_vert_idx
            k += 1

    # this is necessary to initialize  trimesh vertex_normals
    # otherwise, we get trash values
    normals = np.zeros((nv, 3), dtype=np.float64)
    normals[:] = mesh.vertex_normals[:]

    set_rings_order(one_ordered, normals, vertices)

    one_ring["data"] = one_ordered

    return one_ring


def scalar_curvature(mesh: trimesh.Trimesh, kn: np.ndarray) -> np.ndarray:
    """Returns the projection of kn on the mesh normal."""
    return np.sum(mesh.vertex_normals * kn, axis=1)


def integrate(mesh: trimesh.Trimesh, f: np.ndarray):
    """Integration over the surface of the mesh using the trapezoidal rule."""
    trapezes = np.mean(f[mesh.faces], axis=1) * mesh.area_faces
    return np.sum(trapezes)


def orthonormalize(mesh: trimesh.Trimesh, vects: list):
    """Transform the provided list of vectors into an orthonormal basis."""
    for i, v in enumerate(vects):
        for j in range(i):
            v[:] -= integrate(mesh, v * vects[j]) * vects[j]
        v[:] /= np.sqrt(integrate(mesh, v * v))


def apply_constraints(mesh: trimesh.Trimesh, rho: np.ndarray):
    """Discard the composants of a vector which does not abide to a constraint."""
    constraints = [
        np.ones(mesh.vertices.shape[0]),
        mesh.vertex_normals[:, 0].copy(),
        mesh.vertex_normals[:, 1].copy(),
        mesh.vertex_normals[:, 2].copy(),
    ]
    orthonormalize(mesh, constraints)
    for constraint in constraints:
        rho[:] -= integrate(mesh, rho * constraint) * constraint


def eigensolve(M: sparse.csc_array, v: np.ndarray):
    """Finds the eigenvector corresponding to the smallest eigen value."""
    nv = v.shape[0]
    v[:] = 0
    v[::4] = 1.0
    LU = sparse.linalg.splu(M)
    for _ in range(5):
        v[:] = LU.solve(v)
        v[:] /= norm(v)

    v.shape = (nv // 4, 4)
    v[:] /= np.mean(np.sqrt(np.sum(v**2, axis=1)))
    v.shape = (nv,)


def transform(vertices: np.ndarray, rho: np.ndarray, one_ring: dict):
    nv = vertices.shape[0]

    # building the operator (X = D - rho)
    d_i, d_j, d_data = dirac_op(
        vertices,
        one_ring["data"],
        one_ring["max_length"],
        (nv + one_ring["n_total_elements"]) * 16,
        rho,
    )
    X = sparse.csc_array((d_data, (d_i, d_j)), shape=(nv * 4, nv * 4))

    # finding the eigenvector lambd for the minimum eigenvalue
    lambd = np.zeros(4 * nv)
    eigensolve(X, lambd)

    # Applying the transform defined by lambd on the edge vectors
    # Retrives a position constraint for two vertices, and their indices
    # in the vertice list.
    div_e, constridx, constrpos = edges_div(
        vertices, lambd, one_ring["data"], one_ring["max_length"]
    )
    i_sorted = np.argsort(constridx)
    constridx = constridx[i_sorted]
    constrpos = constrpos[i_sorted]
    i1 = constridx[0]
    i2 = constridx[1]

    # Building the laplacian matrix to solve L x = div_e
    idx_i, idx_j, data = quaternionic_laplacian_matrix(
        vertices,
        one_ring["data"],
        one_ring["max_length"],
        (nv + one_ring["n_total_elements"]) * 16,
    )
    # csc for row slicing
    L = sparse.csc_array((data, (idx_i, idx_j)), shape=(nv * 4, nv * 4))

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
    L = sparse.csc_array((data, (idx_i, idx_j)), shape=((nv - 2) * 4, (nv - 2) * 4))

    new_vertices = sparse.linalg.spsolve(L, div_e)

    new_vertices.shape = (nv - 2, 4)
    vertices[:i1, :] = new_vertices[:i1, 1:]
    vertices[i1, :] = constrpos[0, 1:]
    vertices[i1 + 1 : i2, :] = new_vertices[i1 : i2 - 1, 1:]
    vertices[i2, :] = constrpos[1, 1:]
    vertices[i2 + 1 :, :] = new_vertices[i2 - 1 :, 1:]
