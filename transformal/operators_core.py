#!/usr/bin/env python3
"""

This modules contains the lower level differential/graph operators.

Run this module to compile it before use.

"""

from numba.pycc import CC
from numba import njit
import numpy as np
from numpy.linalg import norm

cc = CC("operators")
cc.verbose = True


@njit
@cc.export("mul_quatern", "f8[::1](f8[::1], f8[::1])")
def mul_quatern(u, v):
    """Quaternion multiplication."""
    w0, x0, y0, z0 = u
    w1, x1, y1, z1 = v
    return np.array(
        [
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ],
        dtype=np.float64,
    )


@njit
def fill_quaternionic_matrix_block(
    i,
    j,
    sparse_idx,
    quaternion,
    idx_i,
    idx_j,
    data,
):
    """
    Fill a given quaternion block of the sparse matrix entries list.
    Returns the incremented idex of the sparse list.
    """
    block_ij = np.array(
        [
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            [quaternion[1], quaternion[0], -quaternion[3], quaternion[2]],
            [quaternion[2], quaternion[3], quaternion[0], -quaternion[1]],
            [quaternion[3], -quaternion[2], quaternion[1], quaternion[0]],
        ]
    )

    for m in range(4):
        for n in range(4):
            idx_i[sparse_idx] = i * 4 + m
            idx_j[sparse_idx] = j * 4 + n
            data[sparse_idx] = block_ij[m, n]
            sparse_idx += 1
    return sparse_idx


@cc.export(
    "set_rings_order",
    "(i8[::1], f8[:,::1],f8[:,::1])",
)
def set_rings_order(one_ring, normals, vertices):
    """
    Sort the one-ring indices in an anti-clockwise order
    around the normal, with the normal pointing toward the viewer.
    """
    ei = np.zeros(3, dtype=np.float64)
    ej = np.zeros(3, dtype=np.float64)

    i0 = 0
    vi = 0
    while i0 < one_ring.shape[0]:
        ring_length = one_ring[i0]
        sign = 1
        if ring_length > 1:
            j0 = one_ring[i0 + 1]
            j1 = one_ring[i0 + 2]
            v = vertices[vi, :]
            vertex_normal = normals[vi, :]
            ei[:] = vertices[j0, :] - v
            ej[:] = vertices[j1, :] - v
            sign = int(np.dot(vertex_normal, np.cross(ei, ej)))

        if sign < 0:
            for j in range(ring_length // 2):
                or_tmp = one_ring[i0 + j + 1]
                one_ring[i0 + j + 1] = one_ring[i0 + ring_length - j]
                one_ring[i0 + ring_length - j] = or_tmp

        i0 += one_ring[i0] + 1
        vi += 1

    if vi != normals.shape[0]:
        raise ValueError(
            "Warning wrong number of vertex read in set_rings_order: vi = ",
            vi,
            ", normals.shape =",
            normals.shape,
        )


@cc.export(
    "dirac_op",
    "Tuple((i8[::1],i8[::1],f8[::1]))(f8[:,::1], i8[::1], i8, i8,f8[::1])",
)
def dirac_op(vertices, one_ring, max_one_ring_length, n_entries, rho):
    """
    Returns the indices and values needed to construct the sparse
    representation of the dirac operator on the mesh.
    """

    ei = np.zeros(4, dtype=np.float64)  # quaternion edge representation
    ej = np.zeros(4, dtype=np.float64)
    x_ii = np.zeros(4, dtype=np.float64)  # temporary storage for diagonal entries

    ring_vert = np.zeros((max_one_ring_length, 3), dtype=np.float64)

    # sparse matrix indices
    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries, dtype=np.float64)

    sparse_idx = 0  # current indice in the idx_i, idx_j, and data arrays
    one_ring_i0 = 0  # first element of each one-ring sublist
    for i in range(vertices.shape[0]):
        # TODO on't work for bounded domain
        ring_nv = one_ring[one_ring_i0]
        if ring_nv > 1:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[one_ring_i0 + r_i + 1]]

            x_ii[:] = 0.0
            for k in range(ring_nv):
                j = one_ring[one_ring_i0 + k + 1]
                ei[1:] = ring_vert[k] - ring_vert[(k - 1) % ring_nv]
                ej[1:] = ring_vert[(k - 1) % ring_nv] - vertices[i, :]

                # Area of the dual cell
                area_x2 = norm(np.cross(ei[1:], ej[1:]))

                x_ij = (  # temporary storage for off-diag entries
                    -mul_quatern(ei, ej) / (2 * area_x2)
                    + (rho[i] * ej - rho[j] * ei) / 6.0
                )
                x_ij[0] += rho[i] * rho[j] * area_x2 / 18.0

                ei[1:] = ring_vert[(k + 1) % ring_nv] - ring_vert[k]
                ej[1:] = vertices[i, :] - ring_vert[(k + 1) % ring_nv]

                area_x2 = norm(np.cross(ei[1:], ej[1:]))
                x_ij += (
                    -mul_quatern(ei, ej) / (2 * area_x2)
                    + (rho[i] * ej - rho[j] * ei) / 6.0
                )
                x_ij[0] += rho[i] * rho[j] * area_x2 / 18.0

                sparse_idx = fill_quaternionic_matrix_block(
                    i, j, sparse_idx, x_ij, idx_i, idx_j, data
                )

                x_ii -= mul_quatern(ei, ei) / (2 * area_x2)
                x_ii[0] += rho[i] * rho[i] * area_x2 / 18.0

            sparse_idx = fill_quaternionic_matrix_block(
                i, i, sparse_idx, x_ii, idx_i, idx_j, data
            )
            one_ring_i0 += ring_nv + 1

    return idx_i, idx_j, data


@cc.export(
    "edges_div",
    "Tuple((f8[::1],i8[::1],f8[:,::1]))(f8[:,::1], f8[::1], i8[::1], i8)",
)
def edges_div(vertices, lambd, one_ring, max_one_ring_length):
    """
    Computes the divergence of the mesh edges after applying the lambd
    quaternionic transform to them.
    Return the divergence of the edges as well as a position constraint for two vertices.
    """
    nv = vertices.shape[0]

    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()  # the conjugate of lambda

    constraint_idx = np.zeros(2, dtype=np.int_)
    constraint_pos = np.zeros((2, 4), dtype=np.float64)

    assert lambd.shape[0] % 4 == 0
    for i in range(lambd.shape[0] // 4):
        lambdc[4 * i + 1 : 4 * i + 4] *= -1

    ring_vert = np.zeros((max_one_ring_length, 3), dtype=np.float64)

    i_const = 0
    div = np.zeros((nv * 4))
    one_ring_i0 = 0
    for i in range(nv):
        ring_nv = one_ring[one_ring_i0]
        if ring_nv > 1:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[one_ring_i0 + r_i + 1]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert - vertices[i, :]

            for k in range(ring_nv):
                e1 = -edges_vect[(k - 1) % ring_nv]
                o1 = edges_vect[k] + e1
                cos1 = np.dot(e1, o1)
                sin1 = norm(np.cross(o1, e1))
                cot1 = cos1 / sin1

                e2 = -edges_vect[(k + 1) % ring_nv]
                o2 = edges_vect[k] + e2
                cos2 = np.dot(e2, o2)
                sin2 = norm(np.cross(e2, o2))

                cot2 = cos2 / sin2

                j = one_ring[one_ring_i0 + k + 1]
                eij[1:] = edges_vect[k]
                e_new = (
                    1
                    / 3
                    * mul_quatern(
                        lambdc[i * 4 : i * 4 + 4],
                        mul_quatern(eij, lambd[i * 4 : i * 4 + 4]),
                    )
                )
                e_new += (
                    1
                    / 6
                    * mul_quatern(
                        lambdc[i * 4 : i * 4 + 4],
                        mul_quatern(eij, lambd[j * 4 : j * 4 + 4]),
                    )
                )
                e_new += (
                    1
                    / 6
                    * mul_quatern(
                        lambdc[j * 4 : j * 4 + 4],
                        mul_quatern(eij, lambd[i * 4 : i * 4 + 4]),
                    )
                )
                e_new += (
                    1
                    / 3
                    * mul_quatern(
                        lambdc[j * 4 : j * 4 + 4],
                        mul_quatern(eij, lambd[j * 4 : j * 4 + 4]),
                    )
                )
                # print(norm(e_new-eij))

                div[4 * i + 1 : 4 * i + 4] += e_new[1:] * (cot2 + cot1)
                i_const = i
        one_ring_i0 += ring_nv + 1

    constraint_idx[0] = i_const
    constraint_idx[1] = j
    constraint_pos[0, 1:] = vertices[i_const, :]
    constraint_pos[1, 1:] = vertices[i_const, :] + e_new[1:]

    return div, constraint_idx, constraint_pos


@cc.export(
    "quaternionic_laplacian_matrix",
    "Tuple((i8[::1],i8[::1],f8[::1]))(f8[:,::1], i8[::1], i8, i8)",
)
def quaternionic_laplacian_matrix(vertices, one_ring, max_one_ring_length, n_entries):
    """
    Returns the indices and values needed to construct
    the Laplace-Beltrami operator on the mesh.
    """
    nv = vertices.shape[0]

    ring_vert = np.zeros((max_one_ring_length, 3), dtype=np.float64)

    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries, dtype=np.float64)

    cot = 0.0
    diag = 0.0
    sparse_idx = 0
    one_ring_i0 = 0
    for i in range(nv):
        ring_nv = one_ring[one_ring_i0]
        if ring_nv > 2:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[one_ring_i0 + r_i + 1]]

            # edges connecting the point to its neighbours
            edges_vect = ring_vert - vertices[i, :]

            # iterating over each of the edges adjacent to the vertex i
            diag = 0.0
            for k in range(ring_nv):
                # computing the cotengent weights
                e1 = -edges_vect[(k - 1) % ring_nv]
                o1 = edges_vect[k] + e1
                cos1 = np.dot(e1, o1)
                sin1 = norm(np.cross(o1, e1))
                cot = cos1 / sin1

                e2 = -edges_vect[(k + 1) % ring_nv]
                o2 = edges_vect[k] + e2

                cos2 = np.dot(e2, o2)
                sin2 = norm(np.cross(e2, o2))

                cot += cos2 / sin2

                j = one_ring[one_ring_i0 + k + 1]
                diag -= cot
                for l in range(4):
                    idx_i[sparse_idx] = i * 4 + l
                    idx_j[sparse_idx] = j * 4 + l
                    data[sparse_idx] = cot
                    sparse_idx += 1

            for l in range(4):
                idx_i[sparse_idx] = i * 4 + l
                idx_j[sparse_idx] = i * 4 + l
                data[sparse_idx] = diag
                sparse_idx += 1
        one_ring_i0 += ring_nv + 1

    return idx_i, idx_j, data


@cc.export(
    "mean_curvature",
    "f8[:,::1](f8[:,::1], i8[::1], i8)",
)
def mean_curvature(vertices, one_ring, max_one_ring_length):
    """
    Computes the mean curvature for each vertices of the mesh.

    """

    nv = vertices.shape[0]

    # vertices in the current ring
    ring_vert = np.zeros((max_one_ring_length, 3), dtype=np.float64)

    kn = np.zeros((nv, 3), dtype=np.float64)  # curvature
    one_ring_i0 = 0
    for i in range(nv):
        ring_nv = one_ring[one_ring_i0]
        if ring_nv > 1:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[one_ring_i0 + r_i + 1]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert - vertices[i, :]

            # area of the ring
            area = 0
            for j in range(ring_nv - 1):
                area += norm(np.cross(edges_vect[j], edges_vect[(j + 1)]))

            for j in range(ring_nv):
                e1 = edges_vect[(j - 1) % ring_nv]
                o1 = ring_vert[j] - ring_vert[(j - 1) % ring_nv]
                cos1 = np.dot(e1, o1)
                sin1 = np.linalg.norm(np.cross(o1, e1))
                cot1 = cos1 / sin1

                e2 = edges_vect[(j + 1) % ring_nv]
                o2 = ring_vert[j] - ring_vert[(j + 1) % ring_nv]
                cos2 = np.dot(e2, o2)
                sin2 = np.linalg.norm(np.cross(e2, o2))
                cot2 = cos2 / sin2

                kn[i] -= edges_vect[j] * (cot2 + cot1)
            kn[i] /= 2 * area
        one_ring_i0 += ring_nv + 1
    return kn


@cc.export(
    "symetric_delete",
    "Tuple((i8[::1],i8[::1],f8[::1]))(i8[::1],i8[::1], i8[::1],f8[::1],i8)",
)
def symetric_delete(i_del, idx_i, idx_j, data, n):
    """
    Delete the requested rows and columns sharing the
    same index in a sparse matrix representation.
    """
    idx_table = np.empty(n, np.int_)
    idx_sp = 0
    for k in range(n):
        if k not in i_del:
            idx_table[k] = idx_sp
            idx_sp += 1

    idx_sp = 0
    for k in range(data.shape[0]):
        if (idx_i[k] not in i_del) and (idx_j[k] not in i_del):
            idx_i[idx_sp] = idx_table[idx_i[k]]
            idx_j[idx_sp] = idx_table[idx_j[k]]
            data[idx_sp] = data[k]
            idx_sp += 1

    idx_i = idx_i[:idx_sp]
    idx_j = idx_j[:idx_sp]
    data = data[:idx_sp]
    return idx_i, idx_j, data


if __name__ == "__main__":
    cc.compile()
