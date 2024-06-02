#!/usr/bin/env python3
from logging import raiseExceptions
import numba
from numba.pycc import CC
import numpy as np
from numpy.linalg import norm
from numba import types, njit, jit

cc = CC("operators")
cc.verbose = True


@njit
@cc.export("mul_quatern", "f8[::1](f8[::1], f8[::1])")
def mul_quatern(u, v):
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


@cc.export(
    "set_rings_order",
    "(i8[::1], f8[:,::1],f8[:,::1])",
)
def set_rings_order(one_ring, normals, vertices):
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
        # oring += [one_ring[i0 : i0 + one_ring[i0] + 1].tolist()]
        i0 += one_ring[i0] + 1
        vi += 1

    assert vi == normals.shape[0]


@cc.export(
    "dirac_op",
    "Tuple((i8[::1],i8[::1],f8[::1]))(f8[:,::1], i8[::1], f8[::1])",
)
def dirac_op(vertices, one_ring, rho):
    nv = vertices.shape[0]
    assert rho.shape[0] == nv

    ei = np.zeros(4, dtype=np.float64)
    ej = np.zeros(4, dtype=np.float64)
    X_ii = np.zeros(4, dtype=np.float64)
    block_ij = np.zeros((4, 4), dtype=np.float64)

    max_n_ring = 0
    n_entries = nv
    r_i0 = 0
    len_one_r = 0
    while r_i0 < one_ring.shape[0] and len_one_r < nv:
        n_entries += one_ring[r_i0]
        if one_ring[r_i0] > max_n_ring:
            max_n_ring = one_ring[r_i0]
        r_i0 += one_ring[r_i0] + 1
        len_one_r += 1

    assert len_one_r == nv and r_i0 == one_ring.shape[0]

    ring_vert = np.zeros((max_n_ring, 3), dtype=np.float64)
    n_entries *= 16

    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries, dtype=np.float64)

    sp_k = 0
    r_i0 = 0
    for i in range(nv):
        # TODO on't work for bounded domain
        ring_nv = one_ring[r_i0]
        if ring_nv > 1:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[r_i0 + r_i + 1]]
            v = vertices[i, :]

            X_ii[:] = 0.0
            for k in range(ring_nv):
                j = one_ring[r_i0 + k + 1]
                ei[1:] = ring_vert[k] - ring_vert[(k - 1) % ring_nv]
                ej[1:] = ring_vert[(k - 1) % ring_nv] - v
                # print(ring_vert)
                # print(ei[1:], ej[1:])
                A_x2 = norm(np.cross(ei[1:], ej[1:]))
                X_ij = (
                    -mul_quatern(ei, ej) / (2 * A_x2)
                    + (rho[i] * ej - rho[j] * ei) / 6.0
                )
                X_ij[0] += rho[i] * rho[j] * A_x2 / 18.0

                ei[1:] = ring_vert[(k + 1) % ring_nv] - ring_vert[k]
                ej[1:] = v - ring_vert[(k + 1) % ring_nv]
                A_x2 = norm(np.cross(ei[1:], ej[1:]))
                X_ij += (
                    -mul_quatern(ei, ej) / (2 * A_x2)
                    + (rho[i] * ej - rho[j] * ei) / 6.0
                )
                X_ij[0] += rho[i] * rho[j] * A_x2 / 18.0

                block_ij[:, :] = np.array(
                    [
                        [X_ij[0], -X_ij[1], -X_ij[2], -X_ij[3]],
                        [X_ij[1], X_ij[0], -X_ij[3], X_ij[2]],
                        [X_ij[2], X_ij[3], X_ij[0], -X_ij[1]],
                        [X_ij[3], -X_ij[2], X_ij[1], X_ij[0]],
                    ]
                )

                for l in range(4):
                    for m in range(4):
                        idx_i[sp_k] = i * 4 + l
                        idx_j[sp_k] = j * 4 + m
                        data[sp_k] = block_ij[l, m]
                        sp_k += 1

                X_ii -= mul_quatern(ei, ei) / (2 * A_x2)
                X_ii[0] += rho[i] * rho[i] * A_x2 / 18.0

            block_ij[:, :] = np.array(
                [
                    [X_ii[0], -X_ii[1], -X_ii[2], -X_ii[3]],
                    [X_ii[1], X_ii[0], -X_ii[3], X_ii[2]],
                    [X_ii[2], X_ii[3], X_ii[0], -X_ii[1]],
                    [X_ii[3], -X_ii[2], X_ii[1], X_ii[0]],
                ]
            )

            for l in range(4):
                for m in range(4):
                    idx_i[sp_k] = i * 4 + l
                    idx_j[sp_k] = i * 4 + m
                    data[sp_k] = block_ij[l, m]
                    sp_k += 1
            r_i0 += ring_nv + 1

    return idx_i, idx_j, data


@cc.export(
    "edges_div",
    "Tuple((f8[::1],i8[::1],f8[:,::1]))(f8[:,::1], f8[::1], i8[::1])",
)
def edges_div(vertices, lambd, one_ring):
    nv = vertices.shape[0]

    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()

    constraint_idx = np.zeros(2, dtype=np.int_)
    constraint_pos = np.zeros((2, 4), dtype=np.float64)

    assert lambd.shape[0] % 4 == 0
    for i in range(lambd.shape[0] // 4):
        lambdc[4 * i + 1 : 4 * i + 4] *= -1

    max_n_ring = 0
    n_entries = nv
    r_i0 = 0
    len_one_r = 0
    while r_i0 < one_ring.shape[0] and len_one_r < nv:
        n_entries += one_ring[r_i0]
        if one_ring[r_i0] > max_n_ring:
            max_n_ring = one_ring[r_i0]
        r_i0 += one_ring[r_i0] + 1
        len_one_r += 1

    ring_vert = np.zeros((max_n_ring, 3), dtype=np.float64)

    assert len_one_r == nv and r_i0 == one_ring.shape[0]
    i_const = 0
    div = np.zeros((nv * 4))
    r_i0 = 0
    for i in range(nv):
        ring_nv = one_ring[r_i0]
        if ring_nv > 1:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[r_i0 + r_i + 1]]

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

                j = one_ring[r_i0 + k + 1]
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
        r_i0 += ring_nv + 1

    constraint_idx[0] = i_const
    constraint_idx[1] = j
    constraint_pos[0, 1:] = vertices[i_const, :]
    constraint_pos[1, 1:] = vertices[i_const, :] + e_new[1:]

    return div, constraint_idx, constraint_pos


@cc.export(
    "quaternionic_laplacian_matrix",
    "Tuple((i8[::1],i8[::1],f8[::1]))(f8[:,::1], i8[::1])",
)
def quaternionic_laplacian_matrix(vertices, one_ring):
    nv = vertices.shape[0]

    max_n_ring = 0
    n_entries = nv
    r_i0 = 0
    len_one_r = 0
    while r_i0 < one_ring.shape[0] and len_one_r < nv:
        n_entries += one_ring[r_i0]
        if one_ring[r_i0] > max_n_ring:
            max_n_ring = one_ring[r_i0]
        r_i0 += one_ring[r_i0] + 1
        len_one_r += 1

    assert len_one_r == nv and r_i0 == one_ring.shape[0]

    ring_vert = np.zeros((max_n_ring, 3), dtype=np.float64)
    n_entries *= 16

    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries, dtype=np.float64)

    cot = 0.0
    diag = 0.0
    sp_k = 0
    r_i0 = 0
    for i in range(nv):
        ring_nv = one_ring[r_i0]
        if ring_nv > 2:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[r_i0 + r_i + 1]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert - vertices[i, :]

            # iterating over each of the edges adjacent to the vertex i
            diag = 0.0
            for k in range(ring_nv):
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

                j = one_ring[r_i0 + k + 1]
                diag -= cot
                for l in range(4):
                    idx_i[sp_k] = i * 4 + l
                    idx_j[sp_k] = j * 4 + l
                    data[sp_k] = cot
                    sp_k += 1

            for l in range(4):
                idx_i[sp_k] = i * 4 + l
                idx_j[sp_k] = i * 4 + l
                data[sp_k] = diag
                sp_k += 1
        r_i0 += ring_nv + 1

    return idx_i, idx_j, data


@cc.export(
    "mean_curvature",
    "f8[:,::1](f8[:,::1], i8[::1])",
)
def mean_curvature(vertices, one_ring):

    nv = vertices.shape[0]

    max_n_ring = 0
    r_i0 = 0
    len_one_r = 0
    while r_i0 < one_ring.shape[0] and len_one_r < nv:
        if one_ring[r_i0] > max_n_ring:
            max_n_ring = one_ring[r_i0]
        r_i0 += one_ring[r_i0] + 1
        len_one_r += 1

    assert len_one_r == nv and r_i0 == one_ring.shape[0]
    ring_vert = np.zeros((max_n_ring, 3), dtype=np.float64)

    kN = np.zeros((nv, 3), dtype=np.float64)
    r_i0 = 0
    for i in range(nv):
        ring_nv = one_ring[r_i0]
        if ring_nv > 1:
            for r_i in range(ring_nv):
                ring_vert[r_i, :] = vertices[one_ring[r_i0 + r_i + 1]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert - vertices[i, :]

            # area of the ring
            A = 0
            for j in range(ring_nv - 1):
                A += norm(np.cross(edges_vect[j], edges_vect[(j + 1) % ring_nv]))

            for j in range(ring_nv):
                e1 = edges_vect[(j - 1) % ring_nv].copy()
                o1 = ring_vert[j] - ring_vert[(j - 1) % ring_nv]
                cos1 = np.dot(e1, o1)
                sin1 = np.linalg.norm(np.cross(o1, e1))
                cot1 = cos1 / sin1

                e2 = edges_vect[(j + 1) % ring_nv].copy()
                o2 = ring_vert[j] - ring_vert[(j + 1) % ring_nv]
                cos2 = np.dot(e2, o2)
                sin2 = np.linalg.norm(np.cross(e2, o2))
                cot2 = cos2 / sin2

                kN[i] -= edges_vect[j] * (cot2 + cot1)
            kN[i] /= 2 * A
        r_i0 += ring_nv + 1
    return kN


if __name__ == "__main__":
    cc.compile()
