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
    "dirac",
    "Tuple((i8[::1],i8[::1],f8[::1]))(f8[:,::1], f8[:,::1], List(List(i8)), f8[::1])",
)
def create_dirac_op_sparse(vertices, normals, one_ring_ordered, rho):

    nv = vertices.shape[0]

    assert rho.shape[0] == nv and nv == normals.shape[0]

    ei = np.zeros(4, dtype=np.float64)
    ej = np.zeros(4, dtype=np.float64)
    X_ii = np.zeros(4, dtype=np.float64)
    block_ij = np.zeros((4, 4), dtype=np.float64)

    n_entries = nv
    max_n_ring = 0
    for i in one_ring_ordered:
        n_entries += len(i)
        if len(i) > max_n_ring:
            max_n_ring = len(i)

    n_entries *= 16

    idx_i = np.zeros(n_entries, dtype=np.int_)
    idx_j = np.zeros(n_entries, dtype=np.int_)
    data = np.zeros(n_entries, dtype=np.float64)
    ring_vert = np.zeros((max_n_ring, 3), dtype=np.float64)

    sp_k = 0
    for i in range(nv):
        # TODO won't work for bounded domain
        ring_nv = len(one_ring_ordered[i])
        if ring_nv > 0:
            for ri in range(ring_nv):
                ring_vert[ri, :] = vertices[one_ring_ordered[i][ri]]
            vi = vertices[i, :]
            vertex_normal = normals[i, :]
            ei[1:] = ring_vert[1] - ring_vert[0]
            ej[1:] = ring_vert[0] - vi
            sign = np.dot(vertex_normal, np.cross(ej[1:], ei[1:]))
            sign = int(np.sign(sign))
            ring_vert = ring_vert[::sign, :]
            one_ring_ordered[i] = one_ring_ordered[i][::sign]

            X_ii[:] = 0.0
            for k in range(ring_nv):
                j = one_ring_ordered[i][k]
                ei[1:] = ring_vert[k] - ring_vert[(k - 1) % ring_nv]
                ej[1:] = ring_vert[(k - 1) % ring_nv] - vi

                A_x2 = norm(np.cross(ei[1:], ej[1:]))
                X_ij = (
                    -mul_quatern(ei, ej) / (2 * A_x2)
                    + (rho[i] * ej - rho[j] * ei) / 6.0
                )
                X_ij[0] += rho[i] * rho[j] * A_x2 / 18.0

                ei[1:] = ring_vert[(k + 1) % ring_nv] - ring_vert[k]
                ej[1:] = vi - ring_vert[(k + 1) % ring_nv]
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

    return idx_i, idx_j, data


if __name__ == "__main__":
    cc.compile()
