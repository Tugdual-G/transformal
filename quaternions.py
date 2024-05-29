#!/usr/bin/env python3
import numpy as np

class quaternion():
    def __init__(self, q=np.zeros(4)):
        self.q = q

    def get_matrix_format(self):
        a, b, c, d = self.q
        self.mat = np.array([[a, -b, -c, -d],
                             [b,  a, -d,  c],
                             [c,  d,  a, -b],
                             [d, -c,  b,  a]])
        return self.mat

    def norm2(self):
        return np.sum(self.q**2)

    def norm(self):
        return np.sqrt(np.sum(self.q**2))

    def c(self):
        conj = self.q.copy()
        conj[1:] = -conj[1:]
        return quaternion(conj)

    def mul_mat(self, other):
        return quaternion(self.mat @ other.q)

    def __add__(self, other):
        return self.q + other.q

    def __matmulr__(self, other):
        return 0

    # def __pow__(self, other):

    def __sub__(self, other):
        return self.q - other.q

    def __mul__(self, other):
        w0, x0, y0, z0 = self.q
        w1, x1, y1, z1 = other.q
        return quaternion(np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64))

    def __truediv__(self, other):
        return self *  (other.c / other.norm2)


def trans(axis, theta, v, a=1):
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    q = np.zeros(4)
    q[1:] = a*np.sin(theta/2)*axis
    q[0] = a*np.cos(theta/2)
    mq = q.copy()
    mq[1:] = -mq[1:]
    v[:] = quaternion(mq)*(quaternion(v)*quaternion(q))
