#!/usr/bin/env python3
import numpy as np
import trimesh
import networkx as nx

class quaternion():
    def __init__(self, q=np.zeros(4)):
        self.q = q
        a, b, c, d = q[:]
        self.mat = np.array([[a, -b, -c, -d],
                             [b,  a, -d,  c],
                             [c,  d,  a, -b],
                             [d, -c,  b,  a]])
    def conjugate(self):
        conj = self.q.copy()
        conj[1:] = -conj[1:]
        return quaternion(conj)

    def mul_mat(self, v):
        return quaternion(self.mat @ v.q)

    def __add__(self, v):
        return self.q + v.q

    def __mul__(self, v):
        w0, x0, y0, z0 = self.q
        w1, x1, y1, z1 = v.q
        return quaternion(np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64))



def trans(axis, theta, v, a=1):
    axis = np.array(axis)
    axis /= np.linalg.norm(axis)
    q = np.zeros(4)
    q[1:] = a*np.sin(theta/2)*axis
    q[0] = a*np.cos(theta/2)
    mq = q.copy()
    mq[1:] = -mq[1:]
    v[:] = quaternion(mq)*(quaternion(v)*quaternion(q))


def mean_curvature(trimesh):

    vertices = trimesh.vertices
    nv = vertices.shape[0]
    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(len(vertices))]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]


    kN = np.zeros_like(vertices)
    for i in range(nv):
        if len(one_ordered[i]) >0:

            ring_vert = vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-vertices[i,:]

            # area of the ring
            A = np.sum(np.linalg.norm(np.cross(edges_vect[:-1],edges_vect[1:]), axis=1))

            nv = ring_vert.shape[0]
            for j in range(nv):
                e1 = edges_vect[(j-1)%nv].copy()
                # e1 /= np.linalg.norm(e1)
                o1 = ring_vert[j]-ring_vert[(j-1)%nv]
                # o1 /= np.linalg.norm(o1)
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                if sin1 != 0.0 :
                    cot1 = cos1/sin1
                else :
                    cot1 = 0

                e2 = edges_vect[(j+1)%nv].copy()
                # e2 /= np.linalg.norm(e2)
                o2 = ring_vert[j] - ring_vert[(j+1)%nv]
                # o2 /= np.linalg.norm(o2)
                cos2 = np.dot(e2,o2)
                sin2 = np.linalg.norm(np.cross(e2,o2))

                if sin2 != 0.0 :
                    cot2 = cos2/sin2
                else :
                    cot2 = 0

                kN[i] -= edges_vect[j]*(cot2+cot1)
            kN[i] /= 2*A
    return kN


if __name__=="__main__":

    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from mpl_toolkits.mplot3d import Axes3D, art3d

    def plot_normals(ax,vertices, normals,length=10, color="r"):
        for i in range(normals.shape[0]):
            normalsegm = np.stack((vertices[i],vertices[i,:]+length*normals[i]))
            ax.plot(normalsegm[0,0],normalsegm[0,1],normalsegm[0,2],color+'o')
            ax.plot(normalsegm[:,0],normalsegm[:,1],normalsegm[:,2],color)

    def vertex2face(trimesh, vval):
        assert vval.shape[0] == trimesh.vertices.shape[0]
        return np.mean(vval[trimesh.faces], axis=1)

    trimesh = trimesh.load('knobitle.ply')
    vertices = trimesh.vertices
    kN = mean_curvature(trimesh)
    normals = trimesh.vertex_normals
    k = np.sum(normals*kN,axis=1)
    print(f"{ k.min() =} , { k.max() = }")

    face_k_norm = vertex2face(trimesh, k)
    face_k_norm = (face_k_norm - face_k_norm.min())/(np.ptp(face_k_norm))
    start = 0.15
    face_k_norm = start+(1-start-0.001)*face_k_norm
    cm = plt.cm.inferno(face_k_norm)


    triangles = vertices[trimesh.faces]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")


    pc = art3d.Poly3DCollection(triangles, facecolors=cm, alpha=1, shade=True, edgecolors=(1,1,1,0.2))
    ax.add_collection(pc)

    plot_normals(ax, vertices, kN, 1000)

    xlim = vertices[:,0].min(), vertices[:,0].max()
    ylim = vertices[:,1].min(), vertices[:,1].max()
    zlim = vertices[:,2].min(), vertices[:,2].max()

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])

    plt.show()
