#!/usr/bin/env python3
import numpy as np
import trimesh
import networkx as nx

def create_dirac_op(trimesh):
    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]
    # D = np.zeros((trimesh.faces.shape[0],trimesh.vertices.shape[0], 3))
    D_block = np.zeros((trimesh.faces.shape[0]*4,trimesh.vertices.shape[0]*4))
    # block = np.zeros((4,4))

    for i in range(faces_edges_idx.shape[0]):
        graph = nx.from_edgelist(edges[i])
        vertex_ordered = np.array(nx.cycle_basis(graph)[0][::-1], dtype=np.int16)
        v = trimesh.vertices[vertex_ordered]
        area_x2 = np.linalg.norm(np.cross(v[1]-v[0],v[2]-v[1]))
        for j in range(3):
            e_j = v[(j+2)%3]-v[(j+1)%3]
            Dij = -(e_j)/area_x2
            # D[i,vertex_ordered[j],:] = Dij
            block = np.array([[0, -Dij[0], -Dij[1], -Dij[2]],
                                [Dij[0],  0, -Dij[2],  Dij[1]],
                                [Dij[1],  Dij[2],  0, -Dij[0]],
                                [Dij[2], -Dij[1],  Dij[0],  0]])

            D_block[i*4:i*4+4,vertex_ordered[j]*4:vertex_ordered[j]*4+4] = block
    return D_block

def create_R(trimesh, rho):
    nf = trimesh.faces.shape[0]
    nv = trimesh.vertices.shape[0]
    P = np.zeros((nf*4,nf*4))
    for i in range(nf):
        for k in range(4):
            P[i*4+k, i*4+k] = rho[i]

    B = np.zeros((nf*4,nv*4))

    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]

    for i in range(nf):
        graph = nx.from_edgelist(edges[i])
        vertex_ordered = np.array(nx.cycle_basis(graph)[0][::-1], dtype=np.int16)
        for j in vertex_ordered:
            for k in range(4):
                B[i*4+k,j*4+k] = 1/3.0

    return P@B

def create_Mf(trimesh):
    nf = trimesh.faces.shape[0]
    Mf = np.zeros((nf*4,nf*4))
    for i in range(nf):
        area = trimesh.area_faces[i]
        for k in range(4):
            Mf[i*4+k,i*4+k] = area
    return Mf

def create_Mv(trimesh):
    nv = trimesh.vertices.shape[0]
    Mv = np.zeros((nv*4,nv*4))
    for i in range(nv):
        faces = trimesh.vertex_faces[i]
        area = trimesh.area_faces[faces[faces != -1]]
        area = np.sum(area, axis=0)/3
        for k in range(4):
            Mv[i*4+k,i*4+k] = area
    return Mv

def mul_quatern(u, v):
    w0, x0, y0, z0 = u
    w1, x1, y1, z1 = v
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def new_edges(trimesh, lambd):
    # TODO check rotation order
    assert lambd.shape[0]%4 == 0
    nv = trimesh.vertices.shape[0]
    edges = trimesh.edges_unique
    edge_vects = np.zeros((nv*4,nv*4), dtype=np.float64)
    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()

    for i in range(lambd.shape[0]//4):
        lambdc[4*i+1:4*i+4] *= -1

    for i in range(edges.shape[0]):
        vi = edges[i,0]
        vj = edges[i,1]
        eij[1:] = trimesh.vertices[vj]-trimesh.vertices[vi]
        e_new  = 1/3 * mul_quatern(lambdc[vi*4:vi*4+4], mul_quatern(eij, lambd[vi*4:vi*4+4]))
        e_new += 1/6 * mul_quatern(lambdc[vi*4:vi*4+4], mul_quatern(eij, lambd[vj*4:vj*4+4]))
        e_new += 1/6 * mul_quatern(lambdc[vj*4:vj*4+4], mul_quatern(eij, lambd[vi*4:vi*4+4]))
        e_new += 1/3 * mul_quatern(lambdc[vj*4:vj*4+4], mul_quatern(eij, lambd[vj*4:vj*4+4]))

        block = np.array([[0.0, -e_new[1], -e_new[2], -e_new[3]],
                          [e_new[1],  0.0, -e_new[3],  e_new[2]],
                          [e_new[2],  e_new[3],  0.0, -e_new[1]],
                          [e_new[3], -e_new[2],  e_new[1],  0.0]])

        edge_vects[vi*4:vi*4+4, vj*4:vj*4+4] = block
        edge_vects[vj*4:vj*4+4, vi*4:vi*4+4] = -block

    return edge_vects




if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    from trimesh_curvature import mean_curvature
    from scipy.linalg import eigh


    trimesh = trimesh.load('knobitle.ply')

    kN = mean_curvature(trimesh)
    k = np.linalg.norm(kN, axis=1)
    face_k = k[trimesh.faces]
    face_k = np.mean(face_k, axis=1)
    rho = -0.1*face_k

    D = create_dirac_op(trimesh)
    R = create_R(trimesh, rho)
    print(f"{D.shape=}")
    print(f"{R.shape=}")
    A = D - R
    Mv = create_Mv(trimesh)
    Mv_inv = Mv.copy()
    for i in range(Mv.shape[0]):
        Mv_inv[i,i] = 1/Mv_inv[i,i]

    Mf = create_Mf(trimesh)
    print(f"{Mf.shape=}")
    print(f"{Mv.shape=}")
    print(f"{A.shape=}")
    A_str = Mv_inv @ A.T @ Mf

    eigvals, eigvecs = eigh(A_str @ A, Mv, eigvals_only=False, subset_by_index=[0,0])
    lambd = eigvecs[:,0]

    edges_new = new_edges(trimesh, lambd)


    quit()


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    triangles = trimesh.vertices[trimesh.faces]
    pc = art3d.Poly3DCollection(triangles, facecolors='g',shade=True, edgecolors=(1,1,1,0.2))
    ax.add_collection(pc)
    ax.set_xlim(-90, 90)
    ax.set_ylim(-90, 90)
    ax.set_zlim(-90, 90)

    plt.show()
