#!/usr/bin/env python3
import numpy as np
import trimesh
import networkx as nx
from scipy.sparse.linalg import spsolve, norm, eigsh, inv
from scipy.sparse import csc_array, csr_array
from scipy.linalg import issymmetric, solve, expm_cond, eigh
from trimesh_curvature import scalar_curvature, mean_curvature
from numba import jit


def mul_quatern(u, v):
    w0, x0, y0, z0 = u
    w1, x1, y1, z1 = v
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def create_dirac_op_vv(trimesh, rho):
    vertices = trimesh.vertices
    nv = vertices.shape[0]
    faces_edges_idx = trimesh.faces_unique_edges
    edges = trimesh.edges_unique[faces_edges_idx]
    X_block = np.zeros((nv*4,nv*4))
    e_i = np.zeros(4)
    e_j = np.zeros(4)
    assert rho.shape[0] == nv
    block_ij = np.zeros((4,4))

    # for each triangle
    for f_i in range(faces_edges_idx.shape[0]):
        # Get the ordered vertices list of each triangle
        graph = nx.from_edgelist(edges[f_i])
        vertex_ordered_idx = np.array(nx.cycle_basis(graph)[0], dtype=np.int16)
        f_v = trimesh.vertices[vertex_ordered_idx]

        # vertex_normal = trimesh.face_normals[f_i]
        # order_normal = np.cross(f_v[1]-f_v[0], f_v[2]-f_v[0])
        # sign = np.dot(vertex_normal, order_normal)
        # sign = int(np.sign(sign))
        # vertex_ordered_idx = vertex_ordered_idx[::sign]
        # f_v = f_v[::sign,:]
        #
        area_x2 = np.linalg.norm(np.cross(f_v[1]-f_v[0],f_v[2]-f_v[0]))
        for t_i in range(3):
            e_i[1:] = f_v[(t_i+2)%3]-f_v[(t_i+1)%3]
            i = vertex_ordered_idx[t_i]
            # for t_j in [(t_i+1)%3,(t_i+2)%3]:
            for t_j in range(3):
                j = vertex_ordered_idx[t_j]
                e_j[1:] = f_v[(t_j+2)%3]-f_v[(t_j+1)%3]
                X_ij = - mul_quatern(e_i, e_j)/(2*area_x2) + (rho[i]*e_j-rho[j]*e_i)/6.0
                X_ij[0] +=  rho[i]*rho[j]*area_x2/18.0

                block_ij[:,:] = [[X_ij[0], -X_ij[1], -X_ij[2], -X_ij[3]],
                                  [X_ij[1],  X_ij[0], -X_ij[3],  X_ij[2]],
                                  [X_ij[2],  X_ij[3],  X_ij[0], -X_ij[1]],
                                  [X_ij[3], -X_ij[2],  X_ij[1],  X_ij[0]]]

                X_block[i*4:i*4+4,j*4:j*4+4] += block_ij
    return X_block




def new_edges_divergence(trimesh, lambd):
    nv = trimesh.vertices.shape[0]

    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]

    eij = np.zeros(4, dtype=np.float64)
    lambdc = lambd.copy()

    constraint = [[0,np.zeros(4)],[0,np.zeros(4)]]
    assert lambd.shape[0]%4 == 0
    for i in range(lambd.shape[0]//4):
        lambdc[4*i+1:4*i+4] *= -1


    i_const = 0
    div = np.zeros((nv*4))
    for i in range(nv):
        if len(one_ordered[i]) >0:
            ring_vert = trimesh.vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-trimesh.vertices[i,:]

            # vertex_normal = trimesh.vertex_normals[i]
            # sign = np.dot(vertex_normal,np.cross(edges_vect[0],edges_vect[1]))
            # sign = int(np.sign(sign))
            # edges_vect = edges_vect[::sign,:]
            # one_ordered[i][0] = one_ordered[i][0][::sign]

            ring_nv = ring_vert.shape[0]
            for k in range(ring_nv):
                e1 = -edges_vect[(k-1)%ring_nv]
                o1 =  edges_vect[k]+e1
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                cot1 = cos1/sin1

                e2 = -edges_vect[(k+1)%ring_nv]
                o2 =  edges_vect[k]+e2
                cos2 = np.dot(e2,o2)
                sin2 = np.linalg.norm(np.cross(e2,o2))

                cot2 = cos2/sin2

                j = one_ordered[i][0][k]
                eij[1:] = edges_vect[k]
                e_new  = 1/3 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[i*4:i*4+4], mul_quatern(eij, lambd[j*4:j*4+4]))
                e_new += 1/6 * mul_quatern(lambdc[j*4:j*4+4], mul_quatern(eij, lambd[i*4:i*4+4]))
                e_new += 1/3 * mul_quatern(lambdc[j*4:j*4+4], mul_quatern(eij, lambd[j*4:j*4+4]))
                # print(np.linalg.norm(e_new-eij))

                div[4*i+1:4*i+4] += e_new[1:]*(cot2+cot1)
                i_const = i

    constraint[0][0] = i_const
    constraint[0][1][1:] = trimesh.vertices[i_const,:]
    constraint[1][0] = j
    constraint[1][1][1:] =  trimesh.vertices[i_const,:]+e_new[1:]

    return div, constraint


def quaternionic_laplacian_matrix(trimesh):
    nv = trimesh.vertices.shape[0]
    g = nx.from_edgelist(trimesh.edges_unique)
    one_ring = [list(g[i].keys()) for i in range(nv)]
    one_ordered = [nx.cycle_basis(g.subgraph(i)) for i in one_ring]
    L = np.zeros((nv*4,nv*4))

    idx_i = np.zeros(nv*4*10, dtype=np.int_)
    idx_j = np.zeros(nv*4*10, dtype=np.int_)
    data = np.zeros(nv*4*10,dtype=np.float64)

    cot = 0.0
    diag = 0.0
    sp_k = 0
    for i in range(nv):
        if len(one_ordered[i][0]) >0:
            ring_vert = trimesh.vertices[one_ordered[i][0]]

            # edges connecting the point to itś neighbours
            edges_vect = ring_vert-trimesh.vertices[i,:]
            # vertex_normal = trimesh.vertex_normals[i]
            # sign = np.dot(vertex_normal,np.cross(edges_vect[0],edges_vect[1]))
            # sign = int(np.sign(sign))
            # edges_vect = edges_vect[::sign,:]
            # one_ordered[i][0] = one_ordered[i][0][::sign]
            # ring_vert = ring_vert[::sign,:]

            ring_nv = ring_vert.shape[0]
            # iterating over each of the edges adjacent to the vertex i
            diag = 0.0
            for k in range(ring_nv):
                e1 = -edges_vect[(k-1)%ring_nv]
                o1 =  edges_vect[k]+e1
                cos1 = np.dot(e1,o1)
                sin1 = np.linalg.norm(np.cross(o1,e1))
                cot = cos1/sin1

                e2 = -edges_vect[(k+1)%ring_nv]
                o2 =  edges_vect[k]+e2

                cos2 = np.dot(e2,o2)
                sin2 = np.linalg.norm(np.cross(e2,o2))

                cot += cos2/sin2

                j = one_ordered[i][0][k]
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

    return idx_i[:sp_k], idx_j[:sp_k], data[:sp_k]

def eigensolve(M, v):
    nv = v.shape[0]
    v[:] = 0
    v[::4] = 1/nv
    M = M
    for i in range(4):
        v[:] = spsolve(M, v)
        v.shape = (nv//4,4)
        v /= np.sqrt(np.sum(np.sum(v**2,axis=1)))
        v.shape = (nv, )

    print("eigen vector residual inf norm :",np.linalg.norm(M@v - v , ord=np.inf))
    v.shape = (nv//4,4)
    v /= np.mean(np.sqrt(np.sum(v**2,axis=1)))
    v.shape = (nv, )


def symetric_delete(i_del, idx_i, idx_j, data, n):
    idx_table = np.empty(n, np.int_)
    idx_sp = 0
    for k in range(n):
        if k not in i_del:
            idx_table[k] = idx_sp
            idx_sp +=1

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

# def symetric_delete2(i_del, L):
#     L_wp = np.delete(L,i_del, axis=0)
#     return np.delete(L_wp, i_del, axis=1)


def transform(trimesh, rho):

    vertices = trimesh.vertices
    nv = vertices.shape[0]

    X = create_dirac_op_vv(trimesh, rho)
    print(f"{issymmetric(X) = }")

    X = csc_array(X)

    lambd = np.zeros(4*nv)
    eigensolve(X, lambd)

    div_e, constraint = new_edges_divergence(trimesh, lambd)

    idx_i, idx_j, data = quaternionic_laplacian_matrix(trimesh)

    # csc for row slicing
    L = csc_array((data, (idx_i, idx_j)),shape=(nv*4,nv*4))

    # Applying constraint to the system
    # TODO make it work for bounded shapes
    for c in constraint:
        i = c[0]
        div_e[:] -= L[:,i*4:i*4+4] @ c[1].T

    i_sorted = np.argsort([constraint[0][0],constraint[1][0]])
    i1 = constraint[i_sorted[0]][0]
    i2 = constraint[i_sorted[1]][0]

    i_del = [ic*4+i for ic in [i1, i2] for i in range(4)]

    idx_i, idx_j, data = symetric_delete(i_del, idx_i, idx_j, data, nv*4)

    # csr
    L = csc_array((data, (idx_i, idx_j)),shape=((nv-2)*4,(nv-2)*4))
    # Ls = (L + L.T)/2
    # print(f"{issymmetric(L.todense()) = }")

    div_wp = np.delete(div_e, i_del, axis=0)


    # norm_L = norm(L)
    # inv_L = inv(L)
    # norm_invA = norm(inv_L)
    # cond = norm_L*norm_invA
    # print(f"L condition number = {cond}")

    new_vertices_wp = spsolve(L, div_wp)
    # new_vertices = solve(L, div_e, assume_a='sym')
    residual = np.linalg.norm(L@new_vertices_wp - div_wp)
    if residual > 1e-10 :
        print(f"WARNING : {residual =}")

    new_vertices_wp.shape = (nv-2,4)
    new_vertices = np.insert(new_vertices_wp, i1, constraint[i_sorted[0]][1], axis=0)
    new_vertices = np.insert(new_vertices, i2, constraint[i_sorted[1]][1], axis=0)
    vertices[:,:] = new_vertices[:,1:]

if __name__=="__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import art3d
    from matplotlib.colors import LightSource

    def plot_normals(ax,vertices, normals,length=10, color="r"):
        for i in range(normals.shape[0]):
            normalsegm = np.stack((vertices[i],vertices[i,:]+length*normals[i]))
            ax.plot(normalsegm[0,0],normalsegm[0,1],normalsegm[0,2],color+'o')
            ax.plot(normalsegm[:,0],normalsegm[:,1],normalsegm[:,2],color)

    def vertex2face(trimesh, vval):
        assert vval.shape[0] == trimesh.vertices.shape[0]
        return np.mean(vval[trimesh.faces], axis=1)

    trimesh = trimesh.load('meshes/sphere.ply')
    print("number of vertices", trimesh.vertices.shape[0])

    kN = mean_curvature(trimesh)
    k = scalar_curvature(trimesh, kN)
    mk = np.mean(k)
    print("mean curvature =", mk)

    vertices = trimesh.vertices
    nv = vertices.shape[0]


    R = 50
    pts = []
    pts += [[np.pi/4, 2*np.pi/6]]
    pts += [[-3*np.pi/4, 0]]
    pts += [[-np.pi/4, -np.pi/6]]
    ampl = np.array([-50, -20, -15])*abs(mk)
    rad = [50.0, 200.0, 100.0]

    rho = np.zeros(nv,dtype=np.float64)
    for i, pt in enumerate(pts):
        pt_cart = np.array([R*np.cos(pt[1])*np.cos(pt[0]),
                    R*np.cos(pt[1])*np.sin(pt[0]),
                    R*np.sin(pt[1])])
        dist = np.linalg.norm(vertices-pt_cart,axis=1)
        rho += ampl[i]*np.exp(-dist**2/rad[i])



    rho_fc = vertex2face(trimesh, rho)
    rho_norm = (rho_fc-rho_fc.min())/np.ptp(rho_fc)
    cm = plt.cm.plasma(rho_norm)

    transform(trimesh, rho)

    fig, ax = plt.subplots(1,1,figsize=(8,4),subplot_kw=dict(projection='3d'))


    triangles = vertices[trimesh.faces]
    light = LightSource(90,100)
    pc = art3d.Poly3DCollection(triangles, facecolors=cm,
                                shade=True, edgecolors=(1,1,1,0.2),
                                alpha=1, lightsource=light)
    ax.add_collection(pc)
    # plot_normals(ax, face_center, trimesh.face_normals)
    xlim = (vertices[:,0].min(), vertices[:,0].max())
    ylim = (vertices[:,1].min(), vertices[:,1].max())
    zlim = (vertices[:,2].min(), vertices[:,2].max())

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])

    fig.tight_layout(pad=0)
    plt.show()