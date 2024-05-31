#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import art3d
from matplotlib.colors import LightSource
from transform import transform
from trimesh_curvature import mean_curvature, scalar_curvature
import trimesh
import numpy as np

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

fig, axs = plt.subplots(1,2,figsize=(8,4),subplot_kw=dict(projection='3d'))
ax1, ax2 = axs


triangles = vertices[trimesh.faces]
light = LightSource(90,100)
pc = art3d.Poly3DCollection(triangles, facecolors=cm,
                            shade=True, edgecolors=(1,1,1,0.2),
                            alpha=1, lightsource=light)
ax1.add_collection(pc)
# plot_normals(ax, face_center, trimesh.face_normals)
xlim = [(vertices[:,0].min(), vertices[:,0].max())]
ylim = [(vertices[:,1].min(), vertices[:,1].max())]
zlim = [(vertices[:,2].min(), vertices[:,2].max())]

transform(trimesh, rho)

kN = mean_curvature(trimesh)
k = scalar_curvature(trimesh, kN)

k_fc = vertex2face(trimesh, k)
k_norm = (k_fc-k_fc.min())/np.ptp(k_fc)
cm = plt.cm.plasma(k_norm)

triangles = vertices[trimesh.faces]
pc = art3d.Poly3DCollection(triangles, facecolors=cm,
                            shade=True, edgecolors=(1,1,1,0.2),
                            alpha=1, lightsource=light)
ax2.add_collection(pc)


xlim += [(vertices[:,0].min(), vertices[:,0].max())]
ylim += [(vertices[:,1].min(), vertices[:,1].max())]
zlim += [(vertices[:,2].min(), vertices[:,2].max())]

for i, ax in enumerate(axs):
    ax.set_xlim(*xlim[i])
    ax.set_ylim(*ylim[i])
    ax.set_zlim(*zlim[i])
    # ax.set_xlabel("x")
    # ax.set_ylabel("y")
    # ax.set_zlabel("z")
    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()

# trimesh.visual.face_colors = cm
# trimesh.export("sphananaHQ.ply")
fig.tight_layout(pad=0)
# fig.savefig("ballfig.png")
plt.show()
