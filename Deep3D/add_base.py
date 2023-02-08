import numpy as np
import os
from uv_mesh.mesh import load_obj_mesh, save_obj_mesh_with_rgb
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.io import loadmat, savemat
from tqdm import tqdm


X = []
Y = []
print("Reading obj")
old = loadmat("BFM/BFM_model_front.mat")

front_idx = loadmat("BFM/select_vertex_id.mat")["select_id"] - 1
front_idx = front_idx[:, 0]


for fname in tqdm(os.listdir("toonme_topo")):
    vertices, _, texture = load_obj_mesh("toonme_topo/" + fname)
    X.append(vertices.reshape(1, -1))
    Y.append(texture.reshape(1, -1))

X = np.concatenate(X, axis=0)
Y = np.concatenate(Y, axis=0)
print(X.shape)
print(Y.shape)

old["mean_cartoon_shape"] = X.mean(axis=0, keepdims=True)
old["mean_cartoon_texture"] = Y.mean(axis=0, keepdims=True)
old["meantex"] /= 255
print("Extracting PCA")



pca = PCA(n_components=80)
pca.fit(X)
new_component = pca.components_.transpose()
base = np.concatenate((old["idBase"], new_component), axis= 1)
old["idBase"] = base




del pca

pca = PCA(n_components=80)
pca.fit(Y)
new_component = pca.components_.transpose()
base = np.concatenate((old["texBase"] / 255, new_component), axis= 1)
old["texBase"] = base

savemat("./BFM/fuse_model_front_improve.mat", old)
#savemat("./BFM/fuse_model_front.mat", old)