import os, sys
import point_cloud_utils as pcu
from uv_mesh.mesh import load_obj_mesh, save_obj_mesh
from tqdm import tqdm

### 3dmm low resolution
v, f_simp = load_obj_mesh("template/template_bfm.obj")
#v, f_simp =  pcu.load_mesh_vf("template/3dmm_temp_full.ply")

### toonme target
vs, fs = load_obj_mesh("template/template_toonme.obj")
#vs, fs = pcu.load_mesh_vf("template/3dmm_toonme_full.ply")

in_path = "./raw_data"
out = "./register_toonme"
fnames = [fname for fname in os.listdir(in_path) if ("exp" not in fname)]


d, fi, bc = pcu.closest_points_on_mesh(v, vs, fs)


os.makedirs(out, exist_ok=True)
for fname in tqdm(fnames):
    vt, ft = load_obj_mesh(os.path.join(in_path, fname))
    

    closest_points = pcu.interpolate_barycentric_coords(fs, fi, bc, vt)
    save_obj_mesh(os.path.join(out, fname), closest_points, f_simp)

