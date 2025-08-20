#%%

import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

comm = MPI.COMM_WORLD
start = 0.1
div = 4
h = start / div

# *** Outer Cube Parameters ***
cube_size = 1.0
height = cube_size
lc = 1.0

# *** Inner Cylinder Parameters ***
cyl_center_x, cyl_center_y = 0.5, 0.5      # Centered in cube
cyl_radius = 0.05                          # Cylinder radius  (adjust as needed)
cyl_base_z = 0.4                           # Lower face at z=0
cyl_height = 0.2                           # Full height

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)  # Suppress output

# --- Outer Cube ---
# p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
# p2 = gmsh.model.occ.addPoint(1, 0, 0, lc)
# p3 = gmsh.model.occ.addPoint(1, 1, 0, lc)
# p4 = gmsh.model.occ.addPoint(0, 1, 0, lc)

p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
p2 = gmsh.model.occ.addPoint(cube_size, 0, 0, lc)
p3 = gmsh.model.occ.addPoint(cube_size, cube_size, 0, lc)
p4 = gmsh.model.occ.addPoint(0, cube_size, 0, lc)

cl = gmsh.model.occ.addCurveLoop([
    gmsh.model.occ.addLine(p1, p2),
    gmsh.model.occ.addLine(p2, p3),
    gmsh.model.occ.addLine(p3, p4),
    gmsh.model.occ.addLine(p4, p1)
])


surface = gmsh.model.occ.addPlaneSurface([cl])
extrude_result = gmsh.model.occ.extrude([(2, surface)], 0, 0, height)
outer_volume = extrude_result[1]  # (3, vol_tag)

# --- Inner Cylinder ---
circle = gmsh.model.occ.addCircle(cyl_center_x, cyl_center_y, cyl_base_z, cyl_radius)
curve_loop = gmsh.model.occ.addCurveLoop([circle])
cyl_base_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
extrude_result_inner = gmsh.model.occ.extrude([(2, cyl_base_surface)], 0, 0, cyl_height)
inner_volume = extrude_result_inner[1]  # (3, vol_tag)

# --- Boolean Fragment (Splitting cube and cylinder as volumes) ---
model_dim_tags = gmsh.model.occ.fragment([(3, outer_volume[1])], [(3, inner_volume[1])])
gmsh.model.occ.synchronize()

# --- Tagging volumes (Physical groups for subdomains) ---
gmsh.model.addPhysicalGroup(3, [model_dim_tags[0][0][1]], tag=1)  # Cylinder
gmsh.model.addPhysicalGroup(3, [model_dim_tags[0][1][1]], tag=2)  # Remaining cube

# --- Boundary tags (top 2D surfaces) ---
boundary = gmsh.model.getBoundary([model_dim_tags[0][1]], oriented=False)
boundary_ids = [b[1] for b in boundary]
gmsh.model.occ.synchronize()
# The first 6 faces are typically the inner (cylinder), next 6 are the cube. Adjust if needed!
gmsh.model.addPhysicalGroup(2, boundary_ids[:1], tag=1)   # Lower face of cylinder
gmsh.model.addPhysicalGroup(2, boundary_ids[1:2], tag=2)   # Side face of cylinder
gmsh.model.addPhysicalGroup(2, boundary_ids[2:3], tag=3)   # Upper face of cylinder
gmsh.model.addPhysicalGroup(2, boundary_ids[3:], tag=4) # Cube boundaries


# --- Mesh settings and generation ---
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
gmsh.model.mesh.generate(3)
gmsh.model.mesh.optimize('Netgen')

model_rank = 0
mesh_comm = comm
mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank)
mesh = mesh_data[0]
ct = mesh_data[1]
ft = mesh_data[2]
ct.name = "ct"
ft.name = "ft"

with XDMFFile(mesh.comm, "copper_rod.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
    xdmf.write_meshtags(ft, mesh.geometry)

gmsh.finalize()
