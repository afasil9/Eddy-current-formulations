#%%
import gmsh
from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

comm = MPI.COMM_WORLD
start = 0.1
div = 16  # Number of divisions
h = start / div

# --- Outer Box Parameters ---
Lx = 3.0   # length in x-direction
Ly = 2.0   # length in y-direction
Lz = 0.5   # height in z-direction
lc = 1.0

boundary_tags = {
    "cube_boundary": 1,
    "upper_surface": 2,
    "side_surface": 3,
    "bottom_surface": 4
}

# --- Inner Cylinder Parameters ---
cyl_center_x, cyl_center_y = Lx / 2, Ly / 2
cyl_radius = 0.1
cyl_height = Lz       # full height of the box
cyl_base_z = 0.0      # start at bottom

gmsh.initialize()

# --- Outer Box ---
p1 = gmsh.model.occ.addPoint(0, 0, 0, lc)
p2 = gmsh.model.occ.addPoint(Lx, 0, 0, lc)
p3 = gmsh.model.occ.addPoint(Lx, Ly, 0, lc)
p4 = gmsh.model.occ.addPoint(0, Ly, 0, lc)

cl = gmsh.model.occ.addCurveLoop([
    gmsh.model.occ.addLine(p1, p2),
    gmsh.model.occ.addLine(p2, p3),
    gmsh.model.occ.addLine(p3, p4),
    gmsh.model.occ.addLine(p4, p1),
])

surface = gmsh.model.occ.addPlaneSurface([cl])
extrude_result = gmsh.model.occ.extrude([(2, surface)], 0, 0, Lz)
outer_volume = extrude_result[1]  # (3, vol_tag)

# --- Inner Cylinder ---
circle = gmsh.model.occ.addCircle(cyl_center_x, cyl_center_y, cyl_base_z, cyl_radius)
curve_loop = gmsh.model.occ.addCurveLoop([circle])
cyl_base_surface = gmsh.model.occ.addPlaneSurface([curve_loop])
extrude_result_inner = gmsh.model.occ.extrude([(2, cyl_base_surface)], 0, 0, cyl_height)
inner_volume = extrude_result_inner[1]  # (3, vol_tag)

# --- Boolean Fragment (Split box and cylinder) ---
model_dim_tags = gmsh.model.occ.fragment(
    [(3, outer_volume[1])],
    [(3, inner_volume[1])]
)
gmsh.model.occ.synchronize()

# --- Tagging volumes (Physical groups for subdomains) ---
gmsh.model.addPhysicalGroup(3, [model_dim_tags[0][0][1]], tag=1)  # Cylinder
gmsh.model.addPhysicalGroup(3, [model_dim_tags[0][1][1]], tag=2)  # Remaining box

# --- Boundary tags (top 2D surfaces) ---
boundary_cube = gmsh.model.getBoundary([model_dim_tags[0][1]], oriented=False)
boundary_ids_cube = [b[1] for b in boundary_cube]

boundary_cylinder = gmsh.model.getBoundary([model_dim_tags[0][0]], oriented=False)
boundary_ids_cylinder = [b[1] for b in boundary_cylinder]

gmsh.model.occ.synchronize()

cyl_surfs = [
    s for s in gmsh.model.getBoundary([(3, inner_volume[1])], oriented=False, combined=False)
    if s[0] == 2
]
cyl_surface_tags = [s[1] for s in cyl_surfs]

# --- Mesh refinement near the cylinder ---
distance = gmsh.model.mesh.field.add("Distance")
gmsh.model.mesh.field.setNumbers(distance, "FacesList", cyl_surface_tags)

r = cyl_radius
LcMin = r / 8          # very fine near the rod
LcMax = h * 8          # coarser far away
DistMin = 1.0 * r
DistMax = 8.0 * r

threshold = gmsh.model.mesh.field.add("Threshold")
gmsh.model.mesh.field.setNumber(threshold, "IField", distance)
gmsh.model.mesh.field.setNumber(threshold, "LcMin", LcMin)
gmsh.model.mesh.field.setNumber(threshold, "LcMax", LcMax)
gmsh.model.mesh.field.setNumber(threshold, "DistMin", DistMin)
gmsh.model.mesh.field.setNumber(threshold, "DistMax", DistMax)

minf = gmsh.model.mesh.field.add("Min")
gmsh.model.mesh.field.setNumbers(minf, "FieldsList", [threshold])
gmsh.model.mesh.field.setAsBackgroundMesh(minf)

gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)

# --- Boundary Physical Groups ---
gmsh.model.addPhysicalGroup(2, boundary_ids_cube[1:], tag=boundary_tags["cube_boundary"]) # Box boundaries

# Find intersection
common = list(set(boundary_ids_cube) & set(boundary_ids_cylinder))

gmsh.model.addPhysicalGroup(2, common, tag=boundary_tags["side_surface"])   # Cylinder side
gmsh.model.addPhysicalGroup(2, [boundary_ids_cylinder[0]], tag=boundary_tags["upper_surface"])  # Cylinder top
gmsh.model.addPhysicalGroup(2, [boundary_ids_cylinder[2]], tag=boundary_tags["bottom_surface"]) # Cylinder bottom

# --- Mesh generation ---
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
gmsh.model.mesh.generate(3)

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
