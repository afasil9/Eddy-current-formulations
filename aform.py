import numpy as np
from basix.ufl import element
from dolfinx import fem
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.fem import (
    Function,
    dirichletbc,
    form,
    locate_dofs_topological,
    petsc,
)
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import VTXWriter, XDMFFile
from dolfinx.mesh import GhostMode, create_submesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    curl,
    inner,
    variable,
)

from utils import L2_norm, par_print

with XDMFFile(MPI.COMM_WORLD, "copper_rod2.xdmf", "r") as xdmf:
    domain = xdmf.read_mesh(ghost_mode=GhostMode.none)
    ct = xdmf.read_meshtags(domain, name="ct")
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, 0)
    ft = xdmf.read_meshtags(domain, name="ft")
    material_tags = np.unique(ct.values)
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)


alpha_inner = 1e-8
beta_inner = 5.96e4
copper_tag = 1

alpha_outer = 1e-8
beta_outer = 0.0
outerboxtag = 2

const = fem.functionspace(domain, ("DG", 0))  # Piecewise constant function space

alpha = fem.Function(const)
alpha.interpolate(
    lambda x: np.full_like(x[0], alpha_inner), ct.find(copper_tag)
)  # Ct find which mesh tag you want
alpha.interpolate(lambda x: np.full_like(x[0], alpha_outer), ct.find(outerboxtag))

beta = fem.Function(const)
beta.interpolate(lambda x: np.full_like(x[0], beta_inner), ct.find(copper_tag))
beta.interpolate(lambda x: np.full_like(x[0], beta_outer), ct.find(outerboxtag))


comm = MPI.COMM_WORLD
degree = 1

V_CG = fem.functionspace(domain, ("CG", degree))

interior_nodes_array = fem.Function(V_CG)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = V_CG.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tagged_cells = ct.find(copper_tag)

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0
interior_nodes_array.x.scatter_forward()


ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 100  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)


x = SpatialCoordinate(domain)
a_n = fem.Function(A_space)

a_n_prev = a_n.copy()

f = as_vector((0.0, 0.0, 1.0))


A = TrialFunction(A_space)
v = TestFunction(A_space)

dx = Measure("dx", domain, subdomain_data=ct)

lhs = dt * inner(alpha * curl(A), curl(v)) * dx + inner(beta * A, v) * dx
rhs = dt * inner(f, v) * dx(copper_tag) + inner(beta * a_n, v) * dx


a = form(lhs)
L = form(rhs)

# Boundary conditions

boundary_facets = ft.find(4)

dofs = locate_dofs_topological(V=A_space, entity_dim=fdim, entities=boundary_facets)
u_bc = Function(A_space)
u_bc.x.array[:] = 0
bc = dirichletbc(u_bc, dofs)

# Solver steps
A_mat = assemble_matrix(a, bcs=[bc])
A_mat.assemble()


b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

uh = fem.Function(A_space)


# %%


# ksp = PETSc.KSP().create(domain.comm)
# ksp.setOperators(A_mat)
# ksp.setType("preonly")

# pc = ksp.getPC()
# pc.setType("lu")
# pc.setFactorSolverType("mumps")


ams_opts = {
    "ksp_atol": 1e-10,
    "ksp_rtol": 1e-10,
    "ksp_type": "cg",
    "ksp_max_it": 50,
    "ksp_monitor_true_residual": None,
    "ksp_norm_type": "unpreconditioned",
    "pc_hypre_ams_cycle_type": 13,
    "pc_hypre_ams_tol": 0.0,  # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1,  # Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_projection_frequency": 25,
    "pc_hypre_ams_relax_type": 2,
    "pc_hypre_ams_relax_weight": 1.0,
    "pc_hypre_ams_relax_times": 1,
    "pc_hypre_ams_omega": 1.0,
}

ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A_mat)
ksp.setOptionsPrefix(f"ksp_{id(ksp)}")

opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts.prefixPush(option_prefix)
for option, value in ams_opts.items():
    opts[option] = value
opts.prefixPop()

pc = ksp.getPC()
pc.setType("hypre")
pc.setHYPREType("ams")

G = discrete_gradient(V_CG._cpp_object, A_space._cpp_object)
G.assemble()
pc.setHYPREDiscreteGradient(G)

pc.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)

if degree == 1:
    cvec_0 = Function(A_space)
    cvec_0.interpolate(
        lambda x: np.vstack(
            (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_1 = Function(A_space)
    cvec_1.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
        )
    )
    cvec_2 = Function(A_space)
    cvec_2.interpolate(
        lambda x: np.vstack(
            (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
        )
    )

    pc.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )
else:
    Vec_CG = fem.functionspace(domain, ("CG", degree, (domain.geometry.dim,)))
    Pi = interpolation_matrix(Vec_CG._cpp_object, A_space._cpp_object)
    Pi.assemble()

    # Attach discrete gradient to preconditioner
    pc.setHYPRESetInterpolations(domain.geometry.dim, None, None, Pi, None)


ksp.setFromOptions()
ksp.setUp()
pc.setUp()

ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward()

a_n.x.array[:] = uh.x.array
a_n.x.scatter_forward()


vector_vis = fem.functionspace(
    domain, ("Discontinuous Lagrange", degree + 1, (domain.geometry.dim,))
)

A_vis = Function(vector_vis)
A_file = VTXWriter(domain.comm, "A.bp", A_vis, "BP4")
A_vis.interpolate(a_n)
A_file.write(t)

B = curl(a_n)
B_vis = Function(vector_vis)
B_file = VTXWriter(domain.comm, "B_field.bp", B_vis, "BP4")
Bexpr = fem.Expression(B, vector_vis.element.interpolation_points)
B_vis.interpolate(Bexpr)
B_file.write(t)

da_dt = (a_n - a_n_prev)/ dt
E = -da_dt
E_vis = Function(vector_vis)
Eexpr = fem.Expression(E, vector_vis.element.interpolation_points)
E_vis.interpolate(Eexpr)
E_file = VTXWriter(domain.comm, "E_field.bp", E_vis, "BP4")
E_file.write(t)

target_tags = copper_tag
cell_mask = np.isin(ct.values, target_tags)
inner_cells = ct.indices[cell_mask]
inner_submesh, parent_cells, _, _ = create_submesh(domain, tdim, inner_cells)

# Create submesh function space
DG_submesh_vis = fem.functionspace(
    inner_submesh, ("Discontinuous Lagrange", degree + 1, (inner_submesh.geometry.dim,))
)

B_vis_submesh = fem.Function(DG_submesh_vis)

submesh_cell_indices = np.arange(
    inner_submesh.topology.index_map(inner_submesh.topology.dim).size_local,
    dtype=np.int32,
)
parent_cell_indices = parent_cells.sub_topology_to_topology(submesh_cell_indices, False)

B_vis_submesh.interpolate(
    B_vis, cells0=parent_cell_indices, cells1=submesh_cell_indices
)

B_file_submesh = VTXWriter(domain.comm, "Rod_B.bp", B_vis_submesh, "BP4")
B_file_submesh.write(t)


for n in range(num_steps):
    t.expression().value += d_t

    with b.localForm() as loc_b:
        loc_b.set(0)
    petsc.assemble_vector(b, L)

    petsc.apply_lifting(b, [a], [[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    a_n.x.array[:] = uh.x.array
    a_n.x.scatter_forward()

    iterations = ksp.getIterationNumber()
    # print(f"number of iterations: {iterations}")
    reason = ksp.getConvergedReason()
    # print("Converged reason:", reason)

    B = curl(a_n)

    B_vis.interpolate(Bexpr)
    B_file.write(t)

    E_vis.interpolate(Eexpr)
    E_file.write(t)

    B_vis_submesh.interpolate(
        B_vis, cells0=parent_cell_indices, cells1=submesh_cell_indices
    )
    B_file.write(t)

    par_print(comm, f"L2 norm of B: {L2_norm(B_vis)}")
    par_print(comm, f"L2 norm of E: {L2_norm(E_vis)}")
    par_print(comm, f"L2 norm of B (submesh): {L2_norm(B_vis_submesh)}")
