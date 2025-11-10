# %%
from mpi4py import MPI
from dolfinx import fem
from dolfinx.mesh import create_unit_cube, locate_entities_boundary, CellType
from dolfinx.fem import (
    Function,
    dirichletbc,
    locate_dofs_topological,
    form,
    petsc,
)
from dolfinx.fem.petsc import assemble_matrix
import numpy as np
from ufl import curl, TrialFunction, TestFunction, inner, dx, SpatialCoordinate
from basix.ufl import element
from petsc4py import PETSc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import meshtags
from problems import sinusoidal_time, quadratic_time
from utils import L2_norm
from ufl import variable, diff
import argparse


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--reuse", dest="reuse", action="store_true", help="Reuse KSP/PC between steps")
group.add_argument("--no-reuse", dest="reuse", action="store_false", help="Do not reuse KSP/PC")
parser.set_defaults(reuse=False)  # current default
args = parser.parse_args()

comm = MPI.COMM_WORLD
degree = 1

n = 8
domain = create_unit_cube(MPI.COMM_WORLD, n, n, n, cell_type=CellType.hexahedron)

tdim = domain.topology.dim
fdim = tdim - 1

ti = 0.0  # Start time
T = 0.1  # End time
num_steps = 10  # Number of time steps
d_t = (T - ti) / num_steps  # Time step size

t = variable(fem.Constant(domain, ti))
dt = fem.Constant(domain, d_t)

alpha = 1.0
DG = fem.functionspace(domain, ("DG", 0))
beta = Function(DG)

size_beta = 2.0
beta_loc = size_beta / n
eps = 1e-10

beta.interpolate(
    lambda x: np.where(
        (np.abs(x[0] - 0.5) < beta_loc - eps)
        & (np.abs(x[1] - 0.5) < beta_loc - eps)
        & (np.abs(x[2] - 0.5) < beta_loc - eps),
        1.0,
        0.0,
    )
)

num_cells = domain.topology.index_map(domain.topology.dim).size_local
cell_indices = np.arange(num_cells, dtype=np.int32)
beta_cell_values = beta.x.array.astype(np.int32)

ct = meshtags(domain, domain.topology.dim, cell_indices, beta_cell_values)

V_CG = fem.functionspace(domain, ("CG", degree))

facets = locate_entities_boundary(
    domain,
    dim=fdim,
    marker=lambda x: np.isclose(x[0], 0.0)
    | np.isclose(x[1], 0.0)
    | np.isclose(x[2], 0.0)
    | np.isclose(x[0], 1.0)
    | np.isclose(x[1], 1.0)
    | np.isclose(x[2], 1.0),
)


interior_nodes_array = fem.Function(V_CG)

interior_nodes_array.x.array[:] = 1.0
interior_nodes_array.x.scatter_forward()

dofmap = V_CG.dofmap
num_dofs_per_cell = dofmap.dof_layout.num_dofs
cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

tagged_cells = ct.find(1)

tagged_cell_dofs = cell_dofs[tagged_cells].flatten()
unique_dofs = np.unique(tagged_cell_dofs)

interior_nodes_array.x.array[unique_dofs] = 0.0
interior_nodes_array.x.scatter_forward()

nedelec_elem = element("N1curl", domain.basix_cell(), degree)
A_space = fem.functionspace(domain, nedelec_elem)


x = SpatialCoordinate(domain)
uex = sinusoidal_time(x, t)

a_n = fem.Function(A_space)
uex_expr = fem.Expression(uex, A_space.element.interpolation_points)
a_n.interpolate(uex_expr)

#%%

print(f"t value is {t.expression().value}")

f = curl(alpha * curl(uex)) + diff(beta * uex, t)

A = TrialFunction(A_space)
v = TestFunction(A_space)

lhs = dt * inner(alpha * curl(A), curl(v)) * dx + inner(beta * A, v) * dx
rhs = dt * inner(f, v) * dx + inner(beta * a_n, v) * dx

a = form(lhs)
L = form(rhs)

# Boundary conditions

dofs = locate_dofs_topological(V=A_space, entity_dim=fdim, entities=facets)
u_bc_expr = fem.Expression(uex, A_space.element.interpolation_points)
u_bc = Function(A_space)
u_bc.interpolate(u_bc_expr)
bc = dirichletbc(u_bc, dofs)

# Solver steps
A_mat = assemble_matrix(a, bcs=[bc])
A_mat.assemble()

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])


ams_opts = {
    "ksp_atol": 1e-10,
    "ksp_rtol": 1e-10,
    "ksp_type": "cg",
    "ksp_max_it": 15,
    # "ksp_monitor_true_residual": None,
    "ksp_norm_type": "unpreconditioned",
    "pc_hypre_ams_cycle_type": 13,
    "pc_hypre_ams_tol": 0.0,  # Default is 1e-6 but we set it to 0.0 for AMS to be used as preconditioner
    "pc_hypre_ams_max_iter": 1,  # Set to 1 to use AMS as a preconditioner
    "pc_hypre_ams_print_level": 1,
    "pc_hypre_ams_amg_alpha_options": "10,1,6,6,4",
    "pc_hypre_ams_amg_beta_options": "10,1,6,6,4",
    "pc_hypre_ams_projection_frequency": 200, # Need to change this
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

# opts = PETSc.Options()
# opts[f"{ksp.prefix}pc_hypre_ams_cycle_type"] = 14
# opts[f"{ksp.prefix}pc_hypre_ams_tol"] = 0
# opts[f"{ksp.prefix}pc_hypre_ams_max_iter"] = 1
# opts[f"{ksp.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
# opts[f"{ksp.prefix}pc_hypre_ams_print_level"] = 1
# opts[f"{ksp.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,3"
# opts[f"{ksp.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,3"
# opts[f"{ksp.prefix}pc_hypre_ams_print_level"] = 0

# ksp.setFromOptions()


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


uh = fem.Function(A_space)

print("b norm before solve:", b.norm())

ksp.solve(b, uh.x.petsc_vec)
a_n.x.array[:] = uh.x.array

print(f"norm of f = {L2_norm(f)}")
print(f"norm of uex = {L2_norm(uex)}")
print(f"norm of a_n = {L2_norm(a_n)}")
print(f"Converged reason: {ksp.getConvergedReason()}")
print(f"Iteration count is {ksp.getIterationNumber()}")

t.expression().value += d_t 

reuse = args.reuse

print(f"Reuse is set to {reuse}")


#%%
for i in range(5):

    print("\n")
    print(f"STEP NUMBER IS {i+1}")
    t.expression().value += d_t
    print(f"t value is {t.expression().value}")
    # pc.HYPREAMSResetSolveCounter()

    # Boundary conditions

    u_bc.interpolate(u_bc_expr)


    b = petsc.assemble_vector(L)
    petsc.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])

    if reuse == True:
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
    


       

    uh = fem.Function(A_space)

    print("b norm before solve:", b.norm())

    ksp.solve(b, uh.x.petsc_vec)
    a_n.x.array[:] = uh.x.array

    # print(f"norm of f = {L2_norm(f)}")
    # print(f"norm of uex = {L2_norm(uex)}")
    # print(f"norm of a_n = {L2_norm(a_n)}")
    print(f"Converged reason: {ksp.getConvergedReason()}")
    print(f"Iteration count is {ksp.getIterationNumber()}")

