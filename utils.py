import sys

import numpy as np
import ufl
from dolfinx import mesh
from dolfinx.cpp.refinement import RefinementOption
from dolfinx.fem import (
    Expression,
    Function,
    assemble_scalar,
    form,
    functionspace,
)
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode, refine, transfer_meshtag
from mpi4py import MPI
from ufl import dx, inner
from ufl.core.expr import Expr


def par_print(comm, string):
    if comm.rank == 0:
        print(string)
        sys.stdout.flush()


def L2_norm(v: Expr):
    """Computes the L2-norm of v"""
    return np.sqrt(
        MPI.COMM_WORLD.allreduce(assemble_scalar(form(inner(v, v) * dx)), op=MPI.SUM)
    )


def monitor(ksp, its, rnorm):
    iteration_count = []
    residual_norm = []
    iteration_count.append(its)
    residual_norm.append(rnorm)
    print("Iteration: {}, preconditioned residual: {}".format(its, rnorm))


def boundary_marker(x):
    """Marker function for the boundary of a unit cube"""
    # Collect boundaries perpendicular to each coordinate axis
    boundaries = [
        np.logical_or(np.isclose(x[i], 0.0), np.isclose(x[i], 1.0)) for i in range(3)
    ]
    return np.logical_or(np.logical_or(boundaries[0], boundaries[1]), boundaries[2])


def error_L2(uh, u_ex, degree_raise=4):
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree
    family = uh.function_space.ufl_element().family_name
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = Function(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = Function(W)
    if isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = Function(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(inner(e_W, e_W) * dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)
    return np.sqrt(error_global)


def markers_to_meshtags(msh, tags, markers, dim):
    entities = [mesh.locate_entities_boundary(msh, dim, marker) for marker in markers]
    values = [np.full_like(entities, tag) for (tag, entities) in zip(tags, entities)]
    entities = np.hstack(entities, dtype=np.int32)
    values = np.hstack(values, dtype=np.intc)
    perm = np.argsort(entities)
    return mesh.meshtags(msh, dim, entities[perm], values[perm])

def convert_facet_tags(msh, submesh, cell_map, facet_tag):
    msh_facets = facet_tag.indices

    # Connectivities
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim, tdim - 1)
    msh.topology.create_connectivity(tdim - 1, tdim)
    msh_c_to_f = msh.topology.connectivity(tdim, tdim - 1)
    msh_f_to_c = msh.topology.connectivity(tdim - 1, tdim)
    submesh.topology.create_connectivity(tdim, tdim - 1)
    submesh_c_to_f = submesh.topology.connectivity(tdim, tdim - 1)

    # NOTE: Tagged facets mat not have a cell in the submesh, or may
    # have more than one cell in the submesh
    submesh_facets = []
    submesh_values = []
    for i, facet in enumerate(msh_facets):
        cells = msh_f_to_c.links(facet)
        for cell in cells:
            if cell in cell_map:
                local_facet = msh_c_to_f.links(cell).tolist().index(facet)
                # FIXME Don't hardcode cell type
                assert local_facet >= 0  # and local_facet <= 2
                submesh_cell = np.where(cell_map == cell)[0][0]
                submesh_facet = submesh_c_to_f.links(submesh_cell)[local_facet]
                submesh_facets.append(submesh_facet)
                submesh_values.append(facet_tag.values[i])
    submesh_facets = np.array(submesh_facets)
    submesh_values = np.array(submesh_values, dtype=np.intc)
    # Sort and make unique
    submesh_facets, ind = np.unique(submesh_facets, return_index=True)
    submesh_values = submesh_values[ind]
    submesh_meshtags = mesh.meshtags(
        submesh, submesh.topology.dim - 1, submesh_facets, submesh_values
    )
    return submesh_meshtags


def create_mesh_fenics(comm, n, boundaries):
    # Create mesh
    msh = mesh.create_unit_cube(MPI.COMM_WORLD, n, n, n, mesh.CellType.tetrahedron)

    # Create facet meshtags
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Define markers for all 6 faces of the cube
    markers = [
        lambda x: np.isclose(x[2], 0.0),  # bottom (z = 0)
        lambda x: np.isclose(x[2], 1.0),  # top (z = 1)
        lambda x: np.isclose(x[1], 0.0),  # front (y = 0)
        lambda x: np.isclose(x[0], 1.0),  # right (x = 1)
        lambda x: np.isclose(x[1], 1.0),  # back (y = 1)
        lambda x: np.isclose(x[0], 0.0),  # left (x = 0)
    ]
    
    # Create facet tags
    ft = markers_to_meshtags(msh, boundaries.values(), markers, fdim)
    
    # Create domain tags
    ct = mesh.meshtags(msh, tdim, np.array(range(msh.topology.index_map(tdim).size_local), dtype=np.int32), 
                      np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32))

    # print("number of cells is", msh.topology.index_map(tdim).size_local)
    return msh, ft, ct

def refine_mesh(domain, ft, ct, refinement_level):
    for i in range(refinement_level):
        domain.topology.create_entities(1)

        fine_mesh, parent_cell, parent_facet = refine(domain, option=RefinementOption.parent_cell_and_facet)

        tdim = fine_mesh.topology.dim - 1
        fine_mesh.topology.create_connectivity(fine_mesh.topology.dim, tdim)

        ft_ref = transfer_meshtag(ft, fine_mesh, parent_cell, parent_facet)
        ct_ref = transfer_meshtag(ct, fine_mesh, parent_cell, parent_facet)

        domain, ft, ct = fine_mesh, ft_ref, ct_ref

    domain.topology.create_connectivity(tdim - 1, tdim)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    ct.name = "ct"
    ft.name = "ft"

    with XDMFFile(domain.comm, "copper_rod_refined.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_meshtags(ct, domain.geometry)
        xdmf.write_meshtags(ft, domain.geometry)

    return domain, ct, ft


def my_monitor(ksp, its, rnorm):
    print(f"Iter {its}, residual = {rnorm}")


def interpolate_by_tags(function, value_dict, domain_tags):
    for tag, value in value_dict.items():
        function.interpolate(lambda x: np.full_like(x[0], value), domain_tags.find(tag))
    function.x.scatter_forward()


def boundary_marker_copper(x):
    tol = 1e-6
    markers = np.zeros(x.shape[1], dtype=np.int32)

    # X-normal boundaries
    x_bdy = np.logical_or(np.isclose(x[0], 0.0, atol=tol),
                          np.isclose(x[0], 1000.0, atol=tol))
    markers[x_bdy] = 1

    # Y-normal boundaries
    y_bdy = np.logical_or(np.isclose(x[1], 0.0, atol=tol),
                          np.isclose(x[1], 1000.0, atol=tol))
    markers[y_bdy] = 2

    # Z-normal boundaries (note: interval [425, 575])
    z_bdy = np.logical_or(x[2] <= 425.0 + tol,
                          x[2] >= 575.0 - tol)
    markers[z_bdy] = 3

    return markers