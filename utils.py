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
from dolfinx.mesh import refine, transfer_meshtag
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




def convert_facet_tags(submsh, cell_emap, facet_tags):
    """Convert facet tags from a mesh to a submesh using an entity map."""

    # Each tagged facet may be connected to up to two cells.
    # We create arrays of cells and local facets to store the connected cells
    # if they exist i.e. cells[i] corresponds to the first connected cell to facet
    # facet_tags.indices[i] if it exists, and cells[i + 1] corresponds to the second
    # connected cell if it exists.
    cells = np.full(2 * len(facet_tags.indices), -1, dtype=np.int32)
    # Similar array for local facets. # local_facets[i] corresponds to the local facet
    # in cells[i]
    local_facets = np.full(2 * len(facet_tags.indices), -1, dtype=np.int32)

    tdim = cell_emap.topology.dim
    fdim = tdim - 1

    # Get required connectivities
    cell_emap.topology.create_connectivity(fdim, tdim)
    cell_emap.topology.create_connectivity(tdim, fdim)
    f_to_c = cell_emap.topology.connectivity(fdim, tdim)
    c_to_f = cell_emap.topology.connectivity(tdim, fdim)

    cell_emap.sub_topology.create_connectivity(tdim, fdim)
    c_to_f_sub = cell_emap.sub_topology.connectivity(tdim, fdim)

    # Loop through all facets and get the (cell, local facet index) pairs
    for i, facet in enumerate(facet_tags.indices):
        # Each tagged facet may be connected to up to two cells in the mesh
        cs = f_to_c.links(facet)
        # Add the cells and local facets to the arrays
        for j, c in enumerate(cs):
            cells[2 * i + j] = c
            local_facets[2 * i + j] = np.where(c_to_f.links(c) == facet)[0][0]

    # Map cells to the submesh using the entity map. Note that some facets will only be
    # connected to one cell, so we must only map the cells that are >= 0. Also note that
    # not all cells in the mesh will be present in the submesh, so some of the returned
    # cells may be -1.
    cells[cells >= 0] = cell_emap.sub_topology_to_topology(cells[cells >= 0], inverse=True)

    # Loop through facets and get the corresponding facets in the submesh. Add the index
    # and corresponding value to the lists
    facets_sub = []
    values_sub = []
    for i in range(len(facet_tags.indices)):
        for j in range(2):
            if cells[2 * i + j] >= 0:
                facet_sub = c_to_f_sub.links(cells[2 * i + j])[local_facets[2 * i + j]]
                facets_sub.append(facet_sub)
                values_sub.append(facet_tags.values[i])

    # Sort and make unique
    facets_sub = np.array(facets_sub)
    values_sub = np.array(values_sub, dtype=np.intc)
    facets_sub, ind = np.unique(facets_sub, return_index=True)
    values_sub = values_sub[ind]
    facet_tags_sub = mesh.meshtags(submsh, fdim, facets_sub, values_sub)
    return facet_tags_sub



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