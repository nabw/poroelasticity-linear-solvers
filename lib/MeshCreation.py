#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 14:06:24 2019

@author: barnafi
"""
NONE = 99  # Marker for empty boundary


def generate_square(Nelements, length, refinements=0):
    """
    Creates a square mesh of given elements and length with markers on
    the sides: left, bottom, right and top
    """
    from dolfin import UnitSquareMesh, SubDomain, MeshFunction, Measure, near, refine
    mesh = UnitSquareMesh(Nelements, Nelements)
    for i in range(refinements):
        mesh = refine(mesh)
    mesh.coordinates()[:] *= length

    # Subdomains: Solid
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], length) and on_boundary

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], length) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary
    left, right, top, bottom = Left(), Right(), Top(), Bottom()
    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 3, 4  # Set numbering
    NONE = 99  # Marker for empty boundary

    markers = MeshFunction("size_t", mesh, 1)
    markers.set_all(0)

    boundaries = (left, right, top, bottom)
    def_names = (LEFT, RIGHT, TOP, BOTTOM)
    for side, num in zip(boundaries, def_names):
        side.mark(markers, num)

    return mesh, markers, LEFT, RIGHT, TOP, BOTTOM, NONE


def generate_footing_square(Nelements, length, refinements=0):
    from dolfin import UnitSquareMesh, SubDomain, MeshFunction, Measure, near, refine, cells
    import numpy as np
    # Start from square
    mesh = generate_square(Nelements, length, 0)[0]

    def refine_mesh(mesh):
        # Refine on top
        cell_markers = MeshFunction('bool', mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for c in cells(mesh):
            verts = np.reshape(c.get_vertex_coordinates(), (3, 2))
            verts_x = verts[:, 0]
            verts_y = verts[:, 1]
            newval = verts_y.min() > 2 * length / 3 and verts_x.min() > length / \
                8 and verts_x.max() < 7 / 8 * length
            cell_markers[c] = newval

        # Redefine markers on new mesh
        return refine(mesh, cell_markers)

    mesh = refine_mesh(refine_mesh(mesh))

    for i in range(refinements):
        mesh = refine(mesh)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], length) and on_boundary

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], length) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary

    left, right, top, bottom = Left(), Right(), Top(), Bottom()
    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 3, 4  # Set numbering
    markers = MeshFunction("size_t", mesh, 1)
    markers.set_all(0)

    boundaries = (left, right, top, bottom)
    def_names = (LEFT, RIGHT, TOP, BOTTOM)
    for side, num in zip(boundaries, def_names):
        side.mark(markers, num)
    return mesh, markers, LEFT, RIGHT, TOP, BOTTOM, NONE


def generate_rectangle(x0, y0, x1, y1, nx, ny):
    """
    Creates a square mesh of given elements and length with markers on
    the sides: left, bottom, right and top
    """
    from dolfin import RectangleMesh, Point, SubDomain, MeshFunction, Measure, near
    mesh = RectangleMesh(Point(x0, y0), Point(x1, y1), nx, ny)

    # Subdomains: Solid
    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x0) and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], x1) and on_boundary

    class Top(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], y1) and on_boundary

    class Bottom(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], y0) and on_boundary
    left, right, top, bottom = Left(), Right(), Top(), Bottom()
    LEFT, RIGHT, TOP, BOTTOM = 1, 2, 3, 4  # Set numbering
    NONE = 99  # Marker for empty boundary

    markers = MeshFunction("size_t", mesh, 1)
    markers.set_all(0)

    boundaries = (left, right, top, bottom)
    def_names = (LEFT, RIGHT, TOP, BOTTOM)
    for side, num in zip(boundaries, def_names):
        side.mark(markers, num)

    return mesh, markers, LEFT, RIGHT, TOP, BOTTOM, NONE


def prolateGeometry(filename):

    from dolfin import XDMFFile, Mesh, MeshValueCollection, MeshTransformation
    xdmf_meshfile = "meshes/" + filename + ".xdmf"
    xdmf_meshfile_bm = "meshes/" + filename + "_bm.xdmf"
    mesh = Mesh()
    with XDMFFile(xdmf_meshfile) as infile:
        infile.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile(xdmf_meshfile_bm) as infile:
        infile.read(mvc, "name_to_read")
    from dolfin import cpp
    markers = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    ENDOCARD = 20
    EPICARD = 10
    BASE = 50
    NONE = 99

    MeshTransformation.scale(mesh, 1e-3)
    return mesh, markers, ENDOCARD, EPICARD, BASE, NONE


def generate_cube(Nelements, length, refinements=0):
    """
    Creates a square mesh of given elements and length with markers on
    the sides: left, bottom, right and top
    """
    from dolfin import UnitCubeMesh, SubDomain, MeshFunction, Measure, near, refine
    mesh = UnitCubeMesh(Nelements, Nelements, Nelements)
    for i in range(refinements):
        mesh = refine(mesh)
    mesh.coordinates()[:] *= length

    # Subdomains: Solid
    class Xp(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], length) and on_boundary

    class Xm(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and on_boundary

    class Yp(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], length) and on_boundary

    class Ym(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary

    class Zp(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], length) and on_boundary

    class Zm(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[2], 0.0) and on_boundary
    xp, xm, yp, ym, zp, zm = Xp(), Xm(), Yp(), Ym(), Zp(), Zm()
    XP, XM, YP, YM, ZP, ZM = 1, 2, 3, 4, 5, 6  # Set numbering

    markers = MeshFunction("size_t", mesh, 2)
    markers.set_all(0)

    boundaries = (xp, xm, yp, ym, zp, zm)
    def_names = (XP, XM, YP, YM, ZP, ZM)
    for side, num in zip(boundaries, def_names):
        side.mark(markers, num)

    return mesh, markers, XP, XM, YP, YM, ZP, ZM


def generate_boundary_measure(mesh, markers, tags_list, none_tag=99):
    from dolfin import Measure

    ds = Measure('ds', domain=mesh, subdomain_data=markers,
                 metadata={'optimize': True})
    return sum([ds(i) for i in tags_list], ds(none_tag))
