from dolfin import *
from lib.MeshCreation import generate_square
from lib.Poromechanics import Poromechanics
from time import time
import numpy as np
import sys
initial_time = time()
if len(sys.argv) > 1:
    Nelements = int(sys.argv[1])
else:
    Nelements = 10
length = 64
mesh, markers, LEFT, RIGHT, TOP, BOTTOM, NONE = generate_square(
    Nelements, length)

# Refine on top


def refine_mesh(mesh):
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

# Everything by hand due to markers


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

neumann_solid_markers = [TOP]  # All others get weakly 0'd.
neumann_fluid_markers = []

ds = Measure('ds', domain=mesh, subdomain_data=markers,
             metadata={'optimize': True})
dx = dx(mesh)
dsNs = sum([ds(i) for i in neumann_solid_markers], ds(NONE))
dsNf = sum([ds(i) for i in neumann_fluid_markers], ds(NONE))

# Set up load terms
fs_vol = ff_vol = fs_sur = lambda t: Constant((0., 0.))


def p_source(t): return Constant(0)


def ff_sur(t): return Constant((0, 0))


def fs_sur(t):
    # return Expression(("0", "abs(x[0]-L)<L/2?-t*1e5:0"), t=min(0.5, t), L=length / 2, degree=4)
    return Expression(("0", "abs(x[0]-L)<L/2?-t*1e4:0"), t=3, L=length / 2, degree=6)


E = 3e4
nu = 0.2
mu_s = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
parameters = {"mu_f": 1e-3,
              "rhof": 1e3,
              "rhos": 500,
              "phi0": 1e-3,
              "mu_s": mu_s,
              "lmbda": lmbda,
              "ks": 1e6,
              "kf": 1e-7,
              "dt": 0.1,
              "t0": 0.0,
              "tf": 0.1,
              "fe degree solid": 1,
              "fe degree fluid": 2,
              "fe degree pressure": 1,
              "maxiter": 1000,
              "output solutions": True,
              # "output_name": "monolithic",
              "output name": "swelling",
              "betas": -0.5,
              "betaf": 0.,
              "betap": 1.,
              "solver rtol": 1e-8,
              "solver atol": 1e-10,
              "solver maxiter": 1000,
              "solver monitor": True,
              "solver type": "gmres",  # cg, gmres, aar
              "pc type": "undrained",  # diagonal, undrained, diagonal 3-way
              "inner ksp type": "cg",  # preonly, gmres, cg, bicgstab,
              "inner pc type": "asm",  # bjacobi, ilu, hypre, lu, gamg, asm
              "inner rtol": 1e-8,
              "inner atol": 1e-10,
              "inner maxiter": 1000,
              "inner accel order": 0,  # >1 diverges always, 1 works with gmres only.
              "AAR order": 10,
              "AAR p": 5,
              "AAR omega": 1,
              "AAR beta": 1,
              "dsNs": dsNs,
              "dsNf": dsNf,
              "ff_vol": ff_vol,
              "fs_vol": fs_vol,
              "ff_sur": ff_sur,
              "fs_sur": fs_sur,
              "p_source": p_source}

problem = Poromechanics(parameters, mesh)

# Set up BCs

V = problem.V

bcs_s = [DirichletBC(V.sub(0), Constant((0, 0)), markers, BOTTOM)]


def boundary_foot(x, on_boundary):
    return on_boundary and near(x[1], length) and abs(x[0] - length / 2) < length / 4


def boundary_foot_not(x, on_boundary):
    return on_boundary and not(near(x[1], length) and abs(x[0] - length / 2) < length / 4)


bcs_f = [DirichletBC(V.sub(1), Constant((0, 0)), boundary_foot)]
# At complement of intersection of solid and fluid
bcs_p = [DirichletBC(V.sub(2), Constant(0), boundary_foot_not)]
bcs = bcs_s + bcs_f

problem.set_bcs(bcs, bcs_p)
problem.solve()
