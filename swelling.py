from dolfin import *
from lib.MeshCreation import generate_square
from time import time
from lib.Poromechanics import Poromechanics
initial_time = time()
# Create geometry and set Neumann boundaries
Nelements = 20
side_length = 1e-2
mesh, markers, LEFT, RIGHT, TOP, BOTTOM, NONE = generate_square(
    Nelements, side_length)


neumann_solid_markers = [TOP, RIGHT]
neumann_fluid_markers = [LEFT]

ds = Measure('ds', domain=mesh, subdomain_data=markers,
             metadata={'optimize': True})
dsNs = sum([ds(i) for i in neumann_solid_markers], ds(NONE))
dsNf = sum([ds(i) for i in neumann_fluid_markers], ds(NONE))

# Set up load terms
fs_vol = ff_vol = fs_sur = lambda t: Constant((0., 0.))


def p_source(t): return Constant(0.0)


def ff_sur(t):
    return Constant(-1e3 * 0.1 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


parameters = {"mu_f": 0.035,
              "rhof": 1e3,
              "rhos": 1e3,
              "phi0": 0.1,
              "mu_s": 4000,
              "lmbda": 700,
              "ks": 1e6,
              "kf": 1e-7,
              "dt": 0.1,
              "t0": 0.0,
              "tf": 0.1,
              "fe degree solid": 2,
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
              "pc type": "diagonal 3-way",  # diagonal, undrained, diagonal 3-way
              "inner ksp type": "cg",  # preonly, gmres, cg, bicgstab,
              "inner pc type": "gamg",  # bjacobi, ilu, hypre, lu, gamg
              "inner rtol": 0,
              "inner atol": 1e-8,
              "inner maxiter": 100,
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

bcs_s = [DirichletBC(V.sub(0).sub(0), Constant(0), markers, LEFT),
         DirichletBC(V.sub(0).sub(1), Constant(0), markers, BOTTOM)]
bcs_f = [DirichletBC(V.sub(1), Constant((0, 0)), markers, TOP),
         DirichletBC(V.sub(1), Constant((0, 0)), markers, BOTTOM)]
bcs_p = [DirichletBC(V.sub(2), Constant(0), markers, LEFT),
         DirichletBC(V.sub(
             2), Constant(0), markers, TOP),
         DirichletBC(V.sub(2), Constant(0), markers, RIGHT)]
bcs = bcs_s + bcs_f

problem.set_bcs(bcs, bcs_p)
problem.solve()
