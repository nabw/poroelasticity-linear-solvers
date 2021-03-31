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
              "tf": 1.0,
              "Kdr": 4700,  # Using isotropic formula. Kdr = 2 * mu_s / d + lmbda_s
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
              "solver type": "gmres",  # cg, gmres, AAR
              "pc type": "diagonal",  # diagonal, undrained, diagonal 3-way
              "inner pc type": "lu",  # bjacobi, ilu, hypre, lu
              "inner accel order": 0,
              "AAR order": 10,
              "AAR p": 5,
              "AAR omega": 1,
              "AAR beta": 1,
              "dsNs": dsNs,
              "dsNf": dsNf}

# Set up load terms
fs_vol = ff_vol = fs_sur = lambda t: Constant((0., 0.))


def p_source(t): return Constant(0.0)


def ff_sur(t):
    return Constant(-1e3 * 0.1 * (1 - exp(-(t**2) / 0.25))) * FacetNormal(mesh)


parameters["ff_vol"] = ff_vol
parameters["fs_vol"] = fs_vol
parameters["ff_sur"] = ff_sur
parameters["fs_sur"] = fs_sur
parameters["p_source"] = p_source

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

# Solve
problem.solve()
