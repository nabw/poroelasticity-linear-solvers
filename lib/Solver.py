from petsc4py import PETSc
from mpi4py import MPI
from lib.Printing import parprint
from lib.AAR import AAR
from time import perf_counter as time


def converged(_ksp, _it, _rnorm, *args, **kwargs):
    """
    args must have: index_map, dummy, dummy_s, dummy_f, dummy_p, b0_s, b0_f, b0_p.
    dummy is used to avoid allocation of new vector for residual. [is is somewhere in PETSc...?]
    """
    dummy = kwargs['dummy']
    index_map = kwargs['index_map']
    dummy_s, dummy_f, dummy_p = kwargs['dummy_subs']
    b0_s,  b0_f, b0_p = kwargs['b0_norms']
    normalize = max(b0_s,  b0_f, b0_p)
    _ksp.buildResidual(dummy)

    # Get residual subcomponents
    dummy.getSubVector(index_map.is_s, dummy_s)
    dummy.getSubVector(index_map.is_f, dummy_f)
    dummy.getSubVector(index_map.is_p, dummy_p)

    res_s_a = dummy_s.norm(PETSc.NormType.NORM_INFINITY)
    res_s_r = res_s_a/normalize  # /b0_s
    res_f_a = dummy_f.norm(PETSc.NormType.NORM_INFINITY)
    res_f_r = res_f_a/normalize  # /b0_f
    res_p_a = dummy_p.norm(PETSc.NormType.NORM_INFINITY)
    res_p_r = res_p_a/normalize  # /b0_p

    error_abs = max(res_s_a, res_f_a, res_p_a)
    error_rel = max(res_s_r, res_f_r, res_p_r)
    if kwargs['monitor']:
        width = 11
        if _it == 0:
            parprint("KSP errors: {}, {}, {}, {}, {}, {}".format('abs_s'.rjust(width), 'abs_f'.rjust(
                width), 'abs_p'.rjust(width), 'rel_s'.rjust(width), 'rel_f'.rjust(width), 'rel_p'.rjust(width)))

        parprint("KSP it {}:   {:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}".format(
            _it, res_s_a, res_f_a, res_p_a, res_s_r, res_f_r, res_p_r))
    if error_abs < _ksp.atol or error_rel < _ksp.rtol:
        # Convergence
        parprint("---- [Solver] Converged")
        return 1
    elif _it > _ksp.max_it or error_abs > _ksp.divtol:
        # Divergence
        return -1
    else:
        # Continue
        return 0


class Solver:
    def __init__(self, A, b, PC, parameters, index_map):
        self.A = A
        self.b = b
        self.PC = PC
        self.solver = None
        self.parameters = parameters
        self.index_map = index_map
        self.t_total = 0

    def create_solver(self, A, b, PC):
        t0_create = time()

        b = self.b.vec()
        self.dummy = b.copy()
        self.dummy_s = PETSc.Vec().create()
        self.dummy_f = PETSc.Vec().create()
        self.dummy_p = PETSc.Vec().create()
        b.getSubVector(self.index_map.is_s, self.dummy_s)
        b.getSubVector(self.index_map.is_f, self.dummy_f)
        b.getSubVector(self.index_map.is_p, self.dummy_p)
        b.restoreSubVector(self.index_map.is_s, self.dummy_s)
        b.restoreSubVector(self.index_map.is_f, self.dummy_f)
        b.restoreSubVector(self.index_map.is_p, self.dummy_p)

        solver_type = self.parameters["solver type"]
        atol = self.parameters["solver atol"]
        rtol = self.parameters["solver rtol"]
        maxiter = self.parameters["solver maxiter"]
        monitor_convergence = self.parameters["solver monitor"]
        if self.parameters["solver type"] == "aar":
            order = self.parameters["AAR order"]
            p = self.parameters["AAR p"]
            omega = self.parameters["AAR omega"]
            beta = self.parameters["AAR beta"]
            self.solver = AAR(order, p, omega, beta, self.A.mat(), x0=None, pc=self.PC, atol=atol,
                              rtol=rtol, maxiter=maxiter, monitor_convergence=monitor_convergence)
        else:
            solver = PETSc.KSP().create()
            solver.setOptionsPrefix("global_")
            # solver.setInitialGuessNonzero(True)
            solver.setOperators(self.A.mat())
            solver.setType(self.parameters["solver type"])
            solver.setTolerances(rtol, atol, 1e20, maxiter)
            solver.setPC(self.PC)
            if solver_type == "gmres":
                solver.setGMRESRestart(maxiter)
            solver.setFromOptions()
            self.solver = solver
        parprint("---- [Solver] Solver created in {}s".format(time() - t0_create))

    def set_up(self):
        t0_setup = time()
        # Create linear solver
        solver_type = self.parameters["solver type"]
        atol = self.parameters["solver atol"]
        rtol = self.parameters["solver rtol"]
        maxiter = self.parameters["solver maxiter"]
        monitor_convergence = self.parameters["solver monitor"]

        # Prepare elements for convergence test:
        args = None
        # index_map, b, dummy, dummy_s, dummy_f, dummy_p, b0_s, b0_f, b0_p
        b = self.b.vec()
        dummy = b.copy()
        b_s = PETSc.Vec().create()
        b_f = PETSc.Vec().create()
        b_p = PETSc.Vec().create()
        b.getSubVector(self.index_map.is_s, b_s)
        dummy_s = b_s.copy()
        b.getSubVector(self.index_map.is_f, b_f)
        dummy_f = b_f.copy()
        b.getSubVector(self.index_map.is_p, b_p)
        dummy_p = b_p.copy()
        b0_s = b_s.norm()
        b0_f = b_f.norm()
        b0_p = b_p.norm()
        b.restoreSubVector(self.index_map.is_s, b_s)
        b.restoreSubVector(self.index_map.is_f, b_f)
        b.restoreSubVector(self.index_map.is_p, b_p)
        # if b0_s < 1e-13:
        #     b0_s = 1
        # if b0_f < 1e-13:
        #     b0_f = 1
        # if b0_p < 1e-13:
        #     b0_p = 1
        kwargs = {'index_map': self.index_map, 'b': b, 'dummy': dummy, 'dummy_subs': (
            dummy_s, dummy_f, dummy_p), 'b0_norms': (b0_s, b0_f, b0_p), 'monitor': monitor_convergence}

        parprint("---- [Solver] Solver set up in {}s".format(time() - t0_setup))

    def getIterationNumber(self):
        return self.solver.getIterationNumber()

    def solve(self, b, x):
        # No printing, handled by AbstractPhysics
        t0 = time()
        self.solver.solve(b, x)
        self.t_total += time() - t0

    def print_timings(self):
        parprint("\n===== Timing Solver: {:.3f}s".format(self.t_total))
