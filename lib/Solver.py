from petsc4py import PETSc
from mpi4py import MPI
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
    _ksp.buildResidual(dummy)

    # Get residual subcomponents
    dummy.getSubVector(index_map.is_s, dummy_s)
    dummy.getSubVector(index_map.is_f, dummy_f)
    dummy.getSubVector(index_map.is_p, dummy_p)

    res_s_a = dummy_s.norm(PETSc.NormType.NORM_INFINITY)
    res_s_r = res_s_a/b0_s
    res_f_a = dummy_f.norm(PETSc.NormType.NORM_INFINITY)
    res_f_r = res_f_a/b0_f
    res_p_a = dummy_p.norm(PETSc.NormType.NORM_INFINITY)
    res_p_r = res_p_a/b0_p

    error_abs = max(res_s_a, res_f_a, res_p_a)
    error_rel = max(res_s_r, res_f_r, res_p_r)

    if kwargs['monitor']:
        width = 11
        if _it == 0 and MPI.COMM_WORLD.rank == 0:
            print("KSP errors: {}, {}, {}, {}, {}, {}".format('abs_s'.rjust(width), 'abs_f'.rjust(
                width), 'abs_p'.rjust(width), 'rel_s'.rjust(width), 'rel_f'.rjust(width), 'rel_p'.rjust(width)), flush=True)
        if MPI.COMM_WORLD.rank == 0:
            print("KSP it {}:   {:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}, {:.5e}".format(
                _it, res_s_a, res_f_a, res_p_a, res_s_r, res_f_r, res_p_r), flush=True)
    if error_abs < _ksp.atol or error_rel < _ksp.rtol:
        # Convergence
        if MPI.COMM_WORLD.rank == 0:
            print("KSP converged", flush=True)
        return 1
    elif _it > _ksp.max_it or error_abs > _ksp.divtol:
        # Divergence
        return -1
    else:
        # Continue
        return 0


class Solver:
    def __init__(self, A, b, PC, parameters, index_map):
        t0_solver = time()
        self.A = A
        self.b = b
        self.PC = PC
        self.parameters = parameters
        self.index_map = index_map
        self.set_up()
        if MPI.COMM_WORLD.rank == 0:
            print("---- Solver set up time = {}s".format(time() - t0_solver), flush=True)

    def set_up(self):
        # Create linear solver
        solver_type = self.parameters["solver type"]
        atol = self.parameters["solver atol"]
        rtol = self.parameters["solver rtol"]
        maxiter = self.parameters["solver maxiter"]
        monitor_convergence = self.parameters["solver monitor"]

        # Prepare elements for convergence test:
        args = None
        #index_map, b, dummy, dummy_s, dummy_f, dummy_p, b0_s, b0_f, b0_p
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
        if b0_s < 1e-10:
            b0_s = 1
        if b0_f < 1e-10:
            b0_f = 1
        if b0_p < 1e-10:
            b0_p = 1
        kwargs = {'index_map': self.index_map, 'b': b, 'dummy': dummy, 'dummy_subs': (
            dummy_s, dummy_f, dummy_p), 'b0_norms': (b0_s, b0_f, b0_p), 'monitor': monitor_convergence}
        if solver_type == "aar":
            from lib.AAR import AAR
            order = self.parameters["AAR order"]
            p = self.parameters["AAR p"]
            omega = self.parameters["AAR omega"]
            beta = self.parameters["AAR beta"]
            self.solver = AAR(order, p, omega, beta, self.A.mat(), x0=None, pc=self.PC, atol=atol,
                              rtol=rtol, maxiter=maxiter, monitor_convergence=monitor_convergence)
        else:
            solver = PETSc.KSP().create()
            solver.setOperators(self.A.mat())
            solver.setType(solver_type)
            solver.setTolerances(rtol, atol, 1e20, maxiter)
            solver.setPC(self.PC)
            if solver_type == "gmres":
                solver.setGMRESRestart(maxiter)
            # if monitor_convergence:
                # PETSc.Options().setValue("-ksp_monitor", None)

            solver.setConvergenceTest(converged, args, kwargs)

            solver.setFromOptions()
            self.solver = solver

    def get_solver(self):
        return self.solver
