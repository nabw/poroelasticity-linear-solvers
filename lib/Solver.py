from petsc4py import PETSc
from mpi4py import MPI


def converged(_ksp, _it, _rnorm, *args, **kwargs):
    """
    args must have: index_map, b, dummy, dummy_s, dummy_f, dummy_p, b0_s, b0_f, b0_p.
    dummy is used to avoid allocation of new vector for residual. [is is somewhere in PETSc...?]
    """
    _A, __ = _ksp.getOperators()
    _sol = _ksp.getSolution()
    _A.mult(_sol, dummy)
    dummy.axpy(-1, _ksp.getRhs())

    # Get residual subcomponents
    dummy.getSubVector(index_map.is_s, dummy_s)
    dummy.getSubVector(index_map.is_f, dummy_f)
    dummy.getSubVector(index_map.is_p, dummy_p)

    res_s_a = dummy_s.norm()
    res_s_r = dummy_s.norm()/b0_s
    res_f_a = dummy_f.norm()
    res_f_r = dummy_f.norm()/b0_f
    res_p_a = dummy_p.norm()
    res_p_r = dummy_p.norm()/b0_p

    error_abs = max(res_s_a, res_f_a, res_p_a)
    error_rel = max(res_s_r, res_f_r, res_p_r)

    if kwargs['monitor']:
        if _it == 0 and MPI.COMM_WORLD.rank == 0:
            print("KSP errors: abs_s, abs_f, abs_p, rel_s, rel_f, rel_p")
        if MPI.COMM_WORLD.rank == 0:
            print("KSP it {}: {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}".format(
                _it, res_s_a, res_f_a, res_p_a, res_s_r, res_f_r, res_p_r))
    if erros_abs < ksp.atol or error_rel < ksp.rtol:
        # Convergence
        return 1
    elif _it > ksp.max_it or error_abs > ksp.divtol:
        # Divergence
        return -1
    else:
        return 0


class Solver:
    def __init__(self, A, PC, parameters, index_map):
        self.A = A
        self.PC = PC
        self.parameters = parameters
        self.index_map = index_map

    def set_up(self, b, index_map):
        # Create linear solver
        solver_type = self.parameters["solver type"]
        atol = self.parameters["solver atol"]
        rtol = self.parameters["solver rtol"]
        maxiter = self.parameters["solver maxiter"]
        monitor_convergence = self.parameters["solver monitor"]

        # Prepare elements for convergence test:
        args = None
        #index_map, b, dummy, dummy_s, dummy_f, dummy_p, b0_s, b0_f, b0_p
        dummy = b.copy()
        b_s = PETSc.Vec().create()
        b_f = PETSc.Vec().create()
        b_p = PETSc.Vec().create()
        b.getSubVector(index_map.is_s, b_s)
        dummy_s = b_s.copy()
        b.getSubVector(index_map.is_f, b_f)
        dummy_f = b_f.copy()
        b.getSubVector(index_map.is_p, b_p)
        dummy_p = b_p.copy()
        b0_s = b_s.norm()
        b0_f = b_f.norm()
        b0_p = b_p.norm()
        b.getRestoreVector(index_map.is_s, b_s)
        b.getRestoreVector(index_map.is_f, b_f)
        b.getRestoreVector(index_map.is_p, b_p)
        kwargs = {'index_map': index_map, 'b': b, 'dummy': dummy, 'dummy_subs': (
            dummy_s, dummy_f, dummy_p), 'b0_norms': (b0_s, b0_f, b0_p)}  # TODO
        if solver_type == "aar":
            from lib.AAR import AAR
            order = self.parameters["AAR order"]
            p = self.parameters["AAR p"]
            omega = self.parameters["AAR omega"]
            beta = self.parameters["AAR beta"]
            self.solver = AAR(order, p, omega, beta, A.mat(), x0=None, pc=pc, atol=atol,
                              rtol=rtol, maxiter=maxiter, monitor_convergence=monitor_convergence)
        else:
            solver = PETSc.KSP().create()
            solver.setOperators(A.mat())
            solver.setType(solver_type)
            solver.setTolerances(rtol, atol, 1e20, maxiter)
            solver.setPC(pc)
            if solver_type == "gmres":
                solver.setGMRESRestart(maxiter)
            if monitor_convergence:
                PETSc.Options().setValue("-ksp_monitor", None)

            solver.setConvergenceTest(converged, args, kwargs)

            solver.setFromOptions()
            self.solver = solver

    def get_solver(self):
        return self.solver
