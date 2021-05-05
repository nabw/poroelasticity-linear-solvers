from petsc4py import PETSc
from lib.AndersonAcceleration import AndersonAcceleration
from lib.Printing import parprint
from time import perf_counter as time


class PreconditionerCC(object):

    def __init__(self, M, M_diff, index_map, flag_3_way, inner_ksp_type="gmres", inner_pc_type="lu", inner_rtol=1e-6, inner_atol=1e-6, inner_maxiter=1000, inner_monitor=True, w1=1.0, w2=0.1, accel_order=0, bcs_sub_pressure=None):
        t0_init = time()
        import numpy as np
        self.M = M
        self.M_diff = M_diff
        self.flag_3_way = flag_3_way
        self.w1 = w1
        self.w2 = w2
        self.ns, self.nf, self.np = index_map.get_dimensions()

        # Create index sets for each physics
        self.is_s, self.is_f, self.is_p, self.is_fp = index_map.get_index_sets()

        self.inner_ksp_type = inner_ksp_type
        self.inner_pc_type = inner_pc_type
        self.inner_maxiter = inner_maxiter
        self.inner_rtol = inner_rtol
        self.inner_atol = inner_atol
        self.inner_monitor = inner_monitor
        self.anderson = AndersonAcceleration(accel_order)

        # Used to set pressure bcs on rhs during 3-way
        self.bcs_sub_pressure = bcs_sub_pressure
        self.bc_pressure = np.zeros(len(bcs_sub_pressure))

        # Setup counters
        self.t_solid = 0
        self.t_fluid = 0
        self.t_press = 0
        self.t_total = 0
        self.t_alloc = 0

    def allocate_temp_vectors(self):
        self.temp_sx = PETSc.Vec().create()
        self.temp2_sx = PETSc.Vec().create()
        self.temp3_sx = PETSc.Vec().create()
        self.temp_sy = PETSc.Vec().create()
        self.temp_fx = PETSc.Vec().create()
        self.temp2_fx = PETSc.Vec().create()
        self.temp_fy = PETSc.Vec().create()
        self.temp_px = PETSc.Vec().create()
        self.temp_py = PETSc.Vec().create()
        self.temp_fpx = PETSc.Vec().create()
        self.temp2_fpx = PETSc.Vec().create()
        self.temp_fpy = PETSc.Vec().create()
        # Counter parts for CC solver
        self.temp_s_diffy = PETSc.Vec().create()
        self.temp_f_diffy = PETSc.Vec().create()
        self.temp_p_diffx = PETSc.Vec().create()
        self.temp_p_diffy = PETSc.Vec().create()

    def allocate_submatrices(self):
        self.Ms_s = self.M.createSubMatrix(self.is_s, self.is_s)
        self.Ms_f = self.M.createSubMatrix(self.is_s, self.is_f)
        self.Ms_p = self.M.createSubMatrix(self.is_s, self.is_p)
        self.Mf_f = self.M.createSubMatrix(self.is_f, self.is_f)
        self.Mf_p = self.M.createSubMatrix(self.is_f, self.is_p)
        self.Mp_p = self.M.createSubMatrix(self.is_p, self.is_p)
        self.Mfp_s = self.M.createSubMatrix(self.is_fp, self.is_s)
        self.Mfp_fp = self.M.createSubMatrix(self.is_fp, self.is_fp)

        # Only diagonal blocks, used to create solvers
        self.matrices_elliptic = [self.Ms_s, self.Mf_f, self.Mp_p]

        if self.flag_3_way:
            self.Mp_diff = self.M_diff.createSubMatrix(self.is_p, self.is_p)
            self.matrices_elliptic.append(self.Mp_diff)

    def create_solvers(self):
        self.ksp_s = PETSc.KSP().create()
        self.ksp_s.setOptionsPrefix("s_")
        self.ksp_f = PETSc.KSP().create()
        self.ksp_f.setOptionsPrefix("f_")
        self.ksp_p = PETSc.KSP().create()
        self.ksp_p.setOptionsPrefix("p_")
        self.ksp_fp = PETSc.KSP().create()
        self.ksp_fp.setOptionsPrefix("fp_")

        self.ksps_elliptic = [self.ksp_s, self.ksp_f, self.ksp_p]

        if self.flag_3_way:
            self.ksp_p_diff = PETSc.KSP().create()
            self.ksp_p_diff.setOptionsPrefix("diff_")
            self.ksps_elliptic.append(self.ksp_p_diff)

    def setup_elliptic_solver(self, solver, mat):
        solver.setOperators(mat, mat)
        solver.setType(self.inner_ksp_type)
        pc = solver.getPC()
        pc.setType(self.inner_pc_type)
        pc.setFromOptions()
        solver.setFromOptions()

    def setup_fieldsplit(self, solver, mat):
        solver.setOperators(mat, mat)
        # Prefer GMRES for saddle point problem with asymmetric preconditioner
        solver.setType("gmres")
        solver.setInitialGuessNonzero(True)

        pc = solver.getPC()
        pc.setType('fieldsplit')
        pc.setFieldSplitIS((None, self.is_p))
        pc.setFieldSplitIS((None, self.is_f))

        solver.setFromOptions()
        pc.setFromOptions()
        pc.setUp()  # Must be called after set from options
        ksps = pc.getFieldSplitSubKSP()
        for ksp in ksps:
            ksp.setFromOptions()

    def setUp(self, pc):
        t0_setup = time()
        # create local ksp and pc contexts
        self.create_solvers()

        # Create temp block vectors used in apply()
        self.allocate_temp_vectors()

        # Extract sub-matrices
        self.allocate_submatrices()

        for solver, mat in zip(self.ksps_elliptic, self.matrices_elliptic):
            self.setup_elliptic_solver(solver, mat)

        if not self.flag_3_way:
            if self.inner_pc_type == "lu":
                self.setup_elliptic_solver(self.ksp_fp, self.Mfp_fp)
            else:
                self.setup_fieldsplit(self.ksp_fp, self.Mfp_fp)
        parprint("---- [Preconditioner] Set up in {}s".format(time() - t0_setup))

    def apply(self, pc, x, y):
        t_total = time()
        # Result is y = A^{-1}x

        t0_alloc = time()
        y.getSubVector(self.is_s, self.temp_sy)
        x.getSubVector(self.is_s, self.temp_sx)
        self.t_alloc += time() - t0_alloc

        if self.flag_3_way:

            # Extract subvectors
            t0_alloc = time()
            x.getSubVector(self.is_s, self.temp2_sx)
            x.getSubVector(self.is_s, self.temp3_sx)
            x.getSubVector(self.is_f, self.temp_fx)
            x.getSubVector(self.is_f, self.temp2_fx)
            x.getSubVector(self.is_p, self.temp_px)
            x.getSubVector(self.is_p, self.temp_p_diffx)

            y.getSubVector(self.is_s, self.temp_s_diffy)
            y.getSubVector(self.is_f, self.temp_fy)
            y.getSubVector(self.is_f, self.temp_f_diffy)
            y.getSubVector(self.is_p, self.temp_py)
            y.getSubVector(self.is_p, self.temp_p_diffy)
            self.t_alloc += time() - t0_alloc

            # Solve both pressures first
            t_p = time()
            self.ksp_p.solve(self.temp_px, self.temp_py)
            # Apply bc to pressure rhs first
            self.temp_p_diffx.setValues(self.bcs_sub_pressure, self.bc_pressure)
            self.temp_p_diffx.assemble()
            self.ksp_p_diff.solve(self.temp_p_diffx, self.temp_p_diffy)
            self.t_press += time() - t_p

            # Then fluids
            # FS
            t_f = time()
            self.Mf_p.mult(self.temp_py, self.temp2_fx)
            self.temp2_fx.aypx(-1, self.temp_fx)
            self.ksp_f.solve(self.temp2_fx, self.temp_fy)
            # Diff
            self.Mf_p.mult(self.temp_p_diffy, self.temp2_fx)
            self.temp2_fx.aypx(-1, self.temp_fx)
            self.ksp_f.solve(self.temp2_fx, self.temp_f_diffy)
            self.t_fluid += time() - t_f

            # Finally solids
            # FS
            t_s = time()
            self.Ms_f.mult(self.temp_fy, self.temp2_sx)
            self.Ms_p.mult(self.temp_py, self.temp3_sx)
            self.temp2_sx.axpy(1, self.temp3_sx)  # temp2 + temp3
            self.temp2_sx.aypx(-1, self.temp_sx)
            self.ksp_s.solve(self.temp2_sx, self.temp_sy)
            # Diff
            self.Ms_f.mult(self.temp_f_diffy, self.temp2_sx)
            self.Ms_p.mult(self.temp_p_diffy, self.temp3_sx)
            self.temp2_sx.axpy(1, self.temp3_sx)  # temp2 + temp3
            self.temp2_sx.aypx(-1, self.temp_sx)
            self.ksp_s.solve(self.temp2_sx, self.temp_s_diffy)
            self.t_solid += time() - t_s

            # Weighted CC sum
            t0_alloc = time()
            self.temp_py.scale(self.w1)
            self.temp_fy.scale(self.w1)
            self.temp_sy.scale(self.w1)
            self.temp_py.axpy(self.w2, self.temp_p_diffy)
            self.temp_fy.axpy(self.w2, self.temp_f_diffy)
            self.temp_sy.axpy(self.w2, self.temp_s_diffy)

            x.restoreSubVector(self.is_f, self.temp_fx)
            x.restoreSubVector(self.is_p, self.temp_px)
            y.restoreSubVector(self.is_f, self.temp_fy)
            y.restoreSubVector(self.is_p, self.temp_py)
            self.t_alloc += time() - t0_alloc
        else:  # use 2way
            t_s = time()
            self.ksp_s.solve(self.temp_sx, self.temp_sy)
            self.t_solid += time() - t_s

            t0_alloc = time()
            x.getSubVector(self.is_fp, self.temp_fpx)
            x.getSubVector(self.is_fp, self.temp2_fpx)
            y.getSubVector(self.is_fp, self.temp_fpy)
            self.t_alloc += time() - t0_alloc

            # compute A_fp_s ys, ys resulting vector from before
            t_fp = time()
            self.Mfp_s.mult(self.temp_sy, self.temp2_fpx)
            self.temp2_fpx.aypx(-1, self.temp_fpx)
            self.ksp_fp.solve(self.temp2_fpx, self.temp_fpy)
            self.t_fluid += time() - t_fp
            self.t_press += time() - t_fp

            t0_alloc = time()
            x.restoreSubVector(self.is_fp, self.temp_fpx)
            y.restoreSubVector(self.is_fp, self.temp_fpy)
            self.t_alloc += time() - t0_alloc

        t0_alloc = time()
        x.restoreSubVector(self.is_s, self.temp_sx)
        y.restoreSubVector(self.is_s, self.temp_sy)
        self.t_alloc += time() - t0_alloc

        if self.anderson.order > 0:
            self.anderson.get_next_vector(y)
        self.t_total += time() - t_total

    def print_timings(self):
        parprint("\n===== Timing preconditioner: {:.3f}s".format(self.t_total))
        if self.flag_3_way:
            parprint("\tSolid solver: {:.3f}s\n\tFluid solver: {:.3f}s\n\tPressure solver: {:.3f}s".format(
                self.t_solid, self.t_fluid, self.t_press))
        else:
            parprint(
                "\tSolid solver: {:.3f}s\n\tFluid-pressure solver: {:.3f}s".format(self.t_solid, self.t_fluid))
        parprint("\n\tAllocation time: {:.3f}".format(self.t_alloc))


class Preconditioner:
    def __init__(self, index_map, A, P, P_diff, parameters, bcs_sub_pressure):
        self.index_map = index_map
        self.A = A
        self.P = P
        self.P_diff = P_diff
        self.pc_type = parameters["pc type"]
        self.inner_ksp_type = parameters["inner ksp type"]
        self.inner_pc_type = parameters["inner pc type"]
        self.inner_rtol = parameters["inner rtol"]
        self.inner_atol = parameters["inner atol"]
        self.inner_maxiter = parameters["inner maxiter"]
        self.inner_accel_order = parameters["inner accel order"]
        self.inner_monitor = parameters["inner monitor"]
        self.bcs_sub_pressure = bcs_sub_pressure
        if self.pc_type not in ("undrained", "diagonal", "diagonal 3-way", "lu"):
            import sys
            sys.exit("pc type must be one of lu, undrained, diagonal, diagonal 3-way.")

    def get_pc(self):
        flag_3_way = self.pc_type == "diagonal 3-way"
        ctx = PreconditionerCC(self.P.mat(), self.P_diff.mat(), self.index_map, flag_3_way, self.inner_ksp_type,
                               self.inner_pc_type, self.inner_rtol, self.inner_atol, self.inner_maxiter, self.inner_monitor, 1.0, 0.1, self.inner_accel_order, self.bcs_sub_pressure)
        self.pc = PETSc.PC().create()
        self.pc.setType('python')
        self.pc.setPythonContext(ctx)
        self.pc.setOperators(self.A.mat())
        self.pc.setUp()
        return self.pc

    def print_timings(self):
        ctx = self.pc.getPythonContext()
        ctx.print_timings()
