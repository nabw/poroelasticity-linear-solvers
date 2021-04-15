from petsc4py import PETSc
from lib.AndersonAcceleration import AndersonAcceleration


class PreconditionerCC(object):

    def __init__(self, M, M_diff, index_map, flag_3_way, inner_ksp_type="gmres", inner_pc_type="lu", inner_rtol=1e-6, inner_atol=1e-6, inner_maxiter=1000, inner_monitor=True, w1=1.0, w2=0.1, accel_order=0, bcs_sub_pressure=None):
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
        self.Mp_diff = self.M_diff.createSubMatrix(self.is_p, self.is_p)

        # Only diagonal blocks, used to create solvers
        self.matrices_elliptic = (self.Ms_s, self.Mf_f, self.Mp_p, self.Mp_diff)

    def create_solvers(self):
        self.ksp_s = PETSc.KSP().create()
        self.ksp_f = PETSc.KSP().create()
        self.ksp_p = PETSc.KSP().create()
        self.ksp_fp = PETSc.KSP().create()
        self.ksp_p_diff = PETSc.KSP().create()
        self.ksps_elliptic = (self.ksp_s, self.ksp_f, self.ksp_p, self.ksp_p_diff)

    def setup_elliptic_solver(self, solver, mat):
        solver.setOperators(mat, mat)
        solver.setType(self.inner_ksp_type)
        pc = solver.getPC()
        pc.setType(self.inner_pc_type)

        if self.inner_ksp_type != "preonly":
            solver.setTolerances(self.inner_rtol, self.inner_atol, 1e20, self.inner_maxiter)

        if self.inner_pc_type == "lu":
            factor_method = "mumps"  # Better scaling than the others
            pc.setFactorSolverType(factor_method)

        if self.inner_pc_type == "hypre":
            hypre_type = "boomeramg"  # in 2D only parasails works
            pc.setHYPREType(hypre_type)

        if self.inner_pc_type == "gamg":
            pc.setGAMGSmooths(1)

    def setup_fieldsplit(self, solver, mat):
        solver.setOperators(mat, mat)
        # Prefer GMRES for saddle point problem with asymmetric preconditioner
        solver.setType("gmres")
        pc = solver.getPC()
        pc.setType('fieldsplit')
        pc.setFieldSplitIS(("uf", self.is_f))
        pc.setFieldSplitIS(("p", self.is_p))
        PETSc.Options().setValue("-pc_fieldsplit_type", "schur")
        PETSc.Options().setValue("-pc_fieldsplit_ksp_type", self.inner_ksp_type)
        PETSc.Options().setValue("-pc_fieldsplit_ksp_atol", self.inner_atol)
        PETSc.Options().setValue("-pc_fieldsplit_ksp_rtol", self.inner_rtol)
        PETSc.Options().setValue("-pc_fieldsplit_ksp_maxiter", self.inner_maxiter)
        PETSc.Options().setValue("-pc_fieldsplit_pc_type", self.inner_pc_type)
        if self.inner_pc_type == "lu":
            PETSc.Options().setValue("-pc_fieldsplit_pc_factor_mat_solver_type", "mumps")
        if self.inner_pc_type == "hypre":
            hypre_type = "boomeramg"  # in 2D only parasails works
            PETSc.Options().setValue("-pc_fieldsplit_pc_hypre_type", hypre_type)
        if self.inner_pc_type == "gamg":
            # PETSc.Options().setValue("-pc_fieldsplit_pc_gamg_type", "agg")
            PETSc.Options().setValue("-pc_fieldsplit_pc_gamg_agg_nsmooths", 1)
            PETSc.Options().setValue("-pc_fieldsplit_pc_gamg_sym_graph", True)
        pc.setFromOptions()

    def setUp(self, pc):
        # create local ksp and pc contexts
        self.create_solvers()

        # Create temp block vectors used in apply()
        self.allocate_temp_vectors()

        # Extract sub-matrices
        self.allocate_submatrices()

        for solver, mat in zip(self.ksps_elliptic, self.matrices_elliptic):
            self.setup_elliptic_solver(solver, mat)

        self.setup_fieldsplit(self.ksp_fp, self.Mfp_fp)

    def apply(self, pc, x, y):
        # Result is y = A^{-1}x

        y.getSubVector(self.is_s, self.temp_sy)
        x.getSubVector(self.is_s, self.temp_sx)

        # TODO: use mult to avoid creating temp vectors for off-diagonal contributions
        if self.flag_3_way:

            # Extract subvectors
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

            # Solve both pressures first
            self.ksp_p.solve(self.temp_px, self.temp_py)
            # Apply bc to pressure rhs first
            self.temp_p_diffx.setValues(self.bcs_sub_pressure, self.bc_pressure)
            self.temp_p_diffx.assemble()
            self.ksp_p_diff.solve(self.temp_p_diffx, self.temp_p_diffy)

            # Then fluids
            # FS
            self.Mf_p.mult(self.temp_py, self.temp2_fx)
            self.temp2_fx.aypx(-1, self.temp_fx)
            self.ksp_f.solve(self.temp2_fx, self.temp_fy)
            # Diff
            self.Mf_p.mult(self.temp_p_diffy, self.temp2_fx)
            self.temp2_fx.aypx(-1, self.temp_fx)
            self.ksp_f.solve(self.temp2_fx, self.temp_f_diffy)

            # Finally solids
            # FS
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

            # Weighted CC sum
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
        else:  # use 2way
            self.ksp_s.solve(self.temp_sx, self.temp_sy)
            x.getSubVector(self.is_fp, self.temp_fpx)
            x.getSubVector(self.is_fp, self.temp2_fpx)
            y.getSubVector(self.is_fp, self.temp_fpy)

            # compute A_fp_s ys, ys resulting vector from before
            self.Mfp_s.mult(self.temp_sy, self.temp2_fpx)
            self.temp2_fpx.aypx(-1, self.temp_fpx)
            self.ksp_fp.solve(self.temp2_fpx, self.temp_fpy)
            x.restoreSubVector(self.is_fp, self.temp_fpx)
            y.restoreSubVector(self.is_fp, self.temp_fpy)

        x.restoreSubVector(self.is_s, self.temp_sx)
        y.restoreSubVector(self.is_s, self.temp_sy)

        self.anderson.get_next_vector(y)


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
        pc = PETSc.PC().create()
        pc.setType('python')
        pc.setPythonContext(ctx)
        pc.setOperators(self.A.mat())
        pc.setUp()
        return pc
