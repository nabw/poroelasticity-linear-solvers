from petsc4py import PETSc
from lib.AndersonAcceleration import AndersonAcceleration


class PreconditionerCC(object):

    def __init__(self, M, M_diff, V, flag_3_way, inner_ksp_type="gmres", inner_pc_type="lu", inner_rtol=1e-6, inner_atol=1e-6, inner_maxiter=1000, inner_monitor=True, w1=1.0, w2=0.1, accel_order=0, bcs_sub_pressure=None):
        import numpy as np
        self.M = M
        self.M_diff = M_diff
        self.flag_3_way = flag_3_way
        self.w1 = w1
        self.w2 = w2
        self.ns = V.sub(0).dim()
        self.nf = V.sub(1).dim()
        self.np = V.sub(2).dim()
        self.dofmap_s = V.sub(0).dofmap().dofs()
        self.dofmap_f = V.sub(1).dofmap().dofs()
        self.dofmap_p = V.sub(2).dofmap().dofs()
        self.dofmap_fp = sorted(self.dofmap_f + self.dofmap_p)
        # Create index sets for each physics
        self.is_s = PETSc.IS().createGeneral(self.dofmap_s)
        self.is_f = PETSc.IS().createGeneral(self.dofmap_f)
        self.is_p = PETSc.IS().createGeneral(self.dofmap_p)
        self.is_fp = PETSc.IS().createGeneral(self.dofmap_fp)

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
        self.temp_sy = PETSc.Vec().create()
        self.temp_fx = PETSc.Vec().create()
        self.temp_fy = PETSc.Vec().create()
        self.temp_px = PETSc.Vec().create()
        self.temp_py = PETSc.Vec().create()
        self.temp_fpx = PETSc.Vec().create()
        self.temp_fpy = PETSc.Vec().create()
        # Counter parts for CC solver
        self.temp_s_diffx = PETSc.Vec().create()
        self.temp_s_diffy = PETSc.Vec().create()
        self.temp_f_diffx = PETSc.Vec().create()
        self.temp_f_diffy = PETSc.Vec().create()
        self.temp_p_diffx = PETSc.Vec().create()
        self.temp_p_diffy = PETSc.Vec().create()

    def allocate_submatrices(self):
        self.Ms_s = self.M.createSubMatrix(self.is_s, self.is_s)
        self.Ms_f = self.M.createSubMatrix(self.is_s, self.is_f)
        self.Ms_p = self.M.createSubMatrix(self.is_s, self.is_p)
        self.Mf_s = self.M.createSubMatrix(self.is_f, self.is_s)
        self.Mf_f = self.M.createSubMatrix(self.is_f, self.is_f)
        self.Mf_p = self.M.createSubMatrix(self.is_f, self.is_p)
        self.Mp_s = self.M.createSubMatrix(self.is_p, self.is_s)
        self.Mp_f = self.M.createSubMatrix(self.is_p, self.is_f)
        self.Mp_p = self.M.createSubMatrix(self.is_p, self.is_p)
        self.Mfp_s = self.M.createSubMatrix(self.is_fp, self.is_s)
        self.Mfp_fp = self.M.createSubMatrix(self.is_fp, self.is_fp)
        self.Mp_diff = self.M_diff.createSubMatrix(self.is_p, self.is_p)

        # Only diagonal blocks, used to create solvers
        self.matrices = (self.Ms_s, self.Mf_f, self.Mp_p, self.Mfp_fp, self.Mp_diff)

    def create_solvers(self):
        self.ksp_s = PETSc.KSP().create()
        self.ksp_f = PETSc.KSP().create()
        self.ksp_p = PETSc.KSP().create()
        self.ksp_fp = PETSc.KSP().create()
        self.ksp_p_diff = PETSc.KSP().create()
        self.ksps = (self.ksp_s, self.ksp_f, self.ksp_p, self.ksp_fp, self.ksp_p_diff)

        # self.pc_s = self.ksp_s.getPC()
        # self.pc_f = self.ksp_f.getPC()
        # self.pc_p = self.ksp_p.getPC()
        # self.pc_fp = self.ksp_fp.getPC()
        # self.pc_p_diff = self.ksp_p_diff.getPC()
        # self.pcs = (self.pc_s, self.pc_f, self.pc_p, self.pc_fp, self.pc_p_diff)

    def setup_solver(self, solver, mat):
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
            hypre_type = "parasails"  # only parasails wokrs
            pc.setHYPREType(hypre_type)

        if self.inner_pc_type == "gamg":
            pc.setGAMGSmooths(2)

    def setUp(self, pc):
        # create local ksp and pc contexts
        self.create_solvers()

        # Create temp block vectors used in apply()
        self.allocate_temp_vectors()

        # Extract sub-matrices
        self.allocate_submatrices()

        for solver, mat in zip(self.ksps, self.matrices):
            self.setup_solver(solver, mat)

    def apply(self, pc, x, y):
        # Result is y = A^{-1}x

        x.getSubVector(self.is_s, self.temp_sx)
        y.getSubVector(self.is_s, self.temp_sy)

        # TODO: use mmult to avoid creating temp vectors for off-diagonal contributions
        if self.flag_3_way:

            # Extract subvectors
            x.getSubVector(self.is_f, self.temp_fx)
            y.getSubVector(self.is_f, self.temp_fy)
            x.getSubVector(self.is_f, self.temp_f_diffx)
            x.getSubVector(self.is_s, self.temp_s_diffx)
            y.getSubVector(self.is_s, self.temp_s_diffy)
            y.getSubVector(self.is_f, self.temp_f_diffy)
            x.getSubVector(self.is_p, self.temp_px)
            y.getSubVector(self.is_p, self.temp_py)
            x.getSubVector(self.is_p, self.temp_p_diffx)
            y.getSubVector(self.is_p, self.temp_p_diffy)

            # Solve both pressures first
            self.ksp_p.solve(self.temp_px, self.temp_py)
            # Apply bc to pressure rhs first
            self.temp_p_diffx.setValues(self.bcs_sub_pressure, self.bc_pressure)
            self.temp_p_diffx.assemble()
            self.ksp_p_diff.solve(self.temp_p_diffx, self.temp_p_diffy)

            # Then fluids
            self.ksp_f.solve(self.temp_fx - self.Mf_p * self.temp_py, self.temp_fy)
            self.ksp_f.solve(self.temp_fx - self.Mf_p * self.temp_p_diffy, self.temp_f_diffy)

            # Finally solids
            self.ksp_s.solve(self.temp_sx - self.Ms_p * self.temp_py -
                             self.Ms_f * self.temp_fy, self.temp_sy)
            self.ksp_s.solve(self.temp_sx - self.Ms_p * self.temp_p_diffy -
                             self.Ms_f * self.temp_f_diffy, self.temp_s_diffy)

            # Weighted CC sum
            self.temp_py.scale(self.w1)
            self.temp_fy.scale(self.w1)
            self.temp_sy.scale(self.w1)
            self.temp_py.axpy(self.w2, self.temp_p_diffy)
            self.temp_fy.axpy(self.w2, self.temp_f_diffy)
            self.temp_sy.axpy(self.w2, self.temp_s_diffy)

            x.restoreSubVector(self.is_f, self.temp_fx)
            y.restoreSubVector(self.is_p, self.temp_py)
            x.restoreSubVector(self.is_p, self.temp_px)
            y.restoreSubVector(self.is_f, self.temp_fy)
        else:  # use 2way
            x.getSubVector(self.is_s, self.temp_sx)
            y.getSubVector(self.is_s, self.temp_sy)
            self.ksp_s.solve(self.temp_sx, self.temp_sy)
            x.getSubVector(self.is_fp, self.temp_fpx)
            y.getSubVector(self.is_fp, self.temp_fpy)

            # compute A_fp_s ys, ys resulting vector from before
            self.ksp_fp.solve(self.temp_fpx - self.Mfp_s * self.temp_sy, self.temp_fpy)
            x.restoreSubVector(self.is_fp, self.temp_fpx)
            y.restoreSubVector(self.is_fp, self.temp_fpy)

        x.restoreSubVector(self.is_s, self.temp_sx)
        y.restoreSubVector(self.is_s, self.temp_sy)

        self.anderson.get_next_vector(y)


class Preconditioner:
    def __init__(self, V, A, P, P_diff, pc_type, inner_ksp_type, inner_pc_type, inner_rtol, inner_atol, inner_maxiter, inner_accel_order, inner_monitor, bcs_sub_pressure):
        self.V = V
        self.A = A
        self.P = P
        self.P_diff = P_diff
        self.pc_type = pc_type
        self.inner_ksp_type = inner_ksp_type
        self.inner_pc_type = inner_pc_type
        self.inner_maxiter = inner_maxiter
        self.inner_rtol = inner_rtol
        self.inner_atol = inner_atol
        self.inner_accel_order = inner_accel_order
        self.inner_monitor = inner_monitor
        self.bcs_sub_pressure = bcs_sub_pressure
        if pc_type not in ("undrained", "diagonal", "diagonal 3-way", "lu"):
            import sys
            sys.exit("pc type must be one of lu, undrained, diagonal, diagonal 3-way.")

    def get_pc(self):
        flag_3_way = self.pc_type == "diagonal 3-way"
        ctx = PreconditionerCC(self.P.mat(), self.P_diff.mat(), self.V, flag_3_way, self.inner_ksp_type,
                               self.inner_pc_type, self.inner_rtol, self.inner_atol, self.inner_maxiter, self.inner_monitor, 1.0, 0.1, self.inner_accel_order, self.bcs_sub_pressure)
        pc = PETSc.PC().create()
        pc.setType('python')
        pc.setPythonContext(ctx)
        pc.setOperators(self.A.mat())
        pc.setUp()
        return pc
