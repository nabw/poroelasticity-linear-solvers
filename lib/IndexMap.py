from petsc4py import PETSc


class IndexMap:
    def __init__(self, V):
        # Keep sub dimensions
        self.ns = V.sub(0).dim()
        self.nf = V.sub(1).dim()
        self.np = V.sub(2).dim()

        # Also dofmaps
        self.dofmap_s = V.sub(0).dofmap().dofs()
        self.dofmap_f = V.sub(1).dofmap().dofs()
        self.dofmap_p = V.sub(2).dofmap().dofs()
        self.dofmap_fp = sorted(self.dofmap_f + self.dofmap_p)

        # and Index Sets
        self.is_s = PETSc.IS().createGeneral(self.dofmap_s)
        self.is_f = PETSc.IS().createGeneral(self.dofmap_f)
        self.is_p = PETSc.IS().createGeneral(self.dofmap_p)
        self.is_fp = PETSc.IS().createGeneral(self.dofmap_fp)

    def get_dimensions(self):
        return self.ns, self.nf, self.np

    def get_index_sets(self):
        return self.is_s, self.is_f, self.is_p, self.is_fp
