from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from itertools import chain
from time import perf_counter as time
from lib.Printing import parprint
from numba import jit


@jit(nopython=True, cache=True)
def get_local_fp_dofs(dofs_fp_global, dofmap_f, dofmap_p):
    # Find the corresponding local indexex in f-p subspace
    nf = len(dofmap_f)
    np = len(dofmap_p)
    dofs_f = [0] * nf
    dofs_p = [0] * np
    i_f = i_p = i = 0
    for dof in dofs_fp_global:
        if dof in dofmap_f:
            dofs_f[i_f] = i
            i_f += 1
        elif dof in dofmap_p:
            dofs_p[i_p] = i
            i_p += 1
        i += 1
    return dofs_f, dofs_p


class IndexSet:
    def __init__(self, V, two_way=True):
        t0 = time()
        # Keep sub dimensions
        self.ns = V.sub(0).dim()
        self.nf = V.sub(1).dim()
        self.np = V.sub(2).dim()

        # Also dofmaps
        self.dofmap_s = V.sub(0).dofmap().dofs()
        self.dofmap_f = V.sub(1).dofmap().dofs()
        self.dofmap_p = V.sub(2).dofmap().dofs()
        self.dofmap_fp = sorted(self.dofmap_f + self.dofmap_p)
        # Note that f and p dofmaps are used for the fieldsplit Preconditioner
        # in the 2-way splittings, so they are still useful but bear a different meaning.

        # All gather the global fp dofmap
        if two_way:
            comm = MPI.COMM_WORLD
            dofs_fp_global = self.dofmap_fp.copy()
            dofs_fp_global = comm.allgather(dofs_fp_global)
            dofs_fp_global = np.array(tuple(chain(*dofs_fp_global)))  # Concat and sort

            # Replace global dofmaps with f-p dofmaps.
            self.dofmap_f, self.dofmap_p = get_local_fp_dofs(
                dofs_fp_global, self.dofmap_f, self.dofmap_p)

        # and Index Sets
        self.is_s = PETSc.IS().createGeneral(self.dofmap_s)
        self.is_f = PETSc.IS().createGeneral(self.dofmap_f)
        self.is_p = PETSc.IS().createGeneral(self.dofmap_p)
        self.is_fp = PETSc.IS().createGeneral(self.dofmap_fp)
        parprint("---- [Indexes] computed local indices in {:.3f}s".format(time() - t0))

    def get_dimensions(self):
        return self.ns, self.nf, self.np

    def get_index_sets(self):
        return self.is_s, self.is_f, self.is_p, self.is_fp
