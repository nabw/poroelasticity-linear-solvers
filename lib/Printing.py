from mpi4py import MPI


def parprint(*args):
    if MPI.COMM_WORLD.rank == 0:
        print(*args, flush=True)
