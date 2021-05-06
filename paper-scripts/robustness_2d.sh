# Test robustness of 2D solvers with respect to mesh size, so we use only direct solvers in the diagonal blocks.
# NOTE: Run from repo root.

ROOT_DIR=~/poroelasticity-linear-solvers
OUTDIR=~/poroelasticity-linear-solvers/output

SWELLING_2WAY=$OUTDIR/robustness-2d-swelling-2way.out
SWELLING_3WAY=$OUTDIR/robustness-2d-swelling-3way.out
FOOTING_2WAY=$OUTDIR/robustness-2d-footing-2way.out
FOOTING_3WAY=$OUTDIR/robustness-2d-footing-3way.out

# Reset output files
for FILE in $SWELLING_2WAY $SWELLING_3WAY $FOOTING_2WAY $FOOTING_3WAY; do
    echo "" > $FILE
done

# PETSc options files
PETSC_EXACT=$ROOT_DIR/petsc-options-exact

# First test exact solvers
# Swelling 
for N in 10 20 40 80 160 320; do
    echo "" | tee -a $SWELLING_2WAY
    mpirun -np 12 python swelling.py -N $N --petsc-options $PETSC_EXACT --pc-type "diagonal"| tee -a $SWELLING_2WAY
    echo "" | tee -a $SWELLING_3WAY
    mpirun -np 12 python swelling.py -N $N --petsc-options $PETSC_EXACT --pc-type "diagonal 3-way"| tee -a $SWELLING_3WAY
done
# Footing
for N in 10 20 40 80 160; do
    echo "" | tee -a $FOOTING_2WAY
    mpirun -np 12 python footing.py -N $N --petsc-options $PETSC_EXACT --pc-type "undrained"| tee -a $FOOTING_2WAY
    echo "" | tee -a $FOOTING_3WAY
    mpirun -np 12 python footing.py -N $N --petsc-options $PETSC_EXACT --pc-type "undrained 3-way"| tee -a $FOOTING_3WAY
done

# Now do inexact... it is exactly the same 
SWELLING_2WAY=$OUTDIR/robustness-2d-swelling-2way-inexact.out
SWELLING_3WAY=$OUTDIR/robustness-2d-swelling-3way-inexact.out
FOOTING_2WAY=$OUTDIR/robustness-2d-footing-2way-inexact.out
FOOTING_3WAY=$OUTDIR/robustness-2d-footing-3way-inexact.out

PETSC_INEXACT=$ROOT_DIR/petsc-options-inexact

# Swelling 
for N in 10 20 40 80 160 320; do
    echo "" | tee -a $SWELLING_2WAY
    mpirun -np 12 python swelling.py -N $N --petsc-options $PETSC_INEXACT --pc-type "diagonal"| tee -a $SWELLING_2WAY
    echo "" | tee -a $SWELLING_3WAY
    mpirun -np 12 python swelling.py -N $N --petsc-options $PETSC_INEXACT --pc-type "diagonal 3-way"| tee -a $SWELLING_3WAY
done
# Footing
for N in 10 20 40 80 160; do
    echo "" | tee -a $FOOTING_2WAY
    mpirun -np 12 python footing.py -N $N --petsc-options $PETSC_INEXACT --pc-type "undrained"| tee -a $FOOTING_2WAY
    echo "" | tee -a $FOOTING_3WAY
    mpirun -np 12 python footing.py -N $N --petsc-options $PETSC_INEXACT --pc-type "undrained 3-way"| tee -a $FOOTING_3WAY
done


