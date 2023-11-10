/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include<LinearSystemSolver.h>

//-----------------------------------------------------

LinearSystemSolver::LinearSystemSolver(MPI_Comm &comm_, DM &dm_, PETScKSPOptionsData &ksp_input)
                  : comm(comm_), ctx(NULL), has_linear_operator(false)
{
  DMClone(dm_, &dm);
  KSPCreate(comm, &ksp);
  KSPSetDM(ksp, dm);
  KSPSetInitialGuessNonzero(ksp, true); //!< initial guess is passed to KSPSolve

  if(ksp_input.ksp == PETScKSPOptionsData::KSP_DEFAULT) {
    /* nothing to do */
  } else if(ksp_input.ksp == PETScKSPOptionsData::GMRES) {
    KSPSetType(ksp, KSPGMRES);
  } else if(ksp_input.ksp == PETScKSPOptionsData::FLEXIBLE_GMRES) {
    KSPSetType(ksp, KSPFGMRES);
  } else {
    print_error("*** Error: Detected unknown PETSc KSP type.\n");
    exit_mpi();
  }

  if(ksp_input.pc == PETScKSPOptionsData::PC_DEFAULT) {
    /* nothing to do*/
  } 
  else {
    PC* pc_ptr;
    KSPGetPC(ksp, pc_ptr);

    if(ksp_input.pc == PETScKSPOptionsData::PC_NONE)
      PCSetType(*pc_ptr, PCNONE);
    else if(ksp_input.pc == PETScKSPOptionsData::JACOBI)
      PCSetType(*pc_ptr, PCJACOBI);
    else if(ksp_input.pc == PETScKSPOptionsData::INCOMPLETE_LU)
      PCSetType(*pc_ptr, PCILU);
    else if(ksp_input.pc == PETScKSPOptionsData::INCOMPLETE_CHOLESKY)
      PCSetType(*pc_ptr, PCICC);
    else if(ksp_input.pc == PETScKSPOptionsData::MG_2LEVEL_EXOTIC)
      PCSetType(*pc_ptr, PCEXOTIC);
    else if(ksp_input.pc == PETScKSPOptionsData::MG)
      PCSetType(*pc_ptr, PCMG);
    else { 
      print_error("*** Error: Detected unknown PETSc KSP preconditioner type.\n");
      exit_mpi();
    }
  }

  KSPSetFromOptions(ksp); //overrides any options specified above
}

//-----------------------------------------------------

LinearSystemSolver::~LinearSystemSolver()
{ }

//-----------------------------------------------------

void
LinearSystemSolver::Destroy()
{
  DMDestroy(&dm);
  KSPDestroy(&ksp);
}

//-----------------------------------------------------

void
LinearSystemSolver::SetLinearOperator(void *)
{
I AM HERE
}

//-----------------------------------------------------

int
LinearSystemSolver::Solve(SpaceVariable3D &b, SpaceVariable3D &x)
{
  Vec &bb(b.GetRefToGlobalVec());
  Vec &xx(x.GetRefToGlobalVec());

  PetscErrorCode error_code = KSPSolve(ksp, bb, xx);

  return (int)error_code; //0 means no error
}


//-----------------------------------------------------

//-----------------------------------------------------
