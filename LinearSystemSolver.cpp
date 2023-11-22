/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include<LinearSystemSolver.h>
#include<cassert>

//-----------------------------------------------------

LinearSystemSolver::LinearSystemSolver(MPI_Comm &comm_, DM &dm_, PETScKSPOptionsData &ksp_input)
                  : LinearOperator(comm_, dm_)
{

  KSPCreate(comm, &ksp);
  //KSPSetDM(ksp, dm);
  KSPSetInitialGuessNonzero(ksp, PETSC_TRUE); //!< initial guess is passed to KSPSolve

  SetTolerances(ksp_input);

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
    PC* pc_ptr(NULL);
    KSPGetPC(ksp, pc_ptr);
    assert(pc_ptr);

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

  if(strcmp(ksp_input.options_file, ""))
    PetscOptionsInsert(NULL, NULL, NULL, ksp_input.options_file);

  KSPSetFromOptions(ksp); //overrides any options specified above
}

//-----------------------------------------------------

LinearSystemSolver::~LinearSystemSolver()
{ }

//-----------------------------------------------------

void
LinearSystemSolver::Destroy()
{
  KSPDestroy(&ksp);
  LinearOperator::Destroy();
}

//-----------------------------------------------------

void
LinearSystemSolver::SetTolerances(PETScKSPOptionsData &ksp_input) 
{
  double relative_error = ksp_input.rtol;
  double absolute_error = ksp_input.abstol;
  double divergence_tol = ksp_input.dtol;
  int    max_iterations = ksp_input.maxits;

  KSPSetTolerances(ksp,
                   relative_error>0 ? relative_error : PETSC_DEFAULT,
                   absolute_error>0 ? absolute_error : PETSC_DEFAULT,
                   divergence_tol>0 ? divergence_tol : PETSC_DEFAULT,
                   max_iterations>0 ? max_iterations : PETSC_DEFAULT);
}

//-----------------------------------------------------

void
LinearSystemSolver::GetTolerances(double *rtol, double *abstol, double *dtol, int *maxits)
{
  KSPGetTolerances(ksp, rtol, abstol, dtol, maxits);
}

//-----------------------------------------------------

void
LinearSystemSolver::SetLinearOperator(vector<RowEntries>& row_entries)
{
  LinearOperator::SetLinearOperator(row_entries); //build A
  KSPSetOperators(ksp, A, A);
}

//-----------------------------------------------------

int
LinearSystemSolver::Solve(SpaceVariable3D &b, SpaceVariable3D &x)
{
  // --------------------------------------------------
  // Sanity checks
  int dof_ = b.NumDOF();
  assert(dof_ == dof);
  dof_ = x.NumDOF();
  assert(dof_ == dof);

  int i0_, j0_, k0_, imax_, jmax_, kmax_;
  b.GetCornerIndices(&i0_, &j0_, &k0_, &imax_, &jmax_, &kmax_);
  assert(i0_==i0 && j0_==j0 && k0_==k0 && imax_==imax && jmax_==jmax && kmax_==kmax);
  x.GetCornerIndices(&i0_, &j0_, &k0_, &imax_, &jmax_, &kmax_);
  assert(i0_==i0 && j0_==j0 && k0_==k0 && imax_==imax && jmax_==jmax && kmax_==kmax);

  b.GetGhostedCornerIndices(&i0_, &j0_, &k0_, &imax_, &jmax_, &kmax_);
  assert(i0_==ii0 && j0_==jj0 && k0_==kk0 && imax_==iimax && jmax_==jjmax && kmax_==kkmax);
  x.GetGhostedCornerIndices(&i0_, &j0_, &k0_, &imax_, &jmax_, &kmax_);
  assert(i0_==ii0 && j0_==jj0 && k0_==kk0 && imax_==iimax && jmax_==jjmax && kmax_==kkmax);
  // ---------------------------------------------------
  

  Vec &bb(b.GetRefToGlobalVec());
  Vec &xx(x.GetRefToGlobalVec());

  PetscErrorCode error_code = KSPSolve(ksp, bb, xx);

  return (int)error_code; //0 means no error
}


//-----------------------------------------------------

//-----------------------------------------------------
