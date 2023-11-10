/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include<petscksp.h>
#include<SpaceVariable.h>
#include<IoData.h>

/*********************************************************************
 * class LinearSystemSolver is responsible for solving large-scale linear
 * systems Ax = b, where x and b are SpaceVariable3D. The actual work is
 * done by PETSc. In some sense, LinearSystemSolver is just
 * a wrapper over PETSc/KSP. The class must be constructed with a function
 * that constructs the right-hand-side (b), and another function that
 * constructs the coefficient matrix (A).
 *********************************************************************
*/

class LinearSystemSolver {

  MPI_Comm &comm;
  DM dm; /**< This is a new DM object "cloned" from the one given in the constructor.\n
              According to PETsc documentation, one DM should be constructed for solving each system.*/
  KSP ksp;

  bool has_linear_operator; //!< whether the linear operator (the function that constructs "A") is given

  void *ctx;

public:

  LinearSystemSolver(MPI_Comm &comm_, DM &dm_, PETScKSPOptionsData &ksp_input);
  ~LinearSystemSolver(); 
  void Destroy();

  void SetLinearOperator(void *); //I AM HERE!

  int Solve(SpaceVariable3D &b, SpaceVariable3D &x); //!< x: both input (initial guess) & output (solution)

private:


};

