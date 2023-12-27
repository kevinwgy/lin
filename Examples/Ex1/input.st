under Mesh { 
  X0   = -1.0;
  Xmax =  1.0;
  Y0   = -2.0;
  Ymax =  2.0;
  Z0   = -1.0;
  Zmax =  1.0;

  NumberOfCellsX = 200;
  NumberOfCellsY = 400;
  NumberOfCellsZ = 200;
}


under PoissonEquation {

  BoundaryConditionX0   = Dirichlet;
  BoundaryValueX0       = -1.0;
  BoundaryConditionXmax = Dirichlet;
  BoundaryValueXmax     = 1.0;

  BoundaryConditionY0   = Dirichlet;
  BoundaryValueY0       = -2.5;
  BoundaryConditionYmax = Dirichlet;
  BoundaryValueYmax     = 2.5;

  BoundaryConditionZ0   = Neumann;
  BoundaryConditionZmax = Neumann;
}


under LinearSystemSolver {
  Type = FlexibleGMRes;
  Preconditioner = BlockJacobi;
  RelativeErrorTolerance = 1.0e-4;

  PETScOptionsFile = "petsc_options.txt";
}
