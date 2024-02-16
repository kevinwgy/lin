under Mesh { 
  X0   = -1.0;
  Xmax =  1.0;
  Y0   = -2.0;
  Ymax =  2.0;
  Z0   = -1.0;
  Zmax =  1.0;

  //NumberOfCellsX = 100;
  //NumberOfCellsY = 200;
  //NumberOfCellsZ = 100;
  NumberOfCellsX = 20;
  NumberOfCellsY = 40;
  NumberOfCellsZ = 20;
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
  Type = FlexibleGMRes; //or StabilizedBiCG or ImprovedStabilizedBiCG
  Preconditioner = BlockJacobi; //or None or Jacobe or MultiGrid
  //PrintLogToScreen = No;
  LogFile = "linlog.txt";
  PETScOptionsFile = "petsc_options.txt"; //any other PETSc option/param
}
