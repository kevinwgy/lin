/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include <time.h>
#include <petscdmda.h> //PETSc
#include <MeshGenerator.h>
#include <SpaceOperatorLite.h>
#include <Interpolator.h>
#include <GradientCalculatorCentral.h>
#include <GradientCalculatorFD3.h>
#include <GhostPoint.h>
#include <LinearSystemSolver.h>

#include <petscksp.h>

//#include <limits>

// for timing
//using std::chrono::high_resolution_clock;
//using std::chrono::duration_cast;
//using std::chrono::duration;
//using std::chrono::milliseconds;


int verbose;
MPI_Comm m2c_comm;
double start_time;


//-------------------------------------------------
// Example cases
void BuildLinearSystemEx1(SpaceVariable3D &coordinates,
                          vector<RowEntries> &row_entries, SpaceVariable3D &B, SpaceVariable3D &X);
void BuildLinearSystemEx2(PoissonEquationData &poisson, GlobalMeshInfo &global_mesh,
                          vector<RowEntries> &row_entries, SpaceVariable3D &B, SpaceVariable3D &X);
//-------------------------------------------------


/*************************************
 * Main Function
 ************************************/
int main(int argc, char* argv[])
{

  //! Initialize MPI 
  MPI_Init(NULL,NULL); //called together with all concurrent programs -> MPI_COMM_WORLD
  start_time = walltime();
  MPI_Comm comm = MPI_COMM_WORLD;
  m2c_comm = comm;

  //! Read user's input file (read the parameters)
  IoData iod(argc, argv);
  verbose = iod.output.verbose;
  iod.finalize();

/*
  print("Max Int = %d.\n", INT_MAX);
  print("Max long int = %lld.\n", LONG_MAX);
*/

  //! Calculate mesh coordinates
  vector<double> xcoords, dx, ycoords, dy, zcoords, dz;
  MeshGenerator meshgen;
  meshgen.ComputeMeshCoordinatesAndDeltas(iod.mesh, xcoords, ycoords, zcoords, dx, dy, dz);
  
  //! Setup global mesh info
  GlobalMeshInfo global_mesh(xcoords, ycoords, zcoords, dx, dy, dz);

  //! Initialize PETSc
  PETSC_COMM_WORLD = comm;
  PetscInitialize(&argc, &argv, argc>=3 ? argv[2] : (char*)0, (char*)0);

  //! Setup PETSc data array (da) structure for nodal variables
  DataManagers3D dms(comm, xcoords.size(), ycoords.size(), zcoords.size());

  //! Let global_mesh find subdomain boundaries and neighbors
  global_mesh.FindSubdomainInfo(comm, dms);

  //! Initialize space operator
  SpaceOperatorLite spo(comm, dms, iod, global_mesh);

  //! Initialize interpolator
  InterpolatorBase *interp = NULL;
  if(true) //may add more choices later
    interp = new InterpolatorLinear(comm, dms, spo.GetMeshCoordinates(), spo.GetMeshDeltaXYZ());

  //! Initialize (sptial) gradient calculator
  GradientCalculatorBase *grad = NULL;
  if(true) //may add more choices later
    grad = new GradientCalculatorCentral(comm, dms, spo.GetMeshCoordinates(), spo.GetMeshDeltaXYZ(), *interp);
  

  // *************************************
  // Main Loop 
  // *************************************

  double t = 0.0;
  print("\n");
  print("----------------------------\n");
  print("--          Start         --\n");
  print("----------------------------\n");

  
  LinearSystemSolver linsys(comm, dms.ghosted1_1dof, iod.linear_options);
  double rtol, abstol, dtol;
  int maxits;
  string ksp_type, pc_type;
  linsys.GetTolerances(&rtol, &abstol, &dtol, &maxits);
  linsys.GetSolverType(&ksp_type, &pc_type);
  print("PETSc parameters...\n");
  print("- solver: %s.\n", ksp_type.c_str());
  print("- preconditioner: %s.\n", pc_type.c_str());
  print("- rtol: %e.\n", rtol);
  print("- abstol: %e.\n", abstol);
  print("- dtol: %e.\n", dtol);
  print("- maxits: %d.\n", maxits);

  vector<RowEntries> row_entries;
  SpaceVariable3D X(comm, &(dms.ghosted1_1dof));
  SpaceVariable3D B(comm, &(dms.ghosted1_1dof));
  SpaceVariable3D Res(comm, &(dms.ghosted1_1dof)); //residual

  mpi_barrier();
  double timing1 = walltime();

#if LINEAR_SOLVER_TEST == 1
  BuildLinearSystemEx1(spo.GetMeshCoordinates(), row_entries, B, X);
#elif LINEAR_SOLVER_TEST == 2
  BuildLinearSystemEx2(iod.poisson, global_mesh, row_entries, B, X);
#else //by default, run test case 1
  BuildLinearSystemEx1(spo.GetMeshCoordinates(), row_entries, B, X);
#endif

  mpi_barrier();
  print("Computation time for building A, B, X outside PETSc: %f sec.\n", walltime()-timing1);

//  int mpi_rank;
//  MPI_Comm_rank(comm, &mpi_rank);
//  for(auto&& entries : row_entries) 
//    for(unsigned i=0; i<entries.cols.size(); i++)
//      fprintf(stdout,"[%d] Row (%d,%d,%d): Col (%d,%d,%d), v = %e.\n",
//              mpi_rank, entries.row.i, entries.row.j, entries.row.k,
//              entries.cols[i].i, entries.cols[i].j, entries.cols[i].k, entries.vals[i]);
//  exit_mpi();


  mpi_barrier(); timing1 = walltime();
  linsys.SetLinearOperator(row_entries);
  mpi_barrier();
  print("Computation time for building A in PETSc: %f sec.\n", walltime()-timing1);
  mpi_barrier(); timing1 = walltime();


  LinearSolverConvergenceReason reason;
  int nIters(0);
  vector<double> lin_rnorm;
  vector<int>    lin_rnorm_its;
  bool lin_success = linsys.Solve(B, X, &reason, &nIters, &lin_rnorm, &lin_rnorm_its);
  mpi_barrier();
  print("Computation time for solving AX=B by PETSc: %f sec.\n", walltime()-timing1);


  linsys.ComputeResidual(B, X, Res);
  double res_1norm = Res.CalculateVectorOneNorm();
  double res_2norm = Res.CalculateVectorTwoNorm();
  double res_inorm = Res.CalculateVectorInfNorm();
  
  if(!lin_success) {
    print_warning("  x Warning: Linear solver failed to converge.\n");
    for(int i=0; i<(int)lin_rnorm.size(); i++)
      print_warning("    > It. %d: residual = %e.\n", lin_rnorm_its[i]+1, lin_rnorm[i]);
  } else {
    print("- Linear solver converged in %d iterations (code: %d).\n", nIters,
          (int)reason);
    print("- Residual 1-/2-/inf-norm: %e, %e, %e.\n", res_1norm, res_2norm, res_inorm);
    for(int i=0; i<(int)lin_rnorm.size(); i++)
      print("    > It. %d: residual = %e.\n", lin_rnorm_its[i]+1, lin_rnorm[i]);

    double res_L1norm = Res.CalculateFunctionL1NormConRec(global_mesh); //spo.GetMeshCellVolumes());
    double res_L2norm = Res.CalculateFunctionL2NormConRec(global_mesh); //spo.GetMeshCellVolumes());
    print("- Residual L1 and L2 norm (const. rec.): %e, %e.\n", res_L1norm, res_L2norm);
  }

/*
  linsys.WriteToMatlabFile("A.txt", "A");
  X.WriteToMatlabFile("X.txt", "X");
  B.WriteToMatlabFile("B.txt", "B");
  Res.WriteToMatlabFile("Res.txt", "Res");
*/

  X.StoreMeshCoordinates(spo.GetMeshCoordinates());
  X.WriteToVTRFile("X.vtr","x");


  double lambda_max(0.0), lambda_min(0.0);
  linsys.CalculateExtremeSingularValues(lambda_max, lambda_min);
  print("\n");
  print("- Extreme singular values: %e (max), %e (min, estimate).\n", lambda_max, lambda_min);
  double cond = linsys.EstimateConditionNumber();
  print("- Condition number (estimate): %e.\n", cond);
  

  print("\n");
  print("\033[0;32m==========================================\033[0m\n");
  print("\033[0;32m   NORMAL TERMINATION (t = %e)  \033[0m\n", t); 
  print("\033[0;32m==========================================\033[0m\n");
  print("Total Computation Time: %f sec.\n", walltime()-start_time);
  print("\n");



  //! finalize 
  //! In general, "Destroy" should be called for classes that store Petsc DMDA data (which need to be "destroyed").
  

  linsys.Destroy();
  X.Destroy();
  B.Destroy();
  Res.Destroy();

  spo.Destroy();
  if(grad) {
    grad->Destroy();
    delete grad;
  }
  if(interp) {
    interp->Destroy();
    delete interp;
  }

  dms.DestroyAllDataManagers();

  PetscFinalize();
  MPI_Finalize();

  return 0;
}

//--------------------------------------------------------------
// A trivial diagonoal matrix A
void
BuildLinearSystemEx1(SpaceVariable3D &coordinates, 
                     std::vector<RowEntries> &row_entries, SpaceVariable3D &B, SpaceVariable3D &X)
{
  int i0, j0, k0, ii0, jj0, kk0;
  int imax, jmax, kmax, iimax, jjmax, kkmax;
  coordinates.GetCornerIndices(&i0, &j0, &k0, &imax, &jmax, &kmax);
  coordinates.GetGhostedCornerIndices(&ii0, &jj0, &kk0, &iimax, &jjmax, &kkmax);

  Vec3D*** coords = (Vec3D***)coordinates.GetDataPointer();
  
  double*** xx = X.GetDataPointer();
  double*** bb = B.GetDataPointer();
  

  for(int k=k0; k<kmax; k++)
    for(int j=j0; j<jmax; j++)
      for(int i=i0; i<imax; i++) {
        row_entries.push_back(RowEntries(1));
        RowEntries &entries(row_entries.back());
        entries.row.i = i;
        entries.row.j = j;
        entries.row.k = k;
        entries.cols.push_back(MatStencil());
        entries.cols.back().i = i;
        entries.cols.back().j = j;
        entries.cols.back().k = k;
        entries.vals.push_back(2.0);

        xx[k][j][i] = 0.0; //initial guess
        bb[k][j][i] = 2.0*coords[k][j][i].norm();
      }

  X.RestoreDataPointerAndInsert();
  B.RestoreDataPointerAndInsert();

  coordinates.RestoreDataPointerToLocalVector();

}

//--------------------------------------------------------------
// Poisson's equation (an internal function filling one axis of the scheme)
void 
FillEntriesInOneAxisEx2(int dir/*0,1,2 for x,y,z*/, int N, Int3 &ijk, double db, double df, double dc,
                        PoissonEquationData &poisson, GlobalMeshInfo &global_mesh,
                        RowEntries &entries, double &rhs, double &diag)
{
  assert(dir>=0 && dir<=2);
  int ind = ijk[dir];

  if(ind-1<0) {
    if(ind+1>=N) 
      return; //no derivative in this axis

    entries.cols.push_back(MatStencil());
    entries.cols.back().i = dir==0 ? ijk[0]+1 : ijk[0];
    entries.cols.back().j = dir==1 ? ijk[1]+1 : ijk[1];
    entries.cols.back().k = dir==2 ? ijk[2]+1 : ijk[2];

    // variables that depend on "dir"
    double dmin_i, dmin_p, bcval;
    PoissonEquationData::BoundaryType bctype;
    if(dir==0) {
      dmin_i = global_mesh.GetXmin() - global_mesh.GetX(ind);
      dmin_p = global_mesh.GetXmin() - global_mesh.GetX(ind+1);
      bctype = poisson.bc_x0;
      bcval  = poisson.bc_x0_val;
    } else if(dir==1) {
      dmin_i = global_mesh.GetYmin() - global_mesh.GetY(ind);
      dmin_p = global_mesh.GetYmin() - global_mesh.GetY(ind+1);
      bctype = poisson.bc_y0;
      bcval  = poisson.bc_y0_val;
    } else {//dir==2
      dmin_i = global_mesh.GetZmin() - global_mesh.GetZ(ind);
      dmin_p = global_mesh.GetZmin() - global_mesh.GetZ(ind+1);
      bctype = poisson.bc_z0;
      bcval  = poisson.bc_z0_val;
    }

    // fill entries
    if(bctype == PoissonEquationData::DIRICHLET) {
      rhs  -= 2.0*bcval/(dmin_i*dmin_p); //on the right-hand-side
      diag += 2.0/(dmin_i*df);
      entries.vals.push_back(-2.0/(dmin_p*df));
    }
    else {// Neumann
      rhs        -= 2.0*bcval/(dmin_i + dmin_p); //on the right-hand-side
      double val = 2.0/((dmin_i + dmin_p)*df);
      diag       += val;
      entries.vals.push_back(-val);
    }
  }
  else if(ind+1>=N) {
    if(ind-1<0) 
      return;

    entries.cols.push_back(MatStencil());
    entries.cols.back().i = dir==0 ? ijk[0]-1 : ijk[0];
    entries.cols.back().j = dir==1 ? ijk[1]-1 : ijk[1];
    entries.cols.back().k = dir==2 ? ijk[2]-1 : ijk[2];

    // variables that depend on "dir"
    double dmax_i, dmax_m, bcval;
    PoissonEquationData::BoundaryType bctype;
    if(dir==0) {
      dmax_i = global_mesh.GetXmax() - global_mesh.GetX(ind);
      dmax_m = global_mesh.GetXmax() - global_mesh.GetX(ind-1);
      bctype = poisson.bc_xmax;
      bcval  = poisson.bc_xmax_val;
    } else if(dir==1) {
      dmax_i = global_mesh.GetYmax() - global_mesh.GetY(ind);
      dmax_m = global_mesh.GetYmax() - global_mesh.GetY(ind-1);
      bctype = poisson.bc_ymax;
      bcval  = poisson.bc_ymax_val;
    } else {//dir==2
      dmax_i = global_mesh.GetZmax() - global_mesh.GetZ(ind);
      dmax_m = global_mesh.GetZmax() - global_mesh.GetZ(ind-1);
      bctype = poisson.bc_zmax;
      bcval  = poisson.bc_zmax_val;
    }

    if(bctype == PoissonEquationData::DIRICHLET) {
      rhs  -= 2.0*bcval/(dmax_i*dmax_m); //on the right-hand-side
      diag -= 2.0/(dmax_i*db);
      entries.vals.push_back(2.0/(dmax_m*db));
    }
    else {// Neumann
      rhs       -= 2.0*bcval/(dmax_i + dmax_m); //on the right-hand-side
      double val = 2.0/((dmax_i + dmax_m)*db);
      diag      -= val;
      entries.vals.push_back(val);
    }
  }
  else { //away from boundaries
    entries.cols.push_back(MatStencil());
    entries.cols.back().i = dir==0 ? ijk[0]-1 : ijk[0];
    entries.cols.back().j = dir==1 ? ijk[1]-1 : ijk[1];
    entries.cols.back().k = dir==2 ? ijk[2]-1 : ijk[2];
    entries.vals.push_back(2.0/(db*dc));

    diag -= 2.0/(db*df);

    entries.cols.push_back(MatStencil());
    entries.cols.back().i = dir==0 ? ijk[0]+1 : ijk[0];
    entries.cols.back().j = dir==1 ? ijk[1]+1 : ijk[1];
    entries.cols.back().k = dir==2 ? ijk[2]+1 : ijk[2];
    entries.vals.push_back(2.0/(df*dc));
  }
}

//--------------------------------------------------------------
// Poisson's equation
void
BuildLinearSystemEx2(PoissonEquationData &poisson, GlobalMeshInfo &global_mesh,
                     vector<RowEntries> &row_entries, SpaceVariable3D &B, SpaceVariable3D &X)
{
  int i0, j0, k0, imax, jmax, kmax;
  int NX, NY, NZ;
  X.GetCornerIndices(&i0, &j0, &k0, &imax, &jmax, &kmax);
  X.GetGlobalSize(&NX, &NY, &NZ);

  double*** xx = X.GetDataPointer();
  double*** bb = B.GetDataPointer();
  
  double dxb, dxf, dxc; //(x_i - x_(i-1)), (x_(i+1)-x_i), (x_(i+1)-x_(i-1))
  double dyb, dyf, dyc;
  double dzb, dzf, dzc;

  for(int k=k0; k<kmax; k++) {
    dzb = global_mesh.GetZ(k) - global_mesh.GetZ(k-1);
    dzf = global_mesh.GetZ(k+1) - global_mesh.GetZ(k);
    dzc = dzb + dzf;

    for(int j=j0; j<jmax; j++) {
      dyb = global_mesh.GetY(j) - global_mesh.GetY(j-1);
      dyf = global_mesh.GetY(j+1) - global_mesh.GetY(j);
      dyc = dyb + dyf;

      for(int i=i0; i<imax; i++) {
        dxb = global_mesh.GetX(i) - global_mesh.GetX(i-1);
        dxf = global_mesh.GetX(i+1) - global_mesh.GetX(i);
        dxc = dxb + dxf;
        
        // -----------------------------------------------------------
        // Setup this row
        // -----------------------------------------------------------
        row_entries.push_back(RowEntries(7)); //at most 7 non-zero entries on each row
        RowEntries &entries(row_entries.back());
        entries.row.i = i;
        entries.row.j = j;
        entries.row.k = k;

        bb[k][j][i] = 0.0; //no source term in this example
        double diag = 0.0; //collect the diagonal component from dd/ddx, dd/ddy, and dd/ddz

        // -----------------------------------------------------------
        // Calculate & register the non-zero entries on this row.
        // -----------------------------------------------------------
        Int3 ijk(i,j,k); 
        FillEntriesInOneAxisEx2(0/*x-dir*/, NX, ijk, dxb, dxf, dxc, poisson, global_mesh,
                                entries, bb[k][j][i], diag);
        FillEntriesInOneAxisEx2(1/*y-dir*/, NY, ijk, dyb, dyf, dyc, poisson, global_mesh,
                                entries, bb[k][j][i], diag);
        FillEntriesInOneAxisEx2(2/*z-dir*/, NZ, ijk, dzb, dzf, dzc, poisson, global_mesh,
                                entries, bb[k][j][i], diag);

        // add diagonal
        entries.cols.push_back(MatStencil());
        entries.cols.back().i = i;
        entries.cols.back().j = j;
        entries.cols.back().k = k;
        entries.vals.push_back(diag);

        assert(diag!=0.0);
        xx[k][j][i] = bb[k][j][i]/diag; //initial guess
      }
    }
  }

  X.RestoreDataPointerAndInsert();
  B.RestoreDataPointerAndInsert();
}

//--------------------------------------------------------------
