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
clock_t start_time;


//-------------------------------------------------
// Example cases
void BuildLinearSystemEx1(SpaceVariable3D &coordinates, SpaceVariable3D &delta_xyz,
                          vector<RowEntries> &row_entries, SpaceVariable3D &B, SpaceVariable3D &X);
//-------------------------------------------------


/*************************************
 * Main Function
 ************************************/
int main(int argc, char* argv[])
{
  start_time = clock(); //for timing purpose only

  //! Initialize MPI 
  MPI_Init(NULL,NULL); //called together with all concurrent programs -> MPI_COMM_WORLD
  MPI_Comm comm = MPI_COMM_WORLD;
  m2c_comm = comm;

  //! Read user's input file (read the parameters)
  IoData iod(argc, argv);
  verbose = iod.output.verbose;
  iod.finalize();


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
  global_mesh.GetSubdomainInfo(comm, dms);

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
  

  /*************************************
   * Main Loop 
   ************************************/
  double t = 0.0;
  print("\n");
  print("----------------------------\n");
  print("--          Start         --\n");
  print("----------------------------\n");

  
  LinearSystemSolver linsys(comm, dms.ghosted1_1dof, iod.petsc_ksp_options);

  vector<RowEntries> row_entries;
  SpaceVariable3D X(comm, &(dms.ghosted1_1dof));
  SpaceVariable3D B(comm, &(dms.ghosted1_1dof));

  mpi_barrier();
  clock_t timing1 = clock(); //for timing purpose only
  BuildLinearSystemEx1(spo.GetMeshCoordinates(), spo.GetMeshDeltaXYZ(), row_entries, B, X);
  mpi_barrier();
  print("Computation time for building A, B, X outside PETSc: %f sec.\n",
        ((double)(clock()-timing1))/CLOCKS_PER_SEC);

/* 
  int mpi_rank;
  MPI_Comm_rank(comm, &mpi_rank);
  for(auto&& entries : row_entries) 
    for(unsigned i=0; i<entries.cols.size(); i++)
      fprintf(stdout,"[%d] Row (%d,%d,%d): Col (%d,%d,%d), v = %e.\n",
              mpi_rank, entries.row.i, entries.row.j, entries.row.k,
              entries.cols[i].i, entries.cols[i].j, entries.cols[i].k, entries.vals[i]);
*/

  mpi_barrier(); timing1 = clock();
  linsys.SetLinearOperator(row_entries);
  mpi_barrier();
  print("Computation time for building A in PETSc: %f sec.\n",
        ((double)(clock()-timing1))/CLOCKS_PER_SEC);

  mpi_barrier(); timing1 = clock();
  linsys.Solve(B,X);
  mpi_barrier();
  print("Computation time for solving AX=B by PETSc: %f sec.\n",
        ((double)(clock()-timing1))/CLOCKS_PER_SEC);


  X.StoreMeshCoordinates(spo.GetMeshCoordinates());
  X.WriteToVTRFile("X.vtr","x");

  B.StoreMeshCoordinates(spo.GetMeshCoordinates());
  B.WriteToVTRFile("B.vtr","b");


  print("\n");
  print("\033[0;32m==========================================\033[0m\n");
  print("\033[0;32m   NORMAL TERMINATION (t = %e)  \033[0m\n", t); 
  print("\033[0;32m==========================================\033[0m\n");
  print("Total Computation Time: %f sec.\n", ((double)(clock()-start_time))/CLOCKS_PER_SEC);
  print("\n");



  //! finalize 
  //! In general, "Destroy" should be called for classes that store Petsc DMDA data (which need to be "destroyed").
  
  linsys.Destroy();
  X.Destroy();
  B.Destroy();

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


void
BuildLinearSystemEx1(SpaceVariable3D &coordinates, SpaceVariable3D &delta_xyz,
                     std::vector<RowEntries> &row_entries, SpaceVariable3D &B, SpaceVariable3D &X)
{
  int i0, j0, k0, ii0, jj0, kk0;
  int imax, jmax, kmax, iimax, jjmax, kkmax;
  coordinates.GetCornerIndices(&i0, &j0, &k0, &imax, &jmax, &kmax);
  coordinates.GetGhostedCornerIndices(&ii0, &jj0, &kk0, &iimax, &jjmax, &kkmax);

  [[maybe_unused]] Vec3D*** coords = (Vec3D***)coordinates.GetDataPointer();
  [[maybe_unused]] Vec3D*** dxyz   = (Vec3D***)delta_xyz.GetDataPointer();
  
  [[maybe_unused]] double*** xx = X.GetDataPointer();
  [[maybe_unused]] double*** bb = B.GetDataPointer();
  

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
  delta_xyz.RestoreDataPointerToLocalVector();

  [[maybe_unused]] int n = row_entries.size();
}


//--------------------------------------------------------------
