/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#include <SpaceOperatorLite.h>
#include <Utils.h>
#include <Vector3D.h>
#include <Vector5D.h>

extern int verbose;

//-----------------------------------------------------

SpaceOperatorLite::SpaceOperatorLite(MPI_Comm &comm_, DataManagers3D &dm_all_, IoData &iod_,
                                     GlobalMeshInfo &global_mesh_)
  : comm(comm_), dm_all(dm_all_),
    iod(iod_), 
    coordinates(comm_, &(dm_all_.ghosted1_3dof)),
    delta_xyz(comm_, &(dm_all_.ghosted1_3dof)),
    volume(comm_, &(dm_all_.ghosted1_1dof)), global_mesh(global_mesh_),
    Utmp(comm_, &(dm_all_.ghosted1_5dof)),
    Tag(comm_, &(dm_all_.ghosted1_1dof))
{
  
  coordinates.GetCornerIndices(&i0, &j0, &k0, &imax, &jmax, &kmax);
  coordinates.GetGhostedCornerIndices(&ii0, &jj0, &kk0, &iimax, &jjmax, &kkmax);
  coordinates.GetGlobalSize(&NX, &NY, &NZ);

  SetupMesh(global_mesh.x_glob, global_mesh.y_glob, global_mesh.z_glob,
            global_mesh.dx_glob, global_mesh.dy_glob, global_mesh.dz_glob);

  CreateGhostNodeLists(true); //create ghost_nodes_inner and ghost_nodes_outer

}

//-----------------------------------------------------

SpaceOperatorLite::~SpaceOperatorLite()
{
}

//-----------------------------------------------------

void SpaceOperatorLite::Destroy()
{
  coordinates.Destroy();
  delta_xyz.Destroy();
  volume.Destroy();
  Utmp.Destroy();
  Tag.Destroy();
}

//-----------------------------------------------------

void SpaceOperatorLite::SetupMesh(vector<double> &x, vector<double> &y, vector<double> &z,
                              vector<double> &dx, vector<double> &dy, vector<double> &dz)
{
  //! Setup coordinates of cell centers and dx, dy, dz

  //! get array to edit
  Vec3D*** coords = (Vec3D***)coordinates.GetDataPointer();
  Vec3D*** dxyz   = (Vec3D***)delta_xyz.GetDataPointer();

  //! Fill the actual subdomain, w/o ghost cells 
  for(int k=k0; k<kmax; k++)
    for(int j=j0; j<jmax; j++)
      for(int i=i0; i<imax; i++) {
        coords[k][j][i][0] = x[i];
        coords[k][j][i][1] = y[j];
        coords[k][j][i][2] = z[k];
        dxyz[k][j][i][0] = dx[i];
        dxyz[k][j][i][1] = dy[j];
        dxyz[k][j][i][2] = dz[k];
      } 

  //! restore array
  coordinates.RestoreDataPointerAndInsert(); //update localVec and globalVec;
  delta_xyz.RestoreDataPointerAndInsert(); //update localVec and globalVec;

  //! Populate the ghost cells (coordinates, dx, dy, dz)
  PopulateGhostBoundaryCoordinates();


  //(Obsolete. Can be used for debugging purpose though.) 
  //SetupMeshUniformRectangularDomain();


  //! Compute mesh information
  dxyz = (Vec3D***)delta_xyz.GetDataPointer();
  double*** vol  = (double***)volume.GetDataPointer();


  /** Calculate the volume/area of node-centered control volumes ("cells")
   *  Include ghost cells. 
   */
  for(int k=kk0; k<kkmax; k++)
    for(int j=jj0; j<jjmax; j++)
      for(int i=ii0; i<iimax; i++) {
        vol[k][j][i] /*volume of cv*/ = dxyz[k][j][i][0]*dxyz[k][j][i][1]*dxyz[k][j][i][2];
//        fprintf(stdout,"(%d,%d,%d), dx = %e, dy = %e, dz = %e, vol = %e.\n", i,j,k, dxyz[k][j][i][0], dxyz[k][j][i][1], dxyz[k][j][i][2], vol[k][j][i]);
      }


  delta_xyz.RestoreDataPointerAndInsert();
  volume.RestoreDataPointerAndInsert();

}

//-----------------------------------------------------

void SpaceOperatorLite::SetupMeshUniformRectangularDomain()
{

  double dx = (iod.mesh.xmax - iod.mesh.x0)/NX;
  double dy = (iod.mesh.ymax - iod.mesh.y0)/NY;
  double dz = (iod.mesh.zmax - iod.mesh.z0)/NZ;

  //! get array to edit
  Vec3D*** coords = (Vec3D***)coordinates.GetDataPointer();
  Vec3D*** dxyz   = (Vec3D***)delta_xyz.GetDataPointer();

  //! Fill the actual subdomain, w/o ghost cells 
  for(int k=k0; k<kmax; k++)
    for(int j=j0; j<jmax; j++)
      for(int i=i0; i<imax; i++) {
        coords[k][j][i][0] = iod.mesh.x0 + 0.5*dx + i*dx; 
        coords[k][j][i][1] = iod.mesh.y0 + 0.5*dy + j*dy; 
        coords[k][j][i][2] = iod.mesh.z0 + 0.5*dz + k*dz; 
        dxyz[k][j][i][0] = dx;
        dxyz[k][j][i][1] = dy;
        dxyz[k][j][i][2] = dz;
      } 

  //! restore array
  coordinates.RestoreDataPointerAndInsert(); //update localVec and globalVec;
  delta_xyz.RestoreDataPointerAndInsert(); //update localVec and globalVec;

  //! Populate the ghost cells (coordinates, dx, dy, dz)
  PopulateGhostBoundaryCoordinates();
}

//-----------------------------------------------------
/** Populate the coordinates, dx, dy, and dz of ghost cells */
void SpaceOperatorLite::PopulateGhostBoundaryCoordinates()
{
  Vec3D*** v    = (Vec3D***) coordinates.GetDataPointer();
  Vec3D*** dxyz = (Vec3D***) delta_xyz.GetDataPointer();

  int nnx, nny, nnz;
  coordinates.GetGhostedSize(&nnx, &nny, &nnz);

  // capture the mesh info of the corners 
  double v0[3], v1[3];
  double dxyz0[3], dxyz1[3];
  for(int p=0; p<3; p++) {
    v0[p]    = v[k0][j0][i0][p] - dxyz[k0][j0][i0][p];
    v1[p]    = v[kmax-1][jmax-1][imax-1][p] + dxyz[kmax-1][jmax-1][imax-1][p];
    dxyz0[p] = dxyz[k0][j0][i0][p];
    dxyz1[p] = dxyz[kmax-1][jmax-1][imax-1][p];
  }

  for(int k=kk0; k<kkmax; k++)
    for(int j=jj0; j<jjmax; j++)
      for(int i=ii0; i<iimax; i++) {

        if(k!=-1 && k!=NZ && j!=-1 && j!=NY && i!=-1 && i!=NX)
          continue; //not in the ghost layer of the physical domain

        Vec3D& X  = v[k][j][i];
        Vec3D& dX = dxyz[k][j][i];

        bool xdone = false, ydone = false, zdone = false;

        if(i==-1) {
          X[0]  = v0[0];
          dX[0] = dxyz0[0];
          xdone = true;
        }

        if(i==NX) {
          X[0]  = v1[0];
          dX[0] = dxyz1[0];
          xdone = true;
        }

        if(j==-1) {
          X[1]  = v0[1];
          dX[1] = dxyz0[1];
          ydone = true;
        }

        if(j==NY) {
          X[1]  = v1[1];
          dX[1] = dxyz1[1];
          ydone = true;
        }

        if(k==-1) {
          X[2]  = v0[2];
          dX[2] = dxyz0[2];
          zdone = true;
        }

        if(k==NZ) {
          X[2]  = v1[2];
          dX[2] = dxyz1[2];
          zdone = true;
        }

        if(!xdone) {
          X[0]  = v[k0][j0][i][0];    //x[i]
          dX[0] = dxyz[k0][j0][i][0]; //dx[i]
        }

        if(!ydone) {
          X[1]  = v[k0][j][i0][1];    //y[j]
          dX[1] = dxyz[k0][j][i0][1]; //dy[j]
        }

        if(!zdone) {
          X[2]  = v[k][j0][i0][2];    //z[k]
          dX[2] = dxyz[k][j0][i0][2]; //dz[k]
        }

      }
  
  coordinates.RestoreDataPointerAndInsert();
  delta_xyz.RestoreDataPointerAndInsert();
}

//-----------------------------------------------------

void SpaceOperatorLite::CreateGhostNodeLists(bool screenout)
{
  ghost_nodes_inner.clear();
  ghost_nodes_outer.clear();

  Vec3D*** coords = (Vec3D***)coordinates.GetDataPointer();

  Int3 image;
  Vec3D proj(0.0), out_normal(0.0);
  MeshData::BcType bcType = MeshData::NONE;
  GhostPoint::Side side = GhostPoint::UNDEFINED;
  int counter;
  for(int k=kk0; k<kkmax; k++)
    for(int j=jj0; j<jjmax; j++)
      for(int i=ii0; i<iimax; i++) {

        if(k>=k0 && k<kmax && j>=j0 && j<jmax && i>=i0 && i<imax)
          continue; //interior of the subdomain

        if(coordinates.OutsidePhysicalDomain(i,j,k)) {//outside physical domain

          //determine the image, the projection point and boundary condition
          image      = 0;
          proj       = 0.0; 
          out_normal = 0.0;
          counter    = 0; 
          bcType     = MeshData::NONE;
          side       = GhostPoint::UNDEFINED;

          if(i<0)        {image[0] = -i-1;
                          proj[0]  = iod.mesh.x0;      out_normal[0] = -1.0; 
                          bcType   = iod.mesh.bc_x0;   side = GhostPoint::LEFT;     counter++;}
          else if(i>=NX) {image[0] = NX+(i-NX)-1;
                          proj[0]  = iod.mesh.xmax;    out_normal[0] =  1.0; 
                          bcType   = iod.mesh.bc_xmax; side = GhostPoint::RIGHT;    counter++;}
          else           {image[0] = i;
                          proj[0]  = coords[k][j][i][0];}
                     

          if(j<0)        {image[1] = -j-1;
                          proj[1]  = iod.mesh.y0;      out_normal[1] = -1.0; 
                          bcType   = iod.mesh.bc_y0;   side = GhostPoint::BOTTOM;   counter++;}
          else if(j>=NY) {image[1] = NY+(j-NY)-1;
                          proj[1]  = iod.mesh.ymax;    out_normal[1] =  1.0; 
                          bcType   = iod.mesh.bc_ymax; side = GhostPoint::TOP;      counter++;}
          else           {image[1] = j;
                          proj[1]  = coords[k][j][i][1];}
         

          if(k<0)        {image[2] = -k-1;
                          proj[2]  = iod.mesh.z0;      out_normal[2] = -1.0;
                          bcType   = iod.mesh.bc_z0;   side = GhostPoint::BACK;     counter++;}
          else if(k>=NZ) {image[2] = NZ+(k-NZ)-1;
                          proj[2]  = iod.mesh.zmax;    out_normal[2] =  1.0; 
                          bcType   = iod.mesh.bc_zmax; side = GhostPoint::FRONT;    counter++;}
          else           {image[2] = k;
                          proj[2]  = coords[k][j][i][2];}
         
          out_normal /= out_normal.norm();

          assert(counter<=3 && counter>0);

          if(counter == 1)
            ghost_nodes_outer.push_back(GhostPoint(Int3(i,j,k), image, GhostPoint::FACE,
                                        proj, out_normal, (int)bcType, side));
          else if(counter == 2)
            ghost_nodes_outer.push_back(GhostPoint(Int3(i,j,k), image, GhostPoint::EDGE,
                                        proj, out_normal, 0));
          else
            ghost_nodes_outer.push_back(GhostPoint(Int3(i,j,k), image, GhostPoint::VERTEX,
                                        proj, out_normal, 0));

          // collect ghost nodes along overset boundaries if any (FACE only)
          //if(bcType==MeshData::OVERSET && counter==1)
            //ghost_overset.push_back(std::make_pair(Int3(i,j,k), Vec5D(0.0)));

        } 
        else //inside physical domain
          ghost_nodes_inner.push_back(GhostPoint(i,j,k));

      }

  // Find out the owner of each inner ghost
  int mpi_rank(-1);
  MPI_Comm_rank(comm, &mpi_rank);
  Tag.SetConstantValue(mpi_rank, false);
  double *** tag = Tag.GetDataPointer();
  for(auto it = ghost_nodes_inner.begin(); it != ghost_nodes_inner.end(); it++)
    it->owner_proc = (int)tag[it->ijk[2]][it->ijk[1]][it->ijk[0]];

  for(int k=kk0; k<kkmax; k++)
    for(int j=jj0; j<jjmax; j++)
      for(int i=ii0; i<iimax; i++)
        tag[k][j][i] = 0.0; //restore default value (so it can be used for other purposes w/o confusion)
  Tag.RestoreDataPointerToLocalVector();


  int nInner = ghost_nodes_inner.size();
  int nOuter = ghost_nodes_outer.size();
  MPI_Allreduce(MPI_IN_PLACE, &nInner, 1, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, &nOuter, 1, MPI_INT, MPI_SUM, comm);
  if(screenout) {
    print(comm,"  o Number of ghost nodes inside computational domain (overlapping between subdomains): %d\n",
          nInner);
    print(comm,"  o Number of ghost nodes outside computational domain: %d\n",
          nOuter);
    print(comm,"\n");
  }

  // figure out whether the entire domain has overset ghosts...
  int overset_count = 0; //ghost_overset.size();
  MPI_Allreduce(MPI_IN_PLACE, &overset_count, 1, MPI_INT, MPI_SUM, comm);
  //domain_has_overset = overset_count>0;


  coordinates.RestoreDataPointerToLocalVector();
}

//-----------------------------------------------------

