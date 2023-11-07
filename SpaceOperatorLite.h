/************************************************************************
 * Copyright © 2020 The Multiphysics Modeling and Computation (M2C) Lab
 * <kevin.wgy@gmail.com> <kevinw3@vt.edu>
 ************************************************************************/

#ifndef _SPACEOPERATOR_LITE_H_
#define _SPACEOPERATOR_LITE_H_
#include <GlobalMeshInfo.h>

/*******************************************
 * class SpaceOperator drives computations
 * that require domain/mesh information
 ******************************************/
class SpaceOperator
{
  MPI_Comm&                 comm;
  DataManagers3D&           dm_all;
  IoData&                   iod;

  //! Mesh info
  SpaceVariable3D coordinates;
  SpaceVariable3D delta_xyz;
  SpaceVariable3D volume; //!< volume of node-centered control volumes
  
  vector<GhostPoint> ghost_nodes_inner; //!< ghost nodes inside the physical domain (shared with other subd)
  vector<GhostPoint> ghost_nodes_outer; //!< ghost nodes outside the physical domain

  int i0, j0, k0, imax, jmax, kmax; //!< corners of the real subdomain
  int ii0, jj0, kk0, iimax, jjmax, kkmax; //!< corners of the ghosted subdomain
  int NX, NY, NZ; //!< global size

  GlobalMeshInfo &global_mesh;

  //! For temporary variable (5D)
  SpaceVariable3D Utmp;

  //! internal variable for temporary use (1D)
  SpaceVariable3D Tag;

public:
  SpaceOperator(MPI_Comm &comm_, DataManagers3D &dm_all_, IoData &iod_,
                GlobalMeshInfo &global_mesh_);
  ~SpaceOperator();

  SpaceVariable3D& GetMeshCoordinates() {return coordinates;}
  SpaceVariable3D& GetMeshDeltaXYZ()    {return delta_xyz;}
  SpaceVariable3D& GetMeshCellVolumes() {return volume;}

  vector<GhostPoint>* GetPointerToInnerGhostNodes() {return &ghost_nodes_inner;}
  vector<GhostPoint>* GetPointerToOuterGhostNodes() {return &ghost_nodes_outer;}

  GlobalMeshInfo& GetGlobalMeshInfo() {return global_mesh;}

  void Destroy();


private:

  void SetupMesh(vector<double> &x, vector<double> &y, vector<double> &z,
                 vector<double> &dx, vector<double> &dy, vector<double> &dz);
  void SetupMeshUniformRectangularDomain();
  void PopulateGhostBoundaryCoordinates();

  void CreateGhostNodeLists(bool screenout);

};


#endif
