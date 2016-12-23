/* ----------------------------------------------------------------------
 LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "math.h"
#include "stdlib.h"
#include "pair_sph_solidMechanics_mg.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "memory.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "math_extra_liggghts.h"
#include "sph_kernel_cubicspline.h"
#include "sph_kernel_Bspline.h"
#include "sph_kernel_cubicspline2D.h"
#include "math_extra_pysph.h"

using namespace LAMMPS_NS;
#define det3By3(A) \
	A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) \
   -A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) \
   +A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]) 

#define TOLERANCE 1e-15
/* ---------------------------------------------------------------------- */

PairSPHSolidMechanicsMG::PairSPHSolidMechanicsMG(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

  update_drho_flag_ = 1;
  epsXSPH_ = 0.5;

  comm_forward = 66; // R[9], epsilon[9], sigma[9], tau[9], dTau[9], epsilonBar[9], artStress[9], drho, de, p = 7*9 + 2 = 66
//  ghostneigh = 1;
}

/* ---------------------------------------------------------------------- */

PairSPHSolidMechanicsMG::~PairSPHSolidMechanicsMG() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
//    memory->destroy(cutghost);

    memory->destroy(cut);
    memory->destroy(viscosity);
    memory->destroy(beta_);
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::init_style() {
// need a full neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
//	neighbor->requests[irequest]->ghost = 1;
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::compute(int eflag, int vflag) {

	if(update_drho_flag_) compute_drho(eflag, vflag);
	else compute_no_drho(eflag, vflag);

}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::compute_no_drho(int eflag, int vflag) {
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;

  double wdeltap, wf, fab;

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **v = atom->vest;
  double **vXSPH = atom->vXSPH_;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *rmass = atom->rmass;
  double *de = atom->de;
  double *e = atom->e;
  double *drho = atom->drho;
  double ***dTau_ = atom->dTau_;
  int *type = atom->type;

  double ***epsilon_ = atom->epsilon_;
  double ***sigma_ = atom->sigma_;
  double ***artStress_ = atom->artStress_;
  double ***epsilonBar_ = atom->epsilonBar_;
  double ***R_ = atom->R_;
  double ***tau_ = atom->tau_;
  double *p_ = atom->p_;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double rhoi, rhoj;
  double fix, fiy, fiz;

  // loop over neighbors of my atoms

  if(domain->dimension == 3)
  {
	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	

		#include"computeStrainStress3D.h"

	  }
	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	

		#include"performSolidMomentumEqn_mg_no_drho_3D.h"

	  }
  }
  else
  {
	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	

		#include"computeStrainStress2D.h"

	  }

	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	

		#include"performSolidMomentumEqn_mg_no_drho_2D.h"

	  }
  }
  // communicate densities
  // comm->forward_comm_pair(this);

  // if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::compute_drho(int eflag, int vflag) {
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;

  double wdeltap, wf, fab;

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **v = atom->vest;
  double **vXSPH = atom->vXSPH_;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *rmass = atom->rmass;
  double *de = atom->de;
  double *e = atom->e;
  double *drho = atom->drho;
  double ***dTau_ = atom->dTau_;
  int *type = atom->type;

  double ***epsilon_ = atom->epsilon_;
  double ***sigma_ = atom->sigma_;
  double ***artStress_ = atom->artStress_;
  double ***epsilonBar_ = atom->epsilonBar_;
  double ***R_ = atom->R_;
  double ***tau_ = atom->tau_;
  double *p_ = atom->p_;

//  int nlocal = atom->nlocal;
//  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double rhoi, rhoj;
  double fix, fiy, fiz;

  // loop over neighbors of my atoms

  if(domain->dimension == 3)
  {
	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	

		#include"computeStrainStress3D.h"

	  }

	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];
		ci = c_;	
		#include"performSolidMomentumEqn_mg_with_drho_3D.h"

	  }
  }
  else
  {
	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	
		#include"computeStrainStress2D.h"
	  }

	  for (ii = 0; ii < inum; ii++) {
		i = ilist[ii];
		xtmp = x[i][0];
		ytmp = x[i][1];
		ztmp = x[i][2];
		vxtmp = v[i][0];
		vytmp = v[i][1];
		vztmp = v[i][2];
		itype = type[i];
		jlist = firstneigh[i];
		jnum = numneigh[i];

		imass = rmass[i];
		rhoi = rho[i];

		ci = c_;	
		#include"performSolidMomentumEqn_mg_with_drho_2D.h"
	  }

  }
  // communicate densities
  // comm->forward_comm_pair(this);

  // if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
//  memory->create(cutghost,n+1,n+1,"pair:cutghost");

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(viscosity, n + 1, n + 1, "pair:viscosity");
  memory->create(beta_, n + 1, n + 1, "pair:beta");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/solidMechanics/mg");

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsMG::coeff(int narg, char **arg) {
  if (narg != 11+2+1+1+1)
    error->all(FLERR,"Incorrect number of args for pair_style sph/solidMechanics/mg coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double viscosity_one = force->numeric(FLERR, arg[2]);
  double beta_one = force->numeric(FLERR, arg[3]);
  double cut_one = force->numeric(FLERR, arg[4]);
  h_ = force->numeric(FLERR, arg[4+1]);

  G_ = force->numeric(FLERR, arg[5+1]);
  Gamma_ = force->numeric(FLERR, arg[6+1]);
  c_ = force->numeric(FLERR, arg[7+1]);
  Cs_ = c_;
  S_ = force->numeric(FLERR, arg[8+1]);
  rho0_ = force->numeric(FLERR, arg[9+1]);
  Y0_ = force->numeric(FLERR, arg[10+1]);

  if(strcmp(arg[11+1],"yes") == 0) 
  {
	  update_drho_flag_ = 1;
	  if(comm->me == 0)
		  printf("density change will be computed!\n");
  }
  else
  {
	  update_drho_flag_ = 0;
	  if(comm->me == 0)
		  printf("density change will NOT be computed!\n");
  }

  eps_ = force->numeric(FLERR, arg[12+1]);
  deltap_ = force->numeric(FLERR, arg[13+1]);
  epsXSPH_ = force->numeric(FLERR, arg[13+1+1]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      viscosity[i][j] = viscosity_one;
      viscosity[j][i] = viscosity_one;
      beta_[i][j] = beta_one;
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair sph/solidMechanics/mg coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHSolidMechanicsMG::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair sph/solidMechanics/mg coeffs are not set");
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHSolidMechanicsMG::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}

/* ---------------------------------------------------------------------- */
/*
int PairSPHSolidMechanicsMG::pack_comm(int n, int *list, double *buf, int pbc_flag,
    int *pbc) {
  int i, j, m;
  double ***R_ = atom->R_;
  double ***epsilon_ = atom->epsilon_;
  double ***sigma_ = atom->sigma_;
  double ***tau_ = atom->tau_;
  double ***dTau_ = atom->dTau_;
  double ***epsilonBar_ = atom->epsilonBar_; // may not necessary
  double ***artStress_ = atom->artStress_; // may not necessary
  double *drho = atom->drho;
  double *de = atom->de;
  double *p_ = atom->p_;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
//    buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = epsilon_[j][0][0];
      buf[m++] = epsilon_[j][0][1];
      buf[m++] = epsilon_[j][0][2];
      buf[m++] = epsilon_[j][1][0];
      buf[m++] = epsilon_[j][1][1];
      buf[m++] = epsilon_[j][1][2];
      buf[m++] = epsilon_[j][2][0];
      buf[m++] = epsilon_[j][2][1];
      buf[m++] = epsilon_[j][2][2];

      buf[m++] = sigma_[j][0][0];
      buf[m++] = sigma_[j][0][1];
      buf[m++] = sigma_[j][0][2];
      buf[m++] = sigma_[j][1][0];
      buf[m++] = sigma_[j][1][1];
      buf[m++] = sigma_[j][1][2];
      buf[m++] = sigma_[j][2][0];
      buf[m++] = sigma_[j][2][1];
      buf[m++] = sigma_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = dTau_[j][0][0];
      buf[m++] = dTau_[j][0][1];
      buf[m++] = dTau_[j][0][2];
      buf[m++] = dTau_[j][1][0];
      buf[m++] = dTau_[j][1][1];
      buf[m++] = dTau_[j][1][2];
      buf[m++] = dTau_[j][2][0];
      buf[m++] = dTau_[j][2][1];
      buf[m++] = dTau_[j][2][2];

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = artStress_[j][0][0];
      buf[m++] = artStress_[j][0][1];
      buf[m++] = artStress_[j][0][2];
      buf[m++] = artStress_[j][1][0];
      buf[m++] = artStress_[j][1][1];
      buf[m++] = artStress_[j][1][2];
      buf[m++] = artStress_[j][2][0];
      buf[m++] = artStress_[j][2][1];
      buf[m++] = artStress_[j][2][2];

      buf[m++] = drho[j];
      buf[m++] = de[j];
      buf[m++] = p_[j];

  }
  return m;
}
*/
/* ---------------------------------------------------------------------- */
/*
void PairSPHSolidMechanicsMG::unpack_comm(int n, int first, double *buf) {
  int i, m, last;
  double ***R_ = atom->R_;
  double ***epsilon_ = atom->epsilon_;
  double ***sigma_ = atom->sigma_;
  double ***tau_ = atom->tau_;
  double ***dTau_ = atom->dTau_;
  double ***epsilonBar_ = atom->epsilonBar_; // may not necessary
  double ***artStress_ = atom->artStress_; // may not necessary
  double *drho = atom->drho;
  double *de = atom->de;
  double *p_ = atom->p_;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++)
//    rho[i] = buf[m++];
      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      epsilon_[i][0][0] = buf[m++];
      epsilon_[i][0][1] = buf[m++];
      epsilon_[i][0][2] = buf[m++];
      epsilon_[i][1][0] = buf[m++];
      epsilon_[i][1][1] = buf[m++];
      epsilon_[i][1][2] = buf[m++];
      epsilon_[i][2][0] = buf[m++];
      epsilon_[i][2][1] = buf[m++];
      epsilon_[i][2][2] = buf[m++];

      sigma_[i][0][0] = buf[m++];
      sigma_[i][0][1] = buf[m++];
      sigma_[i][0][2] = buf[m++];
      sigma_[i][1][0] = buf[m++];
      sigma_[i][1][1] = buf[m++];
      sigma_[i][1][2] = buf[m++];
      sigma_[i][2][0] = buf[m++];
      sigma_[i][2][1] = buf[m++];
      sigma_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

      dTau_[i][0][0] = buf[m++];
      dTau_[i][0][1] = buf[m++];
      dTau_[i][0][2] = buf[m++];
      dTau_[i][1][0] = buf[m++];
      dTau_[i][1][1] = buf[m++];
      dTau_[i][1][2] = buf[m++];
      dTau_[i][2][0] = buf[m++];
      dTau_[i][2][1] = buf[m++];
      dTau_[i][2][2] = buf[m++];

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      artStress_[i][0][0] = buf[m++];
      artStress_[i][0][1] = buf[m++];
      artStress_[i][0][2] = buf[m++];
      artStress_[i][1][0] = buf[m++];
      artStress_[i][1][1] = buf[m++];
      artStress_[i][1][2] = buf[m++];
      artStress_[i][2][0] = buf[m++];
      artStress_[i][2][1] = buf[m++];
      artStress_[i][2][2] = buf[m++];

      drho[i] = buf[m++];
      de[i] = buf[m++];
      p_[i] = buf[m++];
}
*/
