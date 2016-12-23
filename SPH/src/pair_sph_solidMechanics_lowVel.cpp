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
#include "pair_sph_solidMechanics_lowVel.h"
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
using namespace MathExtraLiggghts;
#define det3By3(A) \
	A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) \
   -A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) \
   +A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]) 

#define TOLERANCE 1e-15
#define SMALL  1e-15
/* ---------------------------------------------------------------------- */

PairSPHSolidMechanicsLowVel::PairSPHSolidMechanicsLowVel(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

  update_drho_flag_ = 1;
  epsXSPH_ = 0.5;

  comm_forward = 66; // R[9], epsilon[9], sigma[9], tau[9], dTau[9], epsilonBar[9], artStress[9], drho, de, p = 7*9 + 2 = 66
//  ghostneigh = 1;
	epsiloneqv = 0.0;
	iterMax = 100;
}

/* ---------------------------------------------------------------------- */

PairSPHSolidMechanicsLowVel::~PairSPHSolidMechanicsLowVel() {
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

void PairSPHSolidMechanicsLowVel::init_style() {
// need a full neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
//	neighbor->requests[irequest]->ghost = 1;
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsLowVel::compute(int eflag, int vflag) {

	if(update_drho_flag_) compute_drho(eflag, vflag);
	else compute_no_drho(eflag, vflag);

}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsLowVel::find_properties(int i) {

	// compute Y0
	double tauxx = atom->tau_[i][0][0];
	double tauxy = atom->tau_[i][0][1];
	double tauxz = atom->tau_[i][0][2];
	double tauyy = atom->tau_[i][1][1];
	double tauyz = atom->tau_[i][1][2];
	double tauzz = atom->tau_[i][2][2];
	double tauyx = tauxy;
	double tauzx = tauxz;
	double tauzy = tauyz;
	double epsilonxx = lambda_dot*0.5*sqrt(3/J2)*tauxx;
	double epsilonxy = lambda_dot*0.5*sqrt(3/J2)*tauxy;
	double epsilonxz = lambda_dot*0.5*sqrt(3/J2)*tauxz;
	double epsilonyy = lambda_dot*0.5*sqrt(3/J2)*tauyy;
	double epsilonyz = lambda_dot*0.5*sqrt(3/J2)*tauyz;
	double epsilonzz = lambda_dot*0.5*sqrt(3/J2)*tauzz;
	double epsilonyx = epsilonxy;
	double epsilonzx = epsilonxz;
	double epsilonzy = epsilonyz;
    double dt = update->dt;
	double T = Tref;
	double epsilonDoteqv = epsilonxx*epsilonxx + epsilonxy*epsilonxy+ epsilonxz*epsilonxz \
			 + epsilonyx*epsilonyx + epsilonyy*epsilonyy+ epsilonyz*epsilonyz \
			 + epsilonzx*epsilonzx + epsilonzy*epsilonzy+ epsilonzz*epsilonzz;
	epsilonDoteqv = sqrt(2./3.*epsilonDoteqv);
	epsiloneqv += epsilonDoteqv * dt;
	if (epsilonDoteqv > epsilonDot0){
	Y0 = (sigma0 + B*pow(epsiloneqv,n))*(1 + C*log(epsilonDoteqv/epsilonDot0))*(1 - pow((T - Tref)/(Tmelt - Tref),m));}
	else{
	Y0 = (sigma0 + B*pow(epsiloneqv,n))*(1)*(1 - (T - Tref)/(Tmelt - Tref));}

}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsLowVel::compute_no_drho(int eflag, int vflag) {
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;
  double dt = update->dt;
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

  find_sigma(eflag, vflag);

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

		#include"performSolidMomentumEqn_lowVel_no_drho_3D.h"

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

		#include"performSolidMomentumEqn_lowVel_no_drho_2D.h"

	  }
  }
  // communicate densities
  // comm->forward_comm_pair(this);

  // if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- 
 * check whether on the yield surface for plastic deformation
 * check whether f<0 for elastic deformation
 * ---------------------------------------------------------------------*/

bool PairSPHSolidMechanicsLowVel::check_f() {

	// compute value of f
	// name of the criterion is Von Mises 
	fyield = sqrt(3*J2) - Y0;
//	printf("Y0 = %f, J2 = %f, fyield = %f\n", Y0, J2, fyield);

	elastic_flag = (fyield < 0.0);
	return elastic_flag;
}

/* ---------------------------------------------------------------------- 
 * compute lambda_dot for each particle
 * ---------------------------------------------------------------------*/

void PairSPHSolidMechanicsLowVel::find_lambda_dot(int i) {

	double dt = update->dt;
	lambda_dot = fyield/(3*G*dt);

	// if(!elastic_flag) printf("id = %d, lambda_dot = %f\n", i, lambda_dot);
}

/* ---------------------------------------------------------------------- 
 * compute total stress  
 * ---------------------------------------------------------------------*/

void PairSPHSolidMechanicsLowVel::find_sigma(int eflag, int vflag) {

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
  double ***dSigma = atom->dSigma_;
  int *type = atom->type;

  double ***epsilon = atom->epsilon_;
  double ***sigma = atom->sigma_;
  double *p = atom->p_;
  double ***artStress_ = atom->artStress_;
  double ***epsilonBar = atom->epsilonBar_;
  double ***R = atom->R_;
  double ***tau = atom->tau_;

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

		int iter = 0;
  		lambda_dot = 0;
		double dt = update->dt;
		tmp_sigma[0] = sigma[i][0][0]; // xx
		tmp_sigma[1] = sigma[i][0][1]; // xy
		tmp_sigma[2] = sigma[i][0][2]; // xz
		tmp_sigma[3] = sigma[i][1][0]; // yx 
		tmp_sigma[4] = sigma[i][1][1]; // yy
		tmp_sigma[5] = sigma[i][1][2]; // yz
		tmp_sigma[6] = sigma[i][2][0]; // zx
		tmp_sigma[7] = sigma[i][2][1]; // zy
		tmp_sigma[8] = sigma[i][2][2]; // zz
		do
		{
			iter++;
			#include"computeStrainStressLowVelocity3D.h"
			// dsigma_i is known
			// tmp_sigma_i = sigma_i_0 + dsigma_i * dt
			// tmp_sigma_i to test whether elastic or plastic
			// check_f(); with J2 that is based on tmp_sigma_i;
			// till we find the correct dsigma_i
   			check_f();  
			/*
			if(!elastic_flag) 
				if(comm->me == 0) error->warning(FLERR,"Plastic deformation occured!\n");
			 if(comm->me == 0) printf("Time: %d, Iteration: %d\n", update->ntimestep, iter);
			 */
			 if (fyield < TOLERANCE)
				break;
		}	
  		while((iter<iterMax) && (!elastic_flag));

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
		int iter = 0;
  		lambda_dot = 0;
		double dt = update->dt;
		tmp_sigma[0] = sigma[i][0][0]; // xx
		tmp_sigma[1] = sigma[i][0][1]; // xy
		tmp_sigma[2] = sigma[i][1][0]; // yx 
		tmp_sigma[3] = sigma[i][1][1]; // yy
		do
		{
			iter++;
			#include"computeStrainStressLowVelocity2D.h"
			// dsigma_i is known
			// tmp_sigma_i = sigma_i_0 + dsigma_i * dt
			// tmp_sigma_i to test whether elastic or plastic
			// check_f(); with J2 that is based on tmp_sigma_i;
			// till we find the correct dsigma_i
   			check_f();  
			//if(comm->me == 0) {
			//	printf("plastic: %d, fyield = %f, pId = %d\n", elastic_flag, fyield, comm->me);
			//}
			/*
			if(!elastic_flag) 
				if(comm->me == 0) error->warning(FLERR,"Plastic deformation occured!\n");
			if(!elastic_flag && comm->me == 0) printf("Time: %d, Iteration: %d\n", update->ntimestep, iter);
			*/ 
			if (fyield < TOLERANCE)
				break;
		}	
  		while((iter<iterMax) || (!elastic_flag));
	  }

  }
  // communicate densities
  // comm->forward_comm_pair(this);

  // if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsLowVel::compute_drho(int eflag, int vflag) {

	/*
	 *
	 * elastic = false;
	 * lambda_dot = 0;
	 * do
	 * {
	 * 	// find sigma
	 * 	find_sigma();
	 *  // check f
	 *  check_f();
	 *  if (f < TOLERANCE)
	 *  break;
	 * }
	 * while((iter<iterMax) && (!elastic_flag))
	 *
	 */
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;
  double dt = update->dt;
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

  find_sigma(eflag,vflag);

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

		#include"performSolidMomentumEqn_lowVel_with_drho_3D.h"

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

		#include"performSolidMomentumEqn_lowVel_with_drho_2D.h"

	  }
  }
  // communicate densities
  // comm->forward_comm_pair(this);

  // if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsLowVel::allocate() {
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

void PairSPHSolidMechanicsLowVel::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/solidMechanics/lowVel");

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsLowVel::coeff(int narg, char **arg) {
  if (narg != 21)
    error->all(FLERR,"Incorrect number of args for pair_style sph/solidMechanics/lowVel coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double viscosity_one = force->numeric(FLERR, arg[2]);
  double beta_one = force->numeric(FLERR, arg[3]);
  double cut_one = force->numeric(FLERR, arg[4]);
  h_ = force->numeric(FLERR, arg[4+1]);

  G = force->numeric(FLERR, arg[5+1]);
  K = force->numeric(FLERR, arg[6+1]);
  c_ = force->numeric(FLERR, arg[7+1]);
  sigma0 = force->numeric(FLERR, arg[8+1]);
  B = force->numeric(FLERR, arg[9+1]);
  C = force->numeric(FLERR, arg[10+1]);
  epsilonDot0 = force->numeric(FLERR, arg[11+1]);
  Tref = force->numeric(FLERR, arg[12+1]);
  Tmelt = force->numeric(FLERR, arg[13+1]);
  n = force->numeric(FLERR, arg[14+1]);
  m = force->numeric(FLERR, arg[15+1]);

  if(strcmp(arg[16+1],"yes") == 0) 
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

  eps_ = force->numeric(FLERR, arg[17+1]);
  deltap_ = force->numeric(FLERR, arg[18+1]);
  epsXSPH_ = force->numeric(FLERR, arg[19+1]);

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
    error->all(FLERR,"Incorrect args for pair sph/solidMechanics/lowVel coefficients");

}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHSolidMechanicsLowVel::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair sph/solidMechanics/lowVel coeffs are not set");
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHSolidMechanicsLowVel::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}

/* ---------------------------------------------------------------------- */
