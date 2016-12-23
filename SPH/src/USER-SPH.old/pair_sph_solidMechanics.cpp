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
#include "pair_sph_solidMechanics.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "update.h"
#include "error.h"
#include "domain.h"
#include "math_extra_liggghts.h"
#include "sph_kernel_cubicspline.h"
#include "sph_kernel_Bspline.h"
#include "sph_kernel_cubicspline2D.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSPHSolidMechanics::PairSPHSolidMechanics(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

//  comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

PairSPHSolidMechanics::~PairSPHSolidMechanics() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(viscosity);
  }
}


/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanics::compute(int eflag, int vflag) {
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **v = atom->vest;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *rmass = atom->rmass;
  double *de = atom->de;
  double *e = atom->e;
  double *drho = atom->drho;
  int *type = atom->type;

  double ***epsilon_ = atom->epsilon_;
  double ***sigma_ = atom->sigma_;
  double ***epsilonBar_ = atom->epsilonBar_;
  double ***R_ = atom->R_;
  double ***tau_ = atom->tau_;
  double *p_ = atom->p_;

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double rhoi, rhoj;
  double fix, fiy, fiz;

  // loop over neighbors of my atoms

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

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = rmass[j];
	  rhoj = rho[j];

      if (rsq < cutsq[itype][jtype]) {
        h = cut[itype][jtype];
        ih = 1. / h;

		double r = sqrt(rsq);
		double rInv = 1./r;
		double s = r/h;

        if (domain->dimension == 3) {
		  wfd = SPH_KERNEL_NS::sph_kernel_cubicspline_der(s, h, ih)*rInv;
		  //wfd = SPH_KERNEL_NS::sph_kernel_Bspline_der(s, h, ih)*rInv;
        } else {
		  wfd = SPH_KERNEL_NS::sph_kernel_cubicspline2d_der(s, h, ih)*rInv;
        }

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * (vxtmp - v[j][0]) + dely * (vytmp - v[j][1])
            + delz * (vztmp - v[j][2]);
		double delvx = vxtmp - v[j][0];
		double delvy = vytmp - v[j][1];
		double delvz = vztmp - v[j][2];

        // artificial viscosity (Monaghan 1992)
        if (delVdotDelR < 0.) {
          cj = c_;
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = (-viscosity[itype][jtype] * ((ci + cj)/2) * mu + mu * mu) / ((rhoi + rhoj)/2);
        } else {
          fvisc = 0.;
        }

		/*
		 * Larry D. Libersky etc.
		 * High Strain Lagrangian Hydrodynamics
		 * A Three-Dimensional SPH Code for Dynamic Material Response
		 * Journal of Computational Physics 109, 67-75 (1993)
		 * Equation 32 
		 * "a minus sign is due to different definition of stress tension in (34)"
		 */
		fix = imass*jmass*wfd*((sigma_[i][0][0]/rhoi/rhoi + sigma_[j][0][0]/rhoj/rhoj)*delx \
							  + (sigma_[i][0][1]/rhoi/rhoi + sigma_[j][0][1]/rhoj/rhoj)*dely \
							  + (sigma_[i][0][2]/rhoi/rhoi + sigma_[j][0][2]/rhoj/rhoj)*delz);
		fiy = imass*jmass*wfd*((sigma_[i][1][0]/rhoi/rhoi + sigma_[j][1][0]/rhoj/rhoj)*delx \
							  + (sigma_[i][1][1]/rhoi/rhoi + sigma_[j][1][1]/rhoj/rhoj)*dely \
							  + (sigma_[i][1][2]/rhoi/rhoi + sigma_[j][1][2]/rhoj/rhoj)*delz);
		fiz = imass*jmass*wfd*((sigma_[i][2][0]/rhoi/rhoi + sigma_[j][2][0]/rhoj/rhoj)*delx \
							  + (sigma_[i][2][1]/rhoi/rhoi + sigma_[j][2][1]/rhoj/rhoj)*dely \
							  + (sigma_[i][2][2]/rhoi/rhoi + sigma_[j][2][2]/rhoj/rhoj)*delz);


		double PIJ = fvisc*0; // depends ....
		fix += imass*jmass*wfd*PIJ*delx;
		fiy += imass*jmass*wfd*PIJ*dely;
		fiz += imass*jmass*wfd*PIJ*delz;
		
		double eshear = 0.5*imass*jmass*(p_[i]/rhoi/rhoi + p_[j]/rhoj/rhoj + PIJ)*wfd*(delVdotDelR);
        	   eshear += imass*(tau_[i][0][0]*epsilon_[i][0][0] + tau_[i][0][1]*epsilon_[i][0][1] + tau_[i][0][2]*epsilon_[i][0][2] \
							  + tau_[i][1][0]*epsilon_[i][1][0] + tau_[i][1][1]*epsilon_[i][1][1] + tau_[i][1][2]*epsilon_[i][1][2] \
							  + tau_[i][2][0]*epsilon_[i][2][0] + tau_[i][2][1]*epsilon_[i][2][1] + tau_[i][2][2]*epsilon_[i][2][2])/rhoi;

		deltaE = eshear;

		f[i][0] += fix; 
		f[i][1] += fiy; 
		f[i][2] += fiz; 

        // and change in density
        drho[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        de[i] += deltaE;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fix;
          f[j][1] -= fiy;
          f[j][2] -= fiz;
          de[j] += deltaE;
          drho[j] += imass * delVdotDelR * wfd;
        }

         if (evflag)
           ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
      }
    }
  }

//  comm->forward_comm_pair(this);

   if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanics::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(viscosity, n + 1, n + 1, "pair:viscosity");
}

/* ----------------------------------------------------------------------
 global settings
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanics::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/solidMechanics");

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanics::coeff(int narg, char **arg) {
  if (narg != 5)
    error->all(FLERR,"Incorrect number of args for pair_style sph/solidMechanics coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double viscosity_one = force->numeric(FLERR, arg[2]);
  double cut_one = force->numeric(FLERR, arg[3]);
  c_ = force->numeric(FLERR, arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      viscosity[i][j] = viscosity_one;
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair sph/solidMechanics coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHSolidMechanics::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair sph/solidMechanics coeffs are not set");
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHSolidMechanics::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}
