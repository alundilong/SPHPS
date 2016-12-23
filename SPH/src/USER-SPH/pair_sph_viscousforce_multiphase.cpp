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
#include "pair_sph_viscousforce_multiphase.h"
#include "sph_kernel_quintic.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "neigh_list.h"
#include "domain.h"
#include <iostream>

using namespace LAMMPS_NS;

#define EPSILON 1.0e-12

/* ---------------------------------------------------------------------- */

PairSPHViscousForceMultiphase::PairSPHViscousForceMultiphase(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

PairSPHViscousForceMultiphase::~PairSPHViscousForceMultiphase() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHViscousForceMultiphase::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double imass, jmass, h, ih;
  double rsq, wfd;

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  double *rmass = atom->rmass;
  double *rho = atom->rho;
  double *viscosity_ = atom->viscosity_;
  const int ndim = domain->dimension;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double iviscosity, jviscosity;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms and do surface tension

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    itype = type[i];

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    imass = rmass[i];
	iviscosity = viscosity_[i];

    // TODO: FixMe
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = rmass[j];
	  jviscosity = viscosity_[j];

      if (rsq < cutsq[itype][jtype]) {
        h = cut[itype][jtype];
        ih = 1.0 / h;
        if (ndim == 3) {
	  wfd = sph_dw_quintic3d(sqrt(rsq)*ih);
          wfd = wfd * ih * ih * ih * ih;
        } else {
	  wfd = sph_dw_quintic2d(sqrt(rsq)*ih);
          wfd = wfd * ih * ih * ih;
        }

		const double Vi = imass / rho[i];
		const double Vj = jmass / rho[j];
		const double rij = sqrt(rsq);	    

		/*
		 * v = eij vij eij + vij 
		 */
		double vis_eff = 2*iviscosity*jviscosity/(iviscosity + jviscosity);
		double vec[3];
		double delvx = v[i][0] - v[j][0];
		double delvy = v[i][1] - v[j][1];
		double delvz = v[i][2] - v[j][2];
		vec[0] = ((delx*delx + rsq)*delvx + delx*dely*delvy + delx*delz)/rsq;
		vec[1] = (delx*dely*delvx + (dely*dely + rsq)*delvy + dely*delz)/rsq;
		vec[2] = (delx*delz*delvx + dely*delz*delvy + (delz*delz + rsq))/rsq;
		f[i][0] += vis_eff*(Vi*Vi + Vj*Vj)*wfd/rij*vec[0];
		f[i][1] += vis_eff*(Vi*Vi + Vj*Vj)*wfd/rij*vec[1];
		if (ndim==3) {
			f[i][2] += vis_eff*(Vi*Vi + Vj*Vj)*wfd/rij*vec[2];
		}

		if (newton_pair || j < nlocal) {
		  f[j][0] -= vis_eff*(Vi*Vi + Vj*Vj)*wfd/rij*vec[0];
		  f[j][1] -= vis_eff*(Vi*Vi + Vj*Vj)*wfd/rij*vec[1];
		  if (ndim==3) {
			f[j][2] -= vis_eff*(Vi*Vi + Vj*Vj)*wfd/rij*vec[2];
		}
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
   ------------------------------------------------------------------------- */

void PairSPHViscousForceMultiphase::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
}

/* ----------------------------------------------------------------------
   global settings
   ------------------------------------------------------------------------- */

void PairSPHViscousForceMultiphase::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
	       "Illegal number of setting arguments for pair_style sph/viscousforce/multiphase");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void PairSPHViscousForceMultiphase::coeff(int narg, char **arg) {
  if (narg != 3)
    error->all(FLERR,"Incorrect number of args for pair_style sph/viscousforce/multiphase coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double cut_one   = force->numeric(FLERR, arg[2]);
 
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
   ------------------------------------------------------------------------- */

double PairSPHViscousForceMultiphase::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error->all(FLERR,"All pair sph/viscousforce/multiphase coeffs are not set");
  }

  cut[j][i] = cut[i][j];
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHViscousForceMultiphase::single(int i, int j, int itype, int jtype,
				     double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;
  return 0.0;
}

