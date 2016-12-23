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
#include "pair_sph_dragmodel.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "sph_kernel_cubicspline.h"
#include "sph_kernel_Bspline.h"
#include "sph_kernel_cubicspline2D.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSPHDragModel::PairSPHDragModel(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

PairSPHDragModel::~PairSPHDragModel() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHDragModel::init_style() {
// need a full neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
//	neighbor->requests[irequest]->ghost = 1;
}

/* ---------------------------------------------------------------------- */

void PairSPHDragModel::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fdrag, h, ih, ihsq;
  double rsq, wf, wfd, delVdotDelR, mu, deltaE;

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
  double dim = domain->dimension; 
//  double *drho = atom->drho;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

//  double lambda_ = 2.5e-10;
//  double ds = 75e-06;
//  double mu_ = 1.002e-03;
  double Cc = 1. + (2.*lambda_/ds_)*(1.257+0.4*exp(-(1.1*ds_/(2.*lambda_))));
  double beta = 18.*mu_/(ds_*ds_*Cc);
  
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

//    ci = c[itype][jtype]; //sqrt(0.4*e[i]/imass); // speed of sound with heat capacity ratio gamma=1.4

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = rmass[j];

      // if (rsq < cutsq[itype][jtype]) {
      if (rsq < SPH_KERNEL_NS::sph_kernel_cubicspline2d_cut()*h_) {
        h = h_;  //cut[itype][jtype];
        ih = 1. / h;
        
		double r = sqrt(rsq);
		double rInv = 1./r;
		double s = r/h; 
	
        if (domain->dimension == 3) {
		  wf = SPH_KERNEL_NS::sph_kernel_cubicspline(s, h, ih);
		  wfd = SPH_KERNEL_NS::sph_kernel_cubicspline_der(s, h, ih)*rInv;
        } else {
		  wf = SPH_KERNEL_NS::sph_kernel_cubicspline2d(s, h, ih);
		  wfd = SPH_KERNEL_NS::sph_kernel_cubicspline2d_der(s, h, ih)*rInv;
        }

        // dot product of velocity delta and distance vector
		double delvx = vxtmp - v[j][0];
		double delvy = vytmp - v[j][1];
		double delvz = vztmp - v[j][2];
        delVdotDelR = delx * (vxtmp - v[j][0]) + dely * (vytmp - v[j][1])
            + delz * (vztmp - v[j][2]);

        // drag force beta (Monaghan Kocharyan 1995)
        mu = delVdotDelR / (rsq + 0.001 * h * h);
        
		fdrag = -(1.0/dim) * jmass * beta * mu / (rho[i] * rho[j]);

        // total pair force & thermal energy increment
        fpair = imass * fdrag * wf;  
        deltaE = -fpair * delVdotDelR;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;


        // change in thermal energy
        de[i] += deltaE;
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHDragModel::allocate() {
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

void PairSPHDragModel::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/dragmodel");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHDragModel::coeff(int narg, char **arg) {
  if (narg != 7)
    error->all(FLERR,"Incorrect number of args for pair_style sph/dragmodel coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double cut_one = force->numeric(FLERR,arg[2]);
  h_ = force->numeric(FLERR, arg[3]);
  ds_ = force->numeric(FLERR,arg[4]);
  mu_ = force->numeric(FLERR,arg[5]);
  lambda_ = force->numeric(FLERR,arg[6]);

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
    error->all(FLERR,"Incorrect args for pair sph/dragmodel coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHDragModel::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair sph/dragmodel coeffs are not set");
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHDragModel::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}
