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
#include "pair_sph_solidcontact_wall.h"
#include "sph_kernel_quintic.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "neigh_list.h"
#include "domain.h"
#include <iostream>
#include "math_extra_liggghts.h"
#include "sph_kernel_cubicspline.h"
#include "sph_kernel_Bspline.h"
#include "sph_kernel_cubicspline2D.h"
#include "neighbor.h"
#include "neigh_request.h"

using namespace LAMMPS_NS;
using namespace MathExtraLiggghts;

#define EPSILON 1.0e-12
#define SMALL 1.0e-15
#define pi M_PI

/* ---------------------------------------------------------------------- */

PairSPHSolidContactWall::PairSPHSolidContactWall(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
}

/* ---------------------------------------------------------------------- */

PairSPHSolidContactWall::~PairSPHSolidContactWall() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidContactWall::init_style() {
// need a full neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
//	neighbor->requests[irequest]->ghost = 1;
}


/* ---------------------------------------------------------------------- */

void PairSPHSolidContactWall::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz;
  double vxtmp, vytmp, vztmp, delvx, delvy, delvz, lambda1, lambda2;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double imass, jmass, ih;
  double rsq, wfd, wij;

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *rho = atom->rho;
  double **cg = atom->colorgradient;
  const int ndim = domain->dimension;
  double eij[ndim];
  int *type = atom->type;
  double *radius = atom->radius;
  double *csound = atom->csound;
  double **v = atom->v;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

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

    vxtmp = v[i][0];
    vytmp = v[i][1];
    vztmp = v[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    imass = rmass[i];

    /*double abscgi;
    if (ndim == 3) {
      abscgi = sqrt(cg[i][0]*cg[i][0] +
		    cg[i][1]*cg[i][1] +
		    cg[i][2]*cg[i][2]);
    } else {
      abscgi = sqrt(cg[i][0]*cg[i][0] +
		    cg[i][1]*cg[i][1]);
    }*/
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      delvx = vxtmp - v[j][0];
      delvy = vytmp - v[j][1];
      delvz = vztmp - v[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = rmass[j];

	  double maxr = radius[i]; //max(radius[i], radius[j]);
	  double rav = (radius[i] + radius[j]);
	  double rsqc = maxr*maxr + rav*rav;
      if (rsq < cutsq[itype][jtype] && rsq < rsqc) {
		 // printf("============================\n");
	  double reff = (radius[i]*radius[j])/(radius[i]+radius[j]);
        // h = cut[itype][jtype];
        ih = 1.0 / h;
		double rij = sqrt(rsq);
		double rInv = 1./rij;
		double s = rij/h;
        if (ndim == 3) {
			//wfd = SPH_KERNEL_NS::sph_kernel_cubicspline_der(s, h, ih)*rInv;
			wij = SPH_KERNEL_NS::sph_kernel_cubicspline(s, h, ih);
        } else {
			//wfd = SPH_KERNEL_NS::sph_kernel_cubicspline2d_der(s, h, ih)*rInv;
			// printf("s = %f, h = %f, ih = %f\n", s, h, ih);
			wij = SPH_KERNEL_NS::sph_kernel_cubicspline2d(s, h, ih);
        }

	eij[0] = delx/sqrt(rsq); 
	eij[1] = dely/sqrt(rsq);    
	if (ndim==3) {
	  eij[2] = delz/sqrt(rsq);
	}

	// Reference:
	// Songwon Se., Oakkey Min, Axisymmeritric SPH simulation of elasto-plastic contact in the low velocity impact
	// Computer Physics Communications, 2006, 175
	// Eq. 48
	// ni will be computed == cg[i]/abs(cg[i])
	// nj will be computed == cg[j]/abs(cg[j])
	if (ndim==2) {
		double ni[2], nj[2];
		double abscgi, abscgj;
		abscgi = sqrt(cg[i][0]*cg[i][0] + cg[i][1]*cg[i][1]);
		abscgj = sqrt(cg[j][0]*cg[j][0] + cg[j][1]*cg[j][1]);
		ni[0] = -cg[i][0]/(abscgi+SMALL);
		ni[1] = -cg[i][1]/(abscgi+SMALL);
		nj[0] = -cg[j][0]/(abscgj+SMALL);
		nj[1] = -cg[j][1]/(abscgj+SMALL);
		// printf("nix = %f, niy = %f, niz = %f, njx = %f, njy = %f, njz  = %f\n", ni[0], ni[1], ni[2], nj[0], nj[1], nj[2]);
		double nij[2];
		double absnij, absni, absnj;
		nij[0] = ni[0] - nj[0];
		nij[1] = ni[1] - nj[1];
		absnij = sqrt(nij[0]*nij[0] + nij[1]*nij[1]);
		absni = sqrt(ni[0]*ni[0] + ni[1]*ni[1]);
		absnj = sqrt(nj[0]*nj[0] + nj[1]*nj[1]);
		double nav[2];
		// theta = acos(-ni & nj);
		//double theta = acos((-ni[0]*nj[0]-ni[1]*nj[1])/(absni*absnj+SMALL));
		double sin = -ni[0]*nj[1]-ni[1]*nj[0];
		double cos = -ni[0]*nj[0]+ni[1]*nj[1];
		double theta = atan2(sin,cos);
		if(theta < 0) theta += 2*M_PI;
		//double theta = acos((-ni[0]*nj[0]-ni[1]*nj[1])/(absni*absnj+SMALL));
		if(theta < thetac) 
		{
			double test1 = delx*ni[0] + dely*ni[1];
			double test2 = -delx*nj[0] - dely*nj[1];
			double test3 = delx*nij[0] + dely*nij[1];
			test3 /= (absnij+SMALL);
			if(test1 > max(test2, test3)) 
			{
				nav[0] = ni[0];
				nav[1] = ni[1];
			}
			else if(test2 > max(test1, test3)) {
				nav[0] = -nj[0];
				nav[1] = -nj[1];
			}
			else
			{
				nav[0] = nij[0]/(absnij+SMALL);
				nav[1] = nij[1]/(absnij+SMALL);
			}
		} else
		{
				nav[0] = nij[0]/(absnij+SMALL);
				nav[1] = nij[1]/(absnij+SMALL);
		}

		double rijDotnav = delx*nav[0] + dely*nav[1];
		double pnav =  radius[i] + radius[j] - fabs(rijDotnav);
		double pnavDot = delvx*nav[0] + delvy*nav[1];
		double rijav =  radius[i] + radius[j] - fabs(rij);

		const double Vj = jmass / rho[j];
		if (jtype == 3){
		lambda1 = rho[i]*csound[i];
		lambda2 = Ei/d0;
		} else {
		lambda1 = rho[j]*csound[j];
		lambda2 = Ej/d0;
		}
		double coeff = rijDotnav/(rij+SMALL);
		double area = 2*radius[i]; //sqrt(4*rijav*radius[i]-rijav*rijav); //2*pi*radius[i]*0.01;//*h; // ????? what is h
		area *= fabs(coeff);
		//double area = 2*pi*radius[i]*h; // ????? what is h
		
//		printf("area = %f\n", area);
		double fx = (lambda1*pnavDot + lambda2*pnav)*nav[0]*wij*area*Vj;
		double fy = (lambda1*pnavDot + lambda2*pnav)*nav[1]*wij*area*Vj;
		f[i][0] += fx;
		f[i][1] += fy;

		atom->de[i] -= fx*vxtmp + fy*vytmp;

//		if(itype == 1) printf("i = %d, fix = %g, fiy = %g, lambda1 = %g, lambda2 = %g, area = %g, rijav = %g, pn = %g, pnDot = %g, theta = %f, wij = %f\n", i, fx, fy, lambda1, lambda2, area, rijav, pnav, pnavDot, theta, wij);
		//if(itype == 1) printf("fix = %g, fiy = %g, pnDot = %g, delvx = %f, delvy = %f, navx = %f, navy = %f\n", fx, fy, pnavDot, delvx, delvy, nav[0], nav[1]);
		
/*        if (newton_pair || j < nlocal) {
			f[j][0] -= (lambda1*pnavDot + lambda2*pnav)*nav[0]*wij*area*Vj;
			f[j][1] -= (lambda1*pnavDot + lambda2*pnav)*nav[1]*wij*area*Vj;
        }
*/		
		
      } else {
		double ni[3], nj[3];
		double abscgi, abscgj;
		abscgi = sqrt(cg[i][0]*cg[i][0] + cg[i][1]*cg[i][1] + cg[i][2]*cg[i][2]);
		abscgj = sqrt(cg[j][0]*cg[j][0] + cg[j][1]*cg[j][1] + cg[j][2]*cg[j][2]);
		ni[0] = -cg[i][0]/(abscgi+SMALL);
		ni[1] = -cg[i][1]/(abscgi+SMALL);
		ni[2] = -cg[i][2]/(abscgi+SMALL);
		nj[0] = -cg[j][0]/(abscgj+SMALL);
		nj[1] = -cg[j][1]/(abscgj+SMALL);
		nj[2] = -cg[j][2]/(abscgj+SMALL);
		double nij[3];
		double absnij, absni, absnj;
		nij[0] = ni[0] - nj[0];
		nij[1] = ni[1] - nj[1];
		nij[2] = ni[2] - nj[2];
		absnij = sqrt(nij[0]*nij[0] + nij[1]*nij[1] + nij[2]*nij[2]);
		absni = sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2]);
		absnj = sqrt(nj[0]*nj[0] + nj[1]*nj[1] + nj[2]*nj[2]);
		double nav[3];
		// theta = acos(-ni & nj);
		//double theta = acos((-ni[0]*nj[0]-ni[1]*nj[1]-ni[2]*nj[2])/(absni*absnj+SMALL));
		double minusnicrossnj = sqrt((-ni[1]*nj[2]+ni[2]*nj[1])+(-ni[0]*nj[2]+ni[2]*nj[0])+(-ni[0]*nj[1]+ni[1]*nj[0]));
		double sin = (minusnicrossnj)/(absni*absnj+SMALL);
		double cos = (-ni[0]*nj[0]-ni[1]*nj[1]-ni[2]*nj[2])/(absni*absnj+SMALL);
		double theta = atan2(sin, cos);
		if(theta < 0) theta += 2*M_PI;
		// printf("nix = %f, niy = %f, niz = %f, njx = %f, njy = %f, njz = %f, cgx = %f, cgy = %f, cgz = %f, theta = %f\n", ni[0], ni[1], ni[2], nj[0], nj[1], nj[2], cg[0], cg[1], cg[2], theta);
		if(theta < thetac) 
		{
			double test1 = delx*ni[0] + dely*ni[1] + delz*ni[2];
			double test2 = -delx*nj[0] - dely*nj[1] - delz*nj[2];
			double test3 = delx*nij[0] + dely*nij[1] + delz*nij[2];
			test3 /= (absnij+SMALL);
			if(test1 > max(test2, test3)) 
			{
				nav[0] = ni[0];
				nav[1] = ni[1];
				nav[2] = ni[2];
			}
			else if(test2 > max(test1, test3)) {
				nav[0] = -nj[0];
				nav[1] = -nj[1];
				nav[2] = -nj[2];
			}
			else
			{
				nav[0] = nij[0]/(absnij+SMALL);
				nav[1] = nij[1]/(absnij+SMALL);
				nav[2] = nij[2]/(absnij+SMALL);
			}
		} else
		{
				nav[0] = nij[0]/(absnij+SMALL);
				nav[1] = nij[1]/(absnij+SMALL);
				nav[2] = nij[2]/(absnij+SMALL);
		}

		double rijDotnav = delx*nav[0] + dely*nav[1] + delz*nav[2];
		double rijav =  radius[i] + radius[j] - fabs(rij);
		double pnav =  radius[i] + radius[j] - fabs(rijDotnav);
		double pnavDot = delvx*nav[0] + delvy*nav[1] + delvz*nav[2];

		const double Vj = jmass / rho[j];
		if (jtype == 3){
		lambda1 = rho[i]*csound[i];
		lambda2 = Ei/d0;
		} else {
		lambda1 = rho[j]*csound[j];
		lambda2 = Ej/d0;
		}
		//double area = 2*pi*radius[i]*h; // ????? what is h
		//double area = 2*L*sqrt(pnav*reff); //for cylinder with length L
		double coeff = rijDotnav/(rij+SMALL);
		//double area = pi*radius[i]*radius[i]; //pi*rijav*radius[i]; //for sphere
		double area = 4*radius[i]*radius[i]; //for cylinder with length L
		area *= fabs(coeff);

		double fx = (lambda1*pnavDot + lambda2*pnav)*nav[0]*wij*area*Vj;
		double fy = (lambda1*pnavDot + lambda2*pnav)*nav[1]*wij*area*Vj;
		double fz = (lambda1*pnavDot + lambda2*pnav)*nav[2]*wij*area*Vj;
//		if(itype == 1) printf("i = %d, fix = %g, fiy = %g, lambda1 = %g, lambda2 = %g, area = %g, rijav = %g, pn = %g, pnDot = %g, theta = %f, wij = %f\n", i, fx, fy, lambda1, lambda2, area, rijav, pnav, pnavDot, theta, wij);
//		if(itype == 1) printf("fix = %g, fiy = %g, pnDot = %g, delvx = %f, delvy = %f, navx = %f, navy = %f\n", fx, fy, pnavDot, delvx, delvy, nav[0], nav[1]);
		f[i][0] += fx;
		f[i][1] += fy;
		f[i][2] += fz;

		atom->de[i] -= fx*vxtmp + fy*vytmp + fz*vztmp;

/*        if (newton_pair || j < nlocal) {
			f[j][0] -= (lambda1*pnavDot + lambda2*pnav)*nav[0]*wij*area*Vj;
			f[j][1] -= (lambda1*pnavDot + lambda2*pnav)*nav[1]*wij*area*Vj;
			f[j][2] -= (lambda1*pnavDot + lambda2*pnav)*nav[2]*wij*area*Vj;
        }
*/		

    }
  }
}
}
}

/* ----------------------------------------------------------------------
   allocate all arrays
   ------------------------------------------------------------------------- */

void PairSPHSolidContactWall::allocate() {
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

void PairSPHSolidContactWall::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
	       "Illegal number of setting arguments for pair_style sph/solidcontactwall");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void PairSPHSolidContactWall::coeff(int narg, char **arg) {
  if (narg != 3+5)
    error->all(FLERR,"Incorrect number of args for pair_style sph/solidcontactwall coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double cut_one   = force->numeric(FLERR, arg[2]);
  h = force->numeric(FLERR, arg[3]);
  Ei = force->numeric(FLERR, arg[4]);
  Ej = force->numeric(FLERR, arg[5]);
  d0 = force->numeric(FLERR, arg[6]);
  thetac = force->numeric(FLERR, arg[7]);
  thetac = thetac*pi/180;
 
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

double PairSPHSolidContactWall::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error->all(FLERR,"All pair sph/solidcontactwall coeffs are not set");
  }

  cut[j][i] = cut[i][j];
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHSolidContactWall::single(int i, int j, int itype, int jtype,
				     double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;
  return 0.0;
}

