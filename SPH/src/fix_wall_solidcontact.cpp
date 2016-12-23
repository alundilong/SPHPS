/*----------------------------------------------------------------------
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
#include "fix_wall_solidcontact.h"
#include "atom.h"
#include "error.h"
#include "stdlib.h"
#include "sph_kernel_quintic.h"
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
#include "input.h"
#include "variable.h"
#include "modify.h"
#include "respa.h"
#include "update.h"
#include "lattice.h"
//#include "math_const.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathExtraLiggghts;

#define EPSILON 1.0e-12
#define SMALL 1.0e-15
#define pi M_PI

enum{XLO=0,XHI=1,YLO=2,YHI=3,ZLO=4,ZHI=5};
enum{NONE=0,EDGE,CONSTANT,VARIABLE};
enum{EQUAL};

/* ---------------------------------------------------------------------- */

FixWallSolidContact::FixWallSolidContact(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) 
{
  // Read material properties
  coeff(narg, arg);
	
  scalar_flag = 1;
  vector_flag = 1;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;

  // parse args

  nwall = 0;
  int scaleflag = 1;
  int pbcflag = 0;

  for (int i = 0; i < 6; i++) {xstr[i] =  NULL; vxstr[i] = NULL;}

  int iarg = 3 + 5;
  while (iarg < narg) {
    if ((strcmp(arg[iarg],"xlo") == 0) || (strcmp(arg[iarg],"xhi") == 0) ||
        (strcmp(arg[iarg],"ylo") == 0) || (strcmp(arg[iarg],"yhi") == 0) ||
        (strcmp(arg[iarg],"zlo") == 0) || (strcmp(arg[iarg],"zhi") == 0)) {
      // if (iarg+4 > narg) error->all(FLERR,"Illegal fix wall command");

      int newwall;
      if (strcmp(arg[iarg],"xlo") == 0) newwall = XLO;
      else if (strcmp(arg[iarg],"xhi") == 0) newwall = XHI;
      else if (strcmp(arg[iarg],"ylo") == 0) newwall = YLO;
      else if (strcmp(arg[iarg],"yhi") == 0) newwall = YHI;
      else if (strcmp(arg[iarg],"zlo") == 0) newwall = ZLO;
      else if (strcmp(arg[iarg],"zhi") == 0) newwall = ZHI;

      for (int m = 0; m < nwall; m++)
        if (newwall == wallwhich[m])
          error->all(FLERR,"Wall defined twice in fix wall command");

      wallwhich[nwall] = newwall;
      if (strcmp(arg[iarg+1],"EDGE") == 0) {
        xstyle[nwall] = EDGE;
        int dim = wallwhich[nwall] / 2;
        int side = wallwhich[nwall] % 2;
        if (side == 0) coord0[nwall] = domain->boxlo[dim];
        else coord0[nwall] = domain->boxhi[dim];
      } else if (strstr(arg[iarg+1],"v_") == arg[iarg+1]) {
        xstyle[nwall] = VARIABLE;
        int n = strlen(&arg[iarg+1][2]) + 1;
        xstr[nwall] = new char[n];
        strcpy(xstr[nwall],&arg[iarg+1][2]);
      } else {
        xstyle[nwall] = CONSTANT;
        coord0[nwall] = force->numeric(FLERR,arg[iarg+1]);
      }

      if (strcmp(arg[iarg+2],"NULL") == 0) {
        vxstyle[nwall] = NULL;
      } else if (strstr(arg[iarg+2],"v_") == arg[iarg+2]) {
        vxstyle[nwall] = VARIABLE;
        int n = strlen(&arg[iarg+2][2]) + 1;
        vxstr[nwall] = new char[n];
        strcpy(vxstr[nwall],&arg[iarg+2][2]);
      } else {
        vxstyle[nwall] = CONSTANT;
      }

      iarg += 3; //5;
	  nwall++;

    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix wall command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix wall command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"pbc") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix wall command");
      if (strcmp(arg[iarg+1],"yes") == 0) pbcflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) pbcflag = 0;
      else error->all(FLERR,"Illegal fix wall command");
      iarg += 2;
    } else 
	{
		error->all(FLERR,"Illegal fix wall command");
	}
  }

  size_vector = nwall;

  // error checks

  if (nwall == 0) error->all(FLERR,"Illegal fix wall command");

  for (int m = 0; m < nwall; m++)
    if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->dimension == 2)
      error->all(FLERR,"Cannot use fix wall zlo/zhi for a 2d simulation");

  if (!pbcflag) {
    for (int m = 0; m < nwall; m++) {
      if ((wallwhich[m] == XLO || wallwhich[m] == XHI) && domain->xperiodic)
        error->all(FLERR,"Cannot use fix wall in periodic dimension");
      if ((wallwhich[m] == YLO || wallwhich[m] == YHI) && domain->yperiodic)
        error->all(FLERR,"Cannot use fix wall in periodic dimension");
      if ((wallwhich[m] == ZLO || wallwhich[m] == ZHI) && domain->zperiodic)
        error->all(FLERR,"Cannot use fix wall in periodic dimension");
    }
  }

  // scale factors for wall position for CONSTANT and VARIABLE walls

  int flag = 0;
  for (int m = 0; m < nwall; m++)
    if (xstyle[m] != EDGE) flag = 1;

  if (flag) {
    if (scaleflag) {
      xscale = domain->lattice->xlattice;
      yscale = domain->lattice->ylattice;
      zscale = domain->lattice->zlattice;
    }
    else xscale = yscale = zscale = 1.0;

    for (int m = 0; m < nwall; m++) {
      if (xstyle[m] != CONSTANT) continue;
      if (wallwhich[m] < YLO) coord0[m] *= xscale;
      else if (wallwhich[m] < ZLO) coord0[m] *= yscale;
      else coord0[m] *= zscale;
    }
  }

  // set xflag if any wall positions are variable
  // set varflag if any wall positions or parameters are variable

  varflag = xflag = 0;
  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) xflag = 1;
    if (xflag) varflag = 1;
  }

  eflag = 0;
  for (int m = 0; m <= nwall; m++) ewall[m] = 0.0;

}


/* ----------------------------------------------------------------------
   interaction of all particles in group with a wall
   m = index of wall coeffs
   which = xlo,xhi,ylo,yhi,zlo,zhi
   error if any particle is on or behind wall
------------------------------------------------------------------------- */

void FixWallSolidContact::wall_particle(int m, int which, double coord, double vel)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, delta;
  double vxtmp, vytmp, vztmp, delvx, delvy, delvz;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double imass, jmass, ih;
  double rsq, wfd, wij;

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

  int *mask = atom->mask;

  int dim = which / 2;
  int side = which % 2;
  if (side == 0) side = -1;
  double nj[3];
  nj[0] = nj[1] = nj[2] = 0;
  nj[dim] = side/(MathExtraLiggghts::abs(side)+SMALL);

  int onflag = 0;
  cutoff[m] = h;
  double vwall[3];
  vwall[0] = vwall[1] = vwall[2] = 0.0;
  vwall[dim] = vel;
	
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (side < 0) delta = x[i][dim] - coord;
      else delta = coord - x[i][dim];
      if (delta >= cutoff[m]) continue;
      if (delta <= 0.0) {
        onflag = 1;
        continue;
      }
     ih = 1.0 / h;
	 double rij = delta;
	 double rInv = 1./(rij+SMALL);
	 double s = rij*ih;
     if (ndim == 3) {
	 	//wfd = SPH_KERNEL_NS::sph_kernel_cubicspline_der(s, h, ih)*rInv;
	 	wij = SPH_KERNEL_NS::sph_kernel_cubicspline(s, h, ih);
     } else {
	 	//wfd = SPH_KERNEL_NS::sph_kernel_cubicspline2d_der(s, h, ih)*rInv;
	 	// printf("s = %f, h = %f, ih = %f\n", s, h, ih);
	 	wij = SPH_KERNEL_NS::sph_kernel_cubicspline2d(s, h, ih);
     }

	// printf("delta = %f, side = %d, cutoff = %f, wij = %f, vwall = %f\n", delta, side, cutoff[m], wij, vwall[dim]);
      imass = rmass[i];
      double del_x[3];
	  del_x[0] = del_x[1] = del_x[2] = 0.0;
      del_x[dim] = delta;

      delx = del_x[0];
      dely = del_x[1];
      delz = del_x[2];
      delvx = vwall[0]- v[i][0];
      delvy = vwall[1]- v[i][1];
      delvz = vwall[2]- v[i][2];
      double ni[3];
      double abscgi;
      abscgi = sqrt(cg[i][0]*cg[i][0] + cg[i][1]*cg[i][1] + cg[i][2]*cg[i][2]);
      ni[0] = -cg[i][0]/(abscgi+SMALL);
      ni[1] = -cg[i][1]/(abscgi+SMALL);
      ni[2] = -cg[i][2]/(abscgi+SMALL);
      double nij[3];
      double absnij, absni, absnj;
      nij[0] = ni[0] - nj[0];
      nij[1] = ni[1] - nj[1];
      nij[2] = ni[2] - nj[2];
      absnij = sqrt(nij[0]*nij[0] + nij[1]*nij[1] + nij[2]*nij[2]);
      absni = sqrt(ni[0]*ni[0] + ni[1]*ni[1] + ni[2]*ni[2]);
      absnj = sqrt(nj[0]*nj[0] + nj[1]*nj[1] + ni[2]*ni[2]);
      double nav[3];
      // theta = acos(-ni & nj);
      //double theta = acos((-ni[0]*nj[0]-ni[1]*nj[1])/(absni*absnj+SMALL));
		double minusnicrossnj = sqrt((-ni[1]*nj[2]+ni[2]*nj[1])+(-ni[0]*nj[2]+ni[2]*nj[0])+(-ni[0]*nj[1]+ni[1]*nj[0]));
		double sin = (minusnicrossnj)/(absni*absnj+SMALL);
		double cos = (-ni[0]*nj[0]-ni[1]*nj[1]-ni[2]*nj[2])/(absni*absnj+SMALL);
		double theta = atan2(sin, cos);
      if(theta < 0) theta += 2*M_PI;
      //double theta = acos((-ni[0]*nj[0]-ni[1]*nj[1])/(absni*absnj+SMALL));
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
	  double rijav =  radius[i] - fabs(rij);
	  double pnav =  radius[i] - fabs(rijDotnav);
	  double pnavDot = delvx*nav[0] + delvy*nav[1] + delvz*nav[2];
      
      const double Vi = imass / rho[i];
      double lambda1 = rho[i]*csound[i];
      double lambda2 = Ei/d0;
      double c_oeff = rijDotnav/(rij+SMALL);
      //double area = 2*radius[i]; //sqrt(4*rijav*radius[i]-rijav*rijav); //2*pi*radius[i]*0.01;//*h; // ????? what is h
      double area = 2*sqrt(radius[i]*radius[i]-delta*delta); //sqrt(4*rijav*radius[i]-rijav*rijav); //2*pi*radius[i]*0.01;//*h; // ????? what is h
      area *= fabs(c_oeff);
      //f[i][dim] += (lambda1*pnavDot + lambda2*pnav)*nav[dim]*wij*area*Vj;
      //ewall[0] += r6inv*(coeff3[m]*r6inv - coeff4[m]) - offset[m];
      //ewall[m+1] += fwall;
		double fx = (lambda1*pnavDot + lambda2*pnav)*nav[0]*wij*area*Vi;
		double fy = (lambda1*pnavDot + lambda2*pnav)*nav[1]*wij*area*Vi;
		double fz = (lambda1*pnavDot + lambda2*pnav)*nav[2]*wij*area*Vi;
		f[i][0] += fx;
		f[i][1] += fy;
		f[i][2] += fz;

		atom->de[i] -= fx*v[i][0] + fy*v[i][1] + fz*v[i][2];

  if (onflag) error->one(FLERR,"Particle on or inside fix wall surface");
}

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   ------------------------------------------------------------------------- */

void FixWallSolidContact::coeff(int narg, char **arg) {
  if (narg < 12)
    error->all(FLERR,"Incorrect number of args for fix wall/solidcontact coefficients");

  h = force->numeric(FLERR, arg[3]);
  Ei = force->numeric(FLERR, arg[4]);
  Ej = force->numeric(FLERR, arg[5]);
  d0 = force->numeric(FLERR, arg[6]);
  thetac = force->numeric(FLERR, arg[7]);
  thetac = thetac*pi/180;

}


int FixWallSolidContact::setmask()
{
  int mask = 0;

  mask |= POST_FORCE;

  mask |= THERMO_ENERGY;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallSolidContact::init()
{
  dt = update->dt;

  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) {
      xindex[m] = input->variable->find(xstr[m]);
      if (xindex[m] < 0)
        error->all(FLERR,"Variable name for fix wall does not exist");
      if (!input->variable->equalstyle(xindex[m]))
        error->all(FLERR,"Variable for fix wall is invalid style");
    }

    if (vxstyle[m] == VARIABLE) {
      vxindex[m] = input->variable->find(vxstr[m]);
      if (vxindex[m] < 0)
        error->all(FLERR,"Variable name for fix wall does not exist");
      if (!input->variable->equalstyle(vxindex[m]))
        error->all(FLERR,"Variable for fix wall is invalid style");
    }
  }


  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;
}

/* ---------------------------------------------------------------------- */

void FixWallSolidContact::setup(int vflag)
{
  if (strstr(update->integrate_style,"verlet")) {
     post_force(vflag);
  } else {
    ((Respa *) update->integrate)->copy_flevel_f(nlevels_respa-1);
    post_force_respa(vflag,nlevels_respa-1,0);
    ((Respa *) update->integrate)->copy_f_flevel(nlevels_respa-1);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallSolidContact::min_setup(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   only called if fldflag set, in place of post_force
------------------------------------------------------------------------- */

void FixWallSolidContact::pre_force(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallSolidContact::post_force(int vflag)
{
  eflag = 0;
  for (int m = 0; m <= nwall; m++) ewall[m] = 0.0;

  // coord = current position of wall
  // evaluate variables if necessary, wrap with clear/add
  // for epsilon/sigma variables need to re-invoke precompute()

  if (varflag) modify->clearstep_compute();

  double coord;
  double vel;
  for (int m = 0; m < nwall; m++) {
    if (xstyle[m] == VARIABLE) {
      coord = input->variable->compute_equal(xindex[m]);
      if (wallwhich[m] < YLO) coord *= xscale;
      else if (wallwhich[m] < ZLO) coord *= yscale;
      else coord *= zscale;
    } else coord = coord0[m];

    if (vxstyle[m] == VARIABLE) {
      vel = input->variable->compute_equal(vxindex[m]);
    } else vel = vel0[m];

    wall_particle(m,wallwhich[m],coord, vel);
  }

  if (varflag) modify->addstep_compute(update->ntimestep + 1);
}

/* ---------------------------------------------------------------------- */

void FixWallSolidContact::post_force_respa(int vflag, int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallSolidContact::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of wall interaction
------------------------------------------------------------------------- */

double FixWallSolidContact::compute_scalar()
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall,ewall_all,nwall+1,MPI_DOUBLE,MPI_SUM,world);
    eflag = 1;
  }
  return ewall_all[0];
}

/* ----------------------------------------------------------------------
   components of force on wall
------------------------------------------------------------------------- */

double FixWallSolidContact::compute_vector(int n)
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall,ewall_all,nwall+1,MPI_DOUBLE,MPI_SUM,world);
    eflag = 1;
  }
  
  return ewall_all[n+1];

}
