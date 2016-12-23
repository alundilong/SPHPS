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
#include "pair_sph_taitwater_modified_morris.h"
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
#include "sph_kernel_lucy.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterModifiedMorris::PairSPHTaitwaterModifiedMorris(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

  first = 1;
}

/* ---------------------------------------------------------------------- */

PairSPHTaitwaterModifiedMorris::~PairSPHTaitwaterModifiedMorris() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(viscosity);
  }
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterModifiedMorris::init_style() {
// need a full neighbor list
	int irequest = neighbor->request(this);
	neighbor->requests[irequest]->half = 0;
	neighbor->requests[irequest]->full = 1;
}

/* ---------------------------------------------------------------------- */

void PairSPHTaitwaterModifiedMorris::compute(int eflag, int vflag) {

  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih, ihsq, velx, vely, velz;
  double rsq, tmp, wf, wfd, delVdotDelR, mu, deltaE;

  if (eflag || vflag)
    ev_setup(eflag, vflag);
  else
    evflag = vflag_fdotr = 0;

  double **v = atom->vest;
  double **x = atom->x;
  double **f = atom->f;
 // double *rho = atom->rho;
  double *phi = atom->phi;
  double *rmass = atom->rmass;
  double *de = atom->de;
  double *drho = atom->drho;
  double *dphi = atom->dphi;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  // check consistency of pair coefficients

  if (first) {
    for (i = 1; i <= atom->ntypes; i++) {
      for (j = 1; i <= atom->ntypes; i++) {
        if (cutsq[i][j] > 1.e-32) {
          if (!setflag[i][i] || !setflag[j][j]) {
            if (comm->me == 0) {
              printf(
                  "SPH particle types %d and %d interact with cutoff=%g, but not all of their single particle properties are set.\n",
                  i, j, sqrt(cutsq[i][j]));
            }
          }
        }
      }
    }
    first = 0;
  }

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;


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

    if (itype != liquid_type)
	{
		imass = rmass_liquid;
		//rho_i = rho0_one;
  	}
	else
	{	
		imass = rmass[i];
		//rho_i = phi[i];
	}

    // compute pressure of atom i with Tait EOS
    
	//tmp = rho_i / rho0_one;
    //fi = B_one * (fi * fi * tmp - 1.0) / (rho_i * rho_i);
    
    tmp = phi[i] / phi0_one;
    fi = tmp * tmp * tmp;
	fi = b_one * (fi * fi * tmp - 1.0) / (phi[i] * phi[i]);

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      
	  if (jtype != liquid_type) 
	  {
  		 jmass = rmass_liquid;
		 //rho_j = rho0_one;
  	  }
	  else 
	{	
		jmass = rmass[j];
		//rho_j = phi[j];
	}


      if (rsq < cutsq[itype][jtype]) {
        
		//h = h_;  //cut[itype][jtype];
		h = 0.5*cut[itype][jtype];
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

        // compute pressure  of atom j with Tait EOS
        //tmp = rho_j / rho0_one;
        //fj = B_one * (fj * fj * tmp - 1.0) / (rho_j * rho_j);

        tmp = phi[j] / phi0_one;
        fj = tmp * tmp * tmp;
        fj = b_one * (fj * fj * tmp - 1.0) / (phi[j] * phi[j]);

		//printf("fj = %f, B = %f, rho[j] = %f, tmp = %f\n", fj, B[jtype], rho[j], tmp);

        velx=vxtmp - v[j][0];
        vely=vytmp - v[j][1];
        velz=vztmp - v[j][2];

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * velx + dely * vely + delz * velz;

        // Morris Viscosity (Morris, 1996)

        fvisc = 2 * viscosity[itype][jtype] / (phi[i] * phi[j]);

        fvisc *= imass * jmass * wfd;

        fpair = -imass * jmass * (fi + fj) * wfd;
        deltaE = -0.5 *(fpair * delVdotDelR + fvisc * (velx*velx + vely*vely + velz*velz));

	//	if((itype == 1 and jtype == 2) or (itype == 2 and jtype == 1))
        {//printf("r = %g, cut = %g, h = %g, s = %g, i = %d, j = %d, visc = %g\n", r, cut[itype][jtype], h, s, itype, jtype, viscosity[itype][jtype]);
		//printf("fj = %g, fi = %g, wfd = %g, fvisc = %g, delVR = %g, visc = %g\n", fj, fi, wfd, fvisc, delVdotDelR, viscosity[itype][jtype]);
		//printf("fj = %g, fi = %g, phii = %g, phij = %g, phi0 = %g, tmp = %g\n", fj, fi, phi[i], phi[j], phi0_one, tmp);
		}
		f[i][0] += delx * fpair + velx * fvisc;
        f[i][1] += dely * fpair + vely * fvisc;
        f[i][2] += delz * fpair + velz * fvisc;

        // and change in phi(density)
       	double den = jmass * delVdotDelR * wfd;
	    if (itype != liquid_type) 
	    {
        	dphi[i] += den;
  	    }
	    else 
    	{	
        	dphi[i] += den;
        	drho[i] += den;
	    }
        
		/*
		if((itype == 1 and jtype == 3) or (itype == 3 and jtype == 1))
		{
		printf("dphi = %g\n", dphi[i]); 
		printf("phi0_one = %g, phi_i = %g, phi_j = %g, b_one = %g, c_one = %g, fpair = %g, fi = %g, fj = %g, fvisc = %g, deltaE = %g, de = %g\n", phi0_one, phi[i], phi[j], b_one, c_one, fpair, fi, fj, fvisc, deltaE, de[i]);
		}*/
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

void PairSPHTaitwaterModifiedMorris::allocate() {
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

void PairSPHTaitwaterModifiedMorris::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/taitwater/modified/morris");
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHTaitwaterModifiedMorris::coeff(int narg, char **arg) {
  if (narg != 9)
    error->all(FLERR,
        "Incorrect args for pair_style sph/taitwater/modified/morris coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  phi0_one = force->numeric(FLERR,arg[2]);
  c_one = force->numeric(FLERR,arg[3]);
  double viscosity_one = force->numeric(FLERR,arg[4]);
  double cut_one = force->numeric(FLERR,arg[5]);
  double h1 = force->numeric(FLERR, arg[6]);
  liquid_type = force->numeric(FLERR, arg[7]);
  rmass_liquid = force->numeric(FLERR, arg[8]);
  b_one = c_one * c_one * phi0_one / 7.0;
  h_ = h1;
	//printf("rmass_liquid = %g\n", rmass_liquid);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
	//printf("i = %d, ilo = %d, ihi = %d, phi0i = %g, ntypes = %d\n", i, ilo, ihi, phi0[i]);
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      viscosity[i][j] = viscosity_one;
      //printf("setting cut[%d][%d] = %f\n", i, j, cut_one);
      cut[i][j] = cut_one;

      setflag[i][j] = 1;

      //cut[j][i] = cut[i][j];
      //viscosity[j][i] = viscosity[i][j];
      //setflag[j][i] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHTaitwaterModifiedMorris::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
    error->all(FLERR,"Not all pair sph/taitwater coeffs are set");
  }

  cut[j][i] = cut[i][j];
  viscosity[j][i] = viscosity[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHTaitwaterModifiedMorris::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}
