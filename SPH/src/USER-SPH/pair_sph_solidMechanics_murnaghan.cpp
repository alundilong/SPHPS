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
#include "pair_sph_solidMechanics_murnaghan.h"
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
#define det3By3(A) \
	A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) \
   -A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) \
   +A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]) 

/* ---------------------------------------------------------------------- */

PairSPHSolidMechanicsMurnaghan::PairSPHSolidMechanicsMurnaghan(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;

  update_drho_flag_ = 1;

}

/* ---------------------------------------------------------------------- */

PairSPHSolidMechanicsMurnaghan::~PairSPHSolidMechanicsMurnaghan() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(viscosity);
  }
}


/* ---------------------------------------------------------------------- */

double PairSPHSolidMechanicsMurnaghan::determinant(double f[6][6],int rank)
{
  int pr, j, p, q, t;
  double c[6];
  double d = 0;
  double b[6][6];
  if(rank == 2)
  {
    d = 0;
    d = (f[0][0]*f[1][1]) - (f[0][1]*f[1][0]);
    return(d);
   }
  else
  {
    for(j = 0; j < rank; j++)
    {        
      int r = 0, s = 0;
      for(p = 0; p < rank; p++)
        {
          for(q = 0; q < rank; q++)
            {
              if(p!=0 && q != j)
              {
                b[r][s] = f[p][q];
                s++;
                if(s > rank - 2)
                 {
                   r++;
                   s = 0;
                 }
               }
             }
         }

     for(t = 0, pr = 1; t < (1 + j); t++)
     	pr = (-1)*pr;
     	c[j] = pr*determinant(b, rank - 1);
     }
     for(j = 0,d = 0; j < rank; j++)
     {
       d = d + (f[0][j]*c[j]);
     }
     return(d);
   }
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::solve6By6(double &x1, double& x2, double &x3, double &x4, double &x5, double &x6, double A[6][6], double b[6]) {

	double detA1, detA2, detA3, detA4, detA5, detA6, detA;

	double A1[6][6];
	double A2[6][6];
	double A3[6][6];
	double A4[6][6];
	double A5[6][6];
	double A6[6][6];
	for(int i = 0; i < 6; i++)
	{
		for(int j = 0; j < 6; j++)
		{
				A1[i][j] = A[i][j];
				A2[i][j] = A[i][j];
				A3[i][j] = A[i][j];
				A4[i][j] = A[i][j];
				A5[i][j] = A[i][j];
				A6[i][j] = A[i][j];
		}
	}

	A1[0][0] = b[0];
	A1[1][0] = b[1];
	A1[2][0] = b[2];
	A1[3][0] = b[3];
	A1[4][0] = b[4];
	A1[5][0] = b[5];

	A2[0][1] = b[0];
	A2[1][1] = b[1];
	A2[2][1] = b[2];
	A2[3][1] = b[3];
	A2[4][1] = b[4];
	A2[5][1] = b[5];

	A3[0][2] = b[0];
	A3[1][2] = b[1];
	A3[2][2] = b[2];
	A3[3][2] = b[3];
	A3[4][2] = b[4];
	A3[5][2] = b[5];

	A4[0][3] = b[0];
	A4[1][3] = b[1];
	A4[2][3] = b[2];
	A4[3][3] = b[3];
	A4[4][3] = b[4];
	A4[5][3] = b[5];

	A5[0][4] = b[0];
	A5[1][4] = b[1];
	A5[2][4] = b[2];
	A5[3][4] = b[3];
	A5[4][4] = b[4];
	A5[5][4] = b[5];

	A6[0][5] = b[0];
	A6[1][5] = b[1];
	A6[2][5] = b[2];
	A6[3][5] = b[3];
	A6[4][5] = b[4];
	A6[5][5] = b[5];

	detA = determinant(A, 6);
	detA1 = determinant(A1, 6);
	detA2 = determinant(A2, 6);
	detA3 = determinant(A3, 6);
	detA4 = determinant(A4, 6);
	detA5 = determinant(A5, 6);
	detA6 = determinant(A6, 6);

	x1 = detA1/detA;
	x2 = detA2/detA;
	x3 = detA3/detA;
	x4 = detA4/detA;
	x5 = detA5/detA;
	x6 = detA6/detA;
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::computeStrainAndShearStress() {

  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;

  double **v = atom->vest;
  double **x = atom->x;
  double **f = atom->f;
  double *rho = atom->rho;
  double *rmass = atom->rmass;
  double *de = atom->de;
  double *e = atom->e;
  int *type = atom->type;

  double ***epsilon_ = atom->epsilon_;
  double ***sigma_ = atom->sigma_;
  double ***epsilonBar_ = atom->epsilonBar_;
  double ***R_ = atom->R_;
  double ***tau_ = atom->tau_;
  double *p_ = atom->p_;

  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;
  double rhoi, rhoj;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  double A[6][6];
  double b[6];

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

	double oldTauxx = tau_[i][0][0];
	double oldTauxy = tau_[i][0][1];
	double oldTauxz = tau_[i][0][2];
	double oldTauyx = tau_[i][1][0];
	double oldTauyy = tau_[i][1][1];
	double oldTauyz = tau_[i][1][2];
	double oldTauzx = tau_[i][2][0];
	double oldTauzy = tau_[i][2][1];
	double oldTauzz = tau_[i][2][2];

//	/*
	epsilon_[i][0][0] = epsilonBar_[i][0][0] = R_[i][0][0] = sigma_[i][0][0] = tau_[i][0][0] = 0.0;
	epsilon_[i][0][1] = epsilonBar_[i][0][1] = R_[i][0][1] = sigma_[i][0][1] = tau_[i][0][1] = 0.0;
	epsilon_[i][0][2] = epsilonBar_[i][0][2] = R_[i][0][2] = sigma_[i][0][2] = tau_[i][0][2] = 0.0;
	epsilon_[i][1][0] = epsilonBar_[i][1][0] = R_[i][1][0] = sigma_[i][1][0] = tau_[i][1][0] = 0.0;
	epsilon_[i][1][1] = epsilonBar_[i][1][1] = R_[i][1][1] = sigma_[i][1][1] = tau_[i][1][1] = 0.0;
	epsilon_[i][1][2] = epsilonBar_[i][1][2] = R_[i][1][2] = sigma_[i][1][2] = tau_[i][1][2] = 0.0;
	epsilon_[i][2][0] = epsilonBar_[i][2][0] = R_[i][2][0] = sigma_[i][2][0] = tau_[i][2][0] = 0.0;
	epsilon_[i][2][1] = epsilonBar_[i][2][1] = R_[i][2][1] = sigma_[i][2][1] = tau_[i][2][1] = 0.0;
	epsilon_[i][2][2] = epsilonBar_[i][2][2] = R_[i][2][2] = sigma_[i][2][2] = tau_[i][2][2] = 0.0;
//	*/

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

		// - compute strains

		double vjix = v[j][0] - v[i][0];
		double vjiy = v[j][1] - v[i][1];
		double vjiz = v[j][2] - v[i][2];
		
		epsilon_[i][0][0] += jmass/rhoj*vjix*wfd*delx;
		epsilon_[i][1][1] += jmass/rhoj*vjiy*wfd*dely;
		epsilon_[i][2][2] += jmass/rhoj*vjiz*wfd*delz;
		epsilon_[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely + vjiy*delx);
		epsilon_[i][0][2] += 0.5*jmass/rhoj*wfd*(vjix*delz + vjiz*delx);
		epsilon_[i][1][2] += 0.5*jmass/rhoj*wfd*(vjiy*delz + vjiz*dely);

		R_[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely - vjiy*delx);
		R_[i][0][2] += 0.5*jmass/rhoj*wfd*(vjix*delz - vjiz*delx);
		R_[i][1][2] += 0.5*jmass/rhoj*wfd*(vjiy*delz - vjiz*dely);
		
    }

  }
	epsilon_[i][1][0] = epsilon_[i][0][1];
	epsilon_[i][2][0] = epsilon_[i][0][2];
	epsilon_[i][2][1] = epsilon_[i][1][2];

	R_[i][1][0] = -1.0*R_[i][0][1];
	R_[i][2][0] = -1.0*R_[i][0][2];
	R_[i][2][1] = -1.0*R_[i][1][2];

	double averaged = 1./3.*(epsilon_[i][0][0] + epsilon_[i][1][1] + epsilon_[i][2][2]);

	epsilonBar_[i][0][0] = epsilon_[i][0][0] - averaged;
	epsilonBar_[i][1][1] = epsilon_[i][1][1] - averaged;
	epsilonBar_[i][2][2] = epsilon_[i][2][2] - averaged;
	epsilonBar_[i][0][1] = epsilon_[i][0][1];
	epsilonBar_[i][0][2] = epsilon_[i][0][2];
	epsilonBar_[i][1][2] = epsilon_[i][1][2];
	epsilonBar_[i][1][0] = epsilon_[i][0][1];
	epsilonBar_[i][2][0] = epsilon_[i][0][2];
	epsilonBar_[i][2][1] = epsilon_[i][1][2];

	double & tauxx = tau_[i][0][0];
	double & tauyy = tau_[i][1][1];
	double & tauzz = tau_[i][2][2];
	double & tauxy = tau_[i][0][1];
	double & tauxz = tau_[i][0][2];
	double & tauyz = tau_[i][1][2];
	double & tauyx = tau_[i][1][0];
	double & tauzx = tau_[i][2][0];
	double & tauzy = tau_[i][2][1];

	double dt = update->dt;
	A[0][0] = 1.0/dt; 		A[0][1] = 0.0;    			A[0][2] = 0.0;    			A[0][3] = -2.0*R_[i][0][1]; A[0][4] = -2.0*R_[i][0][2]; 	A[0][5] = 0.0;
	A[1][0] = 0.0;    		A[1][1] = 1.0/dt; 			A[1][2] = 0.0;    			A[1][3] = 2.0*R_[i][0][1];  A[1][4] =  0.0; 			  	A[1][5] = -2.0*R_[i][1][2];
	A[2][0] = 0.0;    		A[2][1] = 0.0;    			A[2][2] = 1.0/dt; 			A[2][3] = 0.0; 			  	A[2][4] =  2.0*R_[i][0][2];  	A[2][5] = 2.0*R_[i][1][2];
 	A[3][0] = R_[i][0][1];  A[3][1] = -1.0*R_[i][0][1]; A[3][2] = 0.0;    			A[3][3] = 1.0/dt; 			A[3][4] = -1.0*R_[i][1][2]; 	A[3][5] = -1.0*R_[i][0][2];
	A[4][0] = R_[i][0][2];  A[4][1] = 0.0; 				A[4][2] = -1.0*R_[i][0][2]; A[4][3] = R_[i][1][2]; 		A[4][4] =  1.0/dt;  			A[4][5] = -1.0*R_[i][0][1];
	A[5][0] = 0.0;  		A[5][1] = R_[i][1][2]; 		A[5][2] = -1.0*R_[i][1][2]; A[5][3] = R_[i][0][2]; 		A[5][4] =  R_[i][0][1];  		A[5][5] = 1.0/dt;

	b[0] = 2*G_*epsilonBar_[i][0][0] + oldTauxx/dt;
	b[1] = 2*G_*epsilonBar_[i][1][1] + oldTauyy/dt;
	b[2] = 2*G_*epsilonBar_[i][2][2] + oldTauzz/dt;
	b[3] = 2*G_*epsilonBar_[i][0][1] + oldTauxy/dt;
	b[4] = 2*G_*epsilonBar_[i][0][2] + oldTauxz/dt;
	b[5] = 2*G_*epsilonBar_[i][1][2] + oldTauyz/dt;

	solve6By6(tauxx, tauyy, tauzz, tauxy, tauxz, tauyz, A, b);

	double Jsqr = \
			   tau_[i][0][0]*tau_[i][0][0] + tau_[i][0][1]*tau_[i][0][1] + tau_[i][0][2]*tau_[i][0][2] \
			 + tau_[i][1][0]*tau_[i][1][0] + tau_[i][1][1]*tau_[i][1][1] + tau_[i][1][2]*tau_[i][1][2] \
			 + tau_[i][2][0]*tau_[i][2][0] + tau_[i][2][1]*tau_[i][2][1] + tau_[i][2][2]*tau_[i][2][2];

	Jsqr *= 0.5;

	if(Jsqr > Y0_*Y0_)
	{
		//double scale = sqrt(Y0_/3./Jsqr);
		double scale = Y0_*Y0_/3./Jsqr;
		tauxx *= scale;
		tauyy *= scale;
		tauzz *= scale;
		tauxy *= scale;
		tauxz *= scale;
		tauyz *= scale;
	}

	tauyx = tauxy;
	tauzx = tauxz;
	tauzy = tauyz;

	/*
	 * EOS 
	 * MIe-Gruneisen Equation
	 * Liu's Book, Page 297
	 */ 
	double eta = rho[i]/rho0_ - 1.0;
	a0_ = rho0_*Cs_*Cs_;
	b0_ = a0_*(1. + 2.*(S_ - 1.));
	c0_ = a0_*(2*(S_ - 1.) + 3*(S_ - 1.)*(S_ - 1.));
	double pH = eta > 0 ? (a0_*eta + b0_*eta*eta + c0_*eta*eta*eta) : a0_*eta;
//	p_[i] =  (1.0 - 0.5*Gamma_*eta)*pH + Gamma_*rho[i]*e[i]; 

	p_[i] = Cs_*Cs_*(rho[i] - rho0_);

	sigma_[i][0][0] = -p_[i] + tau_[i][0][0]; sigma_[i][0][1] = tau_[i][0][1]; sigma_[i][0][2] = tau_[i][0][2];
	sigma_[i][1][0] = tau_[i][1][0]; sigma_[i][1][1] = -p_[i] + tau_[i][1][1]; sigma_[i][1][2] = tau_[i][1][2];
	sigma_[i][2][0] = tau_[i][2][0]; sigma_[i][2][1] = tau_[i][2][1]; sigma_[i][2][2] = -p_[i] + tau_[i][2][2];

}

}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::compute(int eflag, int vflag) {

	if(update_drho_flag_) compute_drho(eflag, vflag);
	else compute_no_drho(eflag, vflag);

}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::compute_no_drho(int eflag, int vflag) {
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;

  computeStrainAndShearStress();

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
//  double *drho = atom->drho;
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

		//- the second term is disabled
        if (delVdotDelR < 0.) {
          cj = c_;
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = (-viscosity[itype][jtype] * ((ci + cj)/2) * mu + mu * mu*0) / ((rhoi + rhoj)/2);
		  //printf("fvisc = %.3g, viscosity = %.3g, ci = %.3g, cj = %.3g, mu = %.3g, rhoi = %.3g, rhoj = %3.g\n", fvisc, viscosity[itype][jtype], ci, cj, mu, rhoi, rhoj);
        } else {
          fvisc = 0.;
        }

		/*
		 * Monaghan (1983 paper, shock simulation by the particle method SPH)
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


		double PIJ = fvisc; // depends ....
		fix -= imass*jmass*wfd*PIJ*delx;
		fiy -= imass*jmass*wfd*PIJ*dely;
		fiz -= imass*jmass*wfd*PIJ*delz;

		/*
		// Need double check the units
		fpair = imass*jmass*(p_[i]/rhoi/rhoi + p_[j]/rhoj/rhoj + fvisc)*wfd; // compute pair force ?
		
        double eshear = imass*( tau_[i][0][0]*epsilon_[i][0][0] + tau_[i][0][1]*epsilon_[i][0][1] + tau_[i][0][2]*epsilon_[i][0][2] \
							  + tau_[i][1][0]*epsilon_[i][1][0] + tau_[i][1][1]*epsilon_[i][1][1] + tau_[i][1][2]*epsilon_[i][1][2] \
							  + tau_[i][2][0]*epsilon_[i][2][0] + tau_[i][2][1]*epsilon_[i][2][1] + tau_[i][2][2]*epsilon_[i][2][2])/rhoi;

		double eshear = imass*jmass*wfd*((sigma_[i][0][0]*delvx + sigma_[i][0][1]*delvy + sigma_[i][0][2]*delvz)*delx \
										+(sigma_[i][1][0]*delvx + sigma_[i][1][1]*delvy + sigma_[i][1][2]*delvz)*dely \
										+(sigma_[i][2][0]*delvx + sigma_[i][2][1]*delvy + sigma_[i][2][2]*delvz)*delz)/rhoi/rhoi;
		eshear += 0.5*imass*jmass*PIJ*wfd*(delVdotDelR);
		*/

		double eshear = 0.5*imass*jmass*(p_[i]/rhoi/rhoi + p_[j]/rhoj/rhoj + PIJ)*wfd*(delVdotDelR);
        	   eshear += imass*(tau_[i][0][0]*epsilon_[i][0][0] + tau_[i][0][1]*epsilon_[i][0][1] + tau_[i][0][2]*epsilon_[i][0][2] \
							  + tau_[i][1][0]*epsilon_[i][1][0] + tau_[i][1][1]*epsilon_[i][1][1] + tau_[i][1][2]*epsilon_[i][1][2] \
							  + tau_[i][2][0]*epsilon_[i][2][0] + tau_[i][2][1]*epsilon_[i][2][1] + tau_[i][2][2]*epsilon_[i][2][2])/rhoi;

		deltaE = eshear;

		f[i][0] += fix; 
		f[i][1] += fiy; 
		f[i][2] += fiz; 

        // and change in density
//        drho[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        de[i] += deltaE;

        if (newton_pair || j < nlocal) {
          f[j][0] -= fix;
          f[j][1] -= fiy;
          f[j][2] -= fiz;
          de[j] += deltaE;
//          drho[j] += imass * delVdotDelR * wfd;
        }

         if (evflag)
           ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
      }
    }
  }

   if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ---------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::compute_drho(int eflag, int vflag) {
	
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;

  int *ilist, *jlist, *numneigh, **firstneigh;
  double vxtmp, vytmp, vztmp, imass, jmass, fi, fj, fvisc, h, ih;
  double rsq, wfd, delVdotDelR, mu, deltaE, ci, cj;

  computeStrainAndShearStress();

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

		//- the second term is disabled
        if (delVdotDelR < 0.) {
          cj = c_;
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = (-viscosity[itype][jtype] * ((ci + cj)/2) * mu + mu * mu*0) / ((rhoi + rhoj)/2);
		  //printf("fvisc = %.3g, viscosity = %.3g, ci = %.3g, cj = %.3g, mu = %.3g, rhoi = %.3g, rhoj = %3.g\n", fvisc, viscosity[itype][jtype], ci, cj, mu, rhoi, rhoj);
        } else {
          fvisc = 0.;
        }

		/*
		 * Monaghan (1983 paper, shock simulation by the particle method SPH)
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


		double PIJ = fvisc; // depends ....
		fix -= imass*jmass*wfd*PIJ*delx;
		fiy -= imass*jmass*wfd*PIJ*dely;
		fiz -= imass*jmass*wfd*PIJ*delz;

		/*
		// Need double check the units
		fpair = imass*jmass*(p_[i]/rhoi/rhoi + p_[j]/rhoj/rhoj + fvisc)*wfd; // compute pair force ?
		
        double eshear = imass*( tau_[i][0][0]*epsilon_[i][0][0] + tau_[i][0][1]*epsilon_[i][0][1] + tau_[i][0][2]*epsilon_[i][0][2] \
							  + tau_[i][1][0]*epsilon_[i][1][0] + tau_[i][1][1]*epsilon_[i][1][1] + tau_[i][1][2]*epsilon_[i][1][2] \
							  + tau_[i][2][0]*epsilon_[i][2][0] + tau_[i][2][1]*epsilon_[i][2][1] + tau_[i][2][2]*epsilon_[i][2][2])/rhoi;

		double eshear = imass*jmass*wfd*((sigma_[i][0][0]*delvx + sigma_[i][0][1]*delvy + sigma_[i][0][2]*delvz)*delx \
										+(sigma_[i][1][0]*delvx + sigma_[i][1][1]*delvy + sigma_[i][1][2]*delvz)*dely \
										+(sigma_[i][2][0]*delvx + sigma_[i][2][1]*delvy + sigma_[i][2][2]*delvz)*delz)/rhoi/rhoi;
		eshear += 0.5*imass*jmass*PIJ*wfd*(delVdotDelR);
		*/

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

   if (vflag_fdotr) virial_fdotr_compute();
  
}

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::allocate() {
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

void PairSPHSolidMechanicsMurnaghan::settings(int narg, char **arg) {
  if (narg != 0)
    error->all(FLERR,
        "Illegal number of setting arguments for pair_style sph/solidMechanics/murnaghan");

}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairSPHSolidMechanicsMurnaghan::coeff(int narg, char **arg) {
  if (narg != 11)
    error->all(FLERR,"Incorrect number of args for pair_style sph/solidMechanics/murnaghan coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(arg[0], atom->ntypes, ilo, ihi);
  force->bounds(arg[1], atom->ntypes, jlo, jhi);

  double viscosity_one = force->numeric(FLERR, arg[2]);
  double cut_one = force->numeric(FLERR, arg[3]);

  G_ = force->numeric(FLERR, arg[4]);
  Gamma_ = force->numeric(FLERR, arg[5]);
  c_ = force->numeric(FLERR, arg[6]);
  Cs_ = c_;
  S_ = force->numeric(FLERR, arg[7]);
  rho0_ = force->numeric(FLERR, arg[8]);
  Y0_ = force->numeric(FLERR, arg[9]);

  if(strcmp(arg[10],"yes") == 0) 
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
    error->all(FLERR,"Incorrect args for pair sph/solidMechanics/murnaghan coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairSPHSolidMechanicsMurnaghan::init_one(int i, int j) {

  if (setflag[i][j] == 0) {
      error->all(FLERR,"All pair sph/solidMechanics/murnaghan coeffs are not set");
  }

  cut[j][i] = cut[i][j];

  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

double PairSPHSolidMechanicsMurnaghan::single(int i, int j, int itype, int jtype,
    double rsq, double factor_coul, double factor_lj, double &fforce) {
  fforce = 0.0;

  return 0.0;
}
