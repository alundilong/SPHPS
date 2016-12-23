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

#include "string.h"
#include "stdlib.h"
#include "atom_vec_meso_multiphase.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

AtomVecMesoMultiPhase::AtomVecMesoMultiPhase(LAMMPS *lmp) :
	AtomVec(lmp) {
	molecular = 0;
	mass_type = 0;
	forceclearflag = 1;

	comm_x_only = 0; // we communicate not only x forward but also vest ...
	comm_f_only = 0; // we also communicate de and drho in reverse direction
	size_forward = 12 + 1 + 45 + 1; // 3 + rmass + rho + colorgradient[3] + e + vest[3], that means we may only communicate 8 in hybrid + 2 + 5*9 (pressure + 45)
	size_reverse = 5; // 3 + drho + de
	size_border = 16 + 1 + 45 + 1; // 6 + rmass + rho + colorgradient[3] + e + vest[3] + cv + viscosity
	size_velocity = 3;
	size_data_atom = 8;
	size_data_vel = 4;
	xcol_data = 6;

	atom->rmass_flag = 1;
	atom->e_flag = 1;
	atom->rho_flag = 1;
	atom->colorgradient_flag = 1;
	atom->cv_flag = 1;
	atom->vest_flag = 1;

  atom->epsilon_flag_ = atom->sigma_flag_ = atom->epsilonBar_flag_ = 1;
  atom->R_flag_ = atom->tau_flag_ = 1; atom->p_flag_ = 1; atom->viscosity_flag_ = 1;
}

/* ----------------------------------------------------------------------
   grow atom arrays
   n = 0 grows arrays by a chunk
   n > 0 allocates arrays to size n
   ------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::grow(int n) {
        if (n == 0) grow_nmax();
        else nmax = n;
	atom->nmax = nmax;
	if (nmax < 0 || nmax > MAXSMALLINT)
		error->one(FLERR,"Per-processor system is too big");

	tag = memory->grow(atom->tag, nmax, "atom:tag");
	type = memory->grow(atom->type, nmax, "atom:type");
	mask = memory->grow(atom->mask, nmax, "atom:mask");
	image = memory->grow(atom->image, nmax, "atom:image");
	x = memory->grow(atom->x, nmax, 3, "atom:x");
	v = memory->grow(atom->v, nmax, 3, "atom:v");
	f = memory->grow(atom->f, nmax*comm->nthreads, 3, "atom:f");

	rho = memory->grow(atom->rho, nmax, "atom:rho");
	colorgradient = memory->grow(atom->colorgradient, nmax, 3, "atom:colorgradient");
	drho = memory->grow(atom->drho, nmax*comm->nthreads, "atom:drho");
	rmass = memory->grow(atom->rmass, nmax, "atom:rmass");
	e = memory->grow(atom->e, nmax, "atom:e");
	de = memory->grow(atom->de, nmax*comm->nthreads, "atom:de");
	vest = memory->grow(atom->vest, nmax, 3, "atom:vest");
	cv = memory->grow(atom->cv, nmax, "atom:cv");

  epsilon_ = memory->create(atom->epsilon_, nmax, 3, 3, "atom::epsilon");
  sigma_ = memory->create(atom->sigma_, nmax, 3, 3, "atom::sigma");
  epsilonBar_ = memory->create(atom->epsilonBar_, nmax, 3, 3, "atom::epsilonBar");
  R_ = memory->create(atom->R_, nmax, 3, 3, "atom::R");
  tau_ = memory->create(atom->tau_, nmax, 3, 3, "atom::tau");
  p_ = memory->grow(atom->p_,nmax,"atom:pressure");
  viscosity_ = memory->grow(atom->viscosity_,nmax,"atom:viscosity");

	if (atom->nextra_grow)
		for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
			modify->fix[atom->extra_grow[iextra]]->grow_arrays(nmax);
}

/* ----------------------------------------------------------------------
   reset local array ptrs
   ------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::grow_reset() {
	tag = atom->tag;
	type = atom->type;
	mask = atom->mask;
	image = atom->image;
	x = atom->x;
	v = atom->v;
	f = atom->f;
	rho = atom->rho;
	colorgradient = atom->colorgradient;
	drho = atom->drho;
	rmass = atom->rmass;
	e = atom->e;
	de = atom->de;
	vest = atom->vest;
	cv = atom->cv;

  epsilon_ = atom->epsilon_;
  sigma_ = atom->sigma_;
  epsilonBar_ = atom->epsilonBar_;
  R_ = atom->R_;
  tau_ = atom->tau_;
  p_ = atom->p_;
  viscosity_ = atom->viscosity_;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::copy(int i, int j, int delflag) {
	//printf("in AtomVecMesoMultiPhase::copy\n");
	tag[j] = tag[i];
	type[j] = type[i];
	mask[j] = mask[i];
	image[j] = image[i];
	x[j][0] = x[i][0];
	x[j][1] = x[i][1];
	x[j][2] = x[i][2];
	v[j][0] = v[i][0];
	v[j][1] = v[i][1];
	v[j][2] = v[i][2];

	rho[j] = rho[i];
	colorgradient[j][0] = colorgradient[i][0];
	colorgradient[j][1] = colorgradient[i][1];
	colorgradient[j][2] = colorgradient[i][2];
	drho[j] = drho[i];
	rmass[j] = rmass[i];
	e[j] = e[i];
	de[j] = de[i];
	cv[j] = cv[i];
	vest[j][0] = vest[i][0];
	vest[j][1] = vest[i][1];
	vest[j][2] = vest[i][2];

  epsilon_[j][0][0] = epsilon_[i][0][0];
  epsilon_[j][0][1] = epsilon_[i][0][1];
  epsilon_[j][0][2] = epsilon_[i][0][2];
  epsilon_[j][1][0] = epsilon_[i][1][0];
  epsilon_[j][1][1] = epsilon_[i][1][1];
  epsilon_[j][1][2] = epsilon_[i][1][2];
  epsilon_[j][2][0] = epsilon_[i][2][0];
  epsilon_[j][2][1] = epsilon_[i][2][1];
  epsilon_[j][2][2] = epsilon_[i][2][2];

  sigma_[j][0][0] = sigma_[i][0][0];
  sigma_[j][0][1] = sigma_[i][0][1];
  sigma_[j][0][2] = sigma_[i][0][2];
  sigma_[j][1][0] = sigma_[i][1][0];
  sigma_[j][1][1] = sigma_[i][1][1];
  sigma_[j][1][2] = sigma_[i][1][2];
  sigma_[j][2][0] = sigma_[i][2][0];
  sigma_[j][2][1] = sigma_[i][2][1];
  sigma_[j][2][2] = sigma_[i][2][2];

  epsilonBar_[j][0][0] = epsilonBar_[i][0][0];
  epsilonBar_[j][0][1] = epsilonBar_[i][0][1];
  epsilonBar_[j][0][2] = epsilonBar_[i][0][2];
  epsilonBar_[j][1][0] = epsilonBar_[i][1][0];
  epsilonBar_[j][1][1] = epsilonBar_[i][1][1];
  epsilonBar_[j][1][2] = epsilonBar_[i][1][2];
  epsilonBar_[j][2][0] = epsilonBar_[i][2][0];
  epsilonBar_[j][2][1] = epsilonBar_[i][2][1];
  epsilonBar_[j][2][2] = epsilonBar_[i][2][2];

  R_[j][0][0] = R_[i][0][0];
  R_[j][0][1] = R_[i][0][1];
  R_[j][0][2] = R_[i][0][2];
  R_[j][1][0] = R_[i][1][0];
  R_[j][1][1] = R_[i][1][1];
  R_[j][1][2] = R_[i][1][2];
  R_[j][2][0] = R_[i][2][0];
  R_[j][2][1] = R_[i][2][1];
  R_[j][2][2] = R_[i][2][2];

  tau_[j][0][0] = tau_[i][0][0];
  tau_[j][0][1] = tau_[i][0][1];
  tau_[j][0][2] = tau_[i][0][2];
  tau_[j][1][0] = tau_[i][1][0];
  tau_[j][1][1] = tau_[i][1][1];
  tau_[j][1][2] = tau_[i][1][2];
  tau_[j][2][0] = tau_[i][2][0];
  tau_[j][2][1] = tau_[i][2][1];
  tau_[j][2][2] = tau_[i][2][2];
  p_[j] = p_[i];
  viscosity_[j] = viscosity_[i];

	if (atom->nextra_grow)
		for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
		  modify->fix[atom->extra_grow[iextra]]->copy_arrays(i, j, delflag);
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::force_clear(int n, size_t nbytes)
{
  memset(&de[n],0,nbytes);
  memset(&drho[n],0,nbytes);
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_comm_hybrid(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
	int i, j, m;

	m = 0;
	if (!deform_vremap) {
	  for (i = 0; i < n; i++) {
	    j = list[i];
	    buf[m++] = rho[j];
	    buf[m++] = colorgradient[j][0];
	    buf[m++] = colorgradient[j][1];
	    buf[m++] = colorgradient[j][2];
	    buf[m++] = rmass[j];
	    buf[m++] = e[j];
	    buf[m++] = vest[j][0];
	    buf[m++] = vest[j][1];
	    buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];
	
	  buf[m++] = p_[j];
	  buf[m++] = viscosity_[j];

	  }
	} else {
	  double dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
	  double dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
	  double dvz = pbc[2]*h_rate[2];
	  for (i = 0; i < n; i++) {
	    j = list[i];
	    buf[m++] = rho[j];
	    buf[m++] = colorgradient[j][0];
	    buf[m++] = colorgradient[j][1];
	    buf[m++] = colorgradient[j][2];
	    buf[m++] = rmass[j];
	    buf[m++] = e[j];
	    if (mask[i] & deform_groupbit) {
	      buf[m++] = vest[j][0] + dvx;
	      buf[m++] = vest[j][1] + dvy;
	      buf[m++] = vest[j][2] + dvz;
	    } else {
	      buf[m++] = vest[j][0];
	      buf[m++] = vest[j][1];
	      buf[m++] = vest[j][2];
	    }

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];
	
	  buf[m++] = p_[j];
	  buf[m++] = viscosity_[j];
	  }
	}
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::unpack_comm_hybrid(int n, int first, double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_comm_hybrid\n");
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		rho[i] = buf[m++];
		colorgradient[i][0] = buf[m++];
		colorgradient[i][1] = buf[m++];
		colorgradient[i][2] = buf[m++];
		rmass[i] = buf[m++];
		e[i] = buf[m++];
		vest[i][0] = buf[m++];
		vest[i][1] = buf[m++];
		vest[i][2] = buf[m++];

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

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

	  p_[i] = buf[m++];
	  viscosity_[i] = buf[m++];
	}
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_border_hybrid(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
	//printf("in AtomVecMesoMultiPhase::pack_border_hybrid\n");
	int i, j, m;
	m = 0;
	if (!deform_vremap) {
	  for (i = 0; i < n; i++) {
	    j = list[i];
	    buf[m++] = rho[j];
	    buf[m++] = colorgradient[j][0];
	    buf[m++] = colorgradient[j][1];
	    buf[m++] = colorgradient[j][2];
	    buf[m++] = rmass[j];
	    buf[m++] = e[j];
	    buf[m++] = vest[j][0];
	    buf[m++] = vest[j][1];
	    buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
	  }
	} else {
	  double dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
	  double dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
	  double dvz = pbc[2]*h_rate[2];
	  for (i = 0; i < n; i++) {
	    j = list[i];
	    buf[m++] = rho[j];
	    buf[m++] = colorgradient[j][0];
	    buf[m++] = colorgradient[j][1];
	    buf[m++] = colorgradient[j][2];
	    buf[m++] = rmass[j];
	    buf[m++] = e[j];
	    if (mask[i] & deform_groupbit) {
	      buf[m++] = vest[j][0] + dvx;
	      buf[m++] = vest[j][1] + dvy;
	      buf[m++] = vest[j][2] + dvz;
	    } else {
	      buf[m++] = vest[j][0];
	      buf[m++] = vest[j][1];
	      buf[m++] = vest[j][2];
	    }
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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
	  }
	}
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::unpack_border_hybrid(int n, int first, double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_border_hybrid\n");
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		rho[i] = buf[m++];
		colorgradient[i][0] = buf[m++];
		colorgradient[i][1] = buf[m++];
		colorgradient[i][2] = buf[m++];
		rmass[i] = buf[m++];
		e[i] = buf[m++];
		vest[i][0] = buf[m++];
		vest[i][1] = buf[m++];
		vest[i][2] = buf[m++];

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

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

      p_[i] = buf[m++];
      viscosity_[i] = buf[m++];

	}
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_reverse_hybrid(int n, int first, double *buf) {
  //printf("in AtomVecMesoMultiPhase::pack_reverse_hybrid\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = drho[i];
    buf[m++] = de[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::unpack_reverse_hybrid(int n, int *list, double *buf) {
  //printf("in AtomVecMesoMultiPhase::unpack_reverse_hybrid\n");
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    drho[j] += buf[m++];
    de[j] += buf[m++];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_comm(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
  //	printf("in AtomVecMesoMultiPhase::pack_comm\n");
	int i, j, m;
	double dx, dy, dz;

	m = 0;
	if (pbc_flag == 0) {
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = x[j][0];
			buf[m++] = x[j][1];
			buf[m++] = x[j][2];
			buf[m++] = rho[j];
			buf[m++] = colorgradient[j][0];
			buf[m++] = colorgradient[j][1];
			buf[m++] = colorgradient[j][2];
			buf[m++] = rmass[j];
			buf[m++] = e[j];
			buf[m++] = vest[j][0];
			buf[m++] = vest[j][1];
			buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

	  buf[m++] = p_[j];
	  buf[m++] = viscosity_[j];

		}
	} else {
		if (domain->triclinic == 0) {
			dx = pbc[0] * domain->xprd;
			dy = pbc[1] * domain->yprd;
			dz = pbc[2] * domain->zprd;
		} else {
			dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
			dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
			dz = pbc[2] * domain->zprd;
		}
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = x[j][0] + dx;
			buf[m++] = x[j][1] + dy;
			buf[m++] = x[j][2] + dz;
			buf[m++] = rho[j];
			buf[m++] = colorgradient[j][0];
			buf[m++] = colorgradient[j][1];
			buf[m++] = colorgradient[j][2];
			buf[m++] = rmass[j];
			buf[m++] = e[j];
			buf[m++] = vest[j][0];
			buf[m++] = vest[j][1];
			buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		}
	}
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_comm_vel(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
	//printf("in AtomVecMesoMultiPhase::pack_comm_vel\n");
	int i, j, m;
	double dx, dy, dz;

	m = 0;
	if (pbc_flag == 0) {
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = x[j][0];
			buf[m++] = x[j][1];
			buf[m++] = x[j][2];
			buf[m++] = v[j][0];
			buf[m++] = v[j][1];
			buf[m++] = v[j][2];
			buf[m++] = rho[j];
			buf[m++] = colorgradient[j][0];
			buf[m++] = colorgradient[j][1];
			buf[m++] = colorgradient[j][2];
			buf[m++] = rmass[j];
			buf[m++] = e[j];
			buf[m++] = vest[j][0];
			buf[m++] = vest[j][1];
			buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		}
	} else {
		if (domain->triclinic == 0) {
			dx = pbc[0] * domain->xprd;
			dy = pbc[1] * domain->yprd;
			dz = pbc[2] * domain->zprd;
		} else {
			dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
			dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
			dz = pbc[2] * domain->zprd;
		}
		if (!deform_vremap) {
		  for (i = 0; i < n; i++) {
		    j = list[i];
		    buf[m++] = x[j][0] + dx;
		    buf[m++] = x[j][1] + dy;
		    buf[m++] = x[j][2] + dz;
		    buf[m++] = v[j][0];
		    buf[m++] = v[j][1];
		    buf[m++] = v[j][2];
		    buf[m++] = rho[j];
		    buf[m++] = colorgradient[j][0];
		    buf[m++] = colorgradient[j][1];
		    buf[m++] = colorgradient[j][2];
		    buf[m++] = rmass[j];
		    buf[m++] = e[j];
		    buf[m++] = vest[j][0];
		    buf[m++] = vest[j][1];
		    buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		  }
		} else {
		  double dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
		  double dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
		  double dvz = pbc[2]*h_rate[2];
		  for (i = 0; i < n; i++) {
		    j = list[i];
		    buf[m++] = x[j][0] + dx;
		    buf[m++] = x[j][1] + dy;
		    buf[m++] = x[j][2] + dz;
		    if (mask[i] & deform_groupbit) {
		      buf[m++] = v[j][0] + dvx;
		      buf[m++] = v[j][1] + dvy;
		      buf[m++] = v[j][2] + dvz;
		    } else {
		      buf[m++] = v[j][0];
		      buf[m++] = v[j][1];
		      buf[m++] = v[j][2];
		    }
		    buf[m++] = rho[j];
		    buf[m++] = colorgradient[j][0];
		    buf[m++] = colorgradient[j][1];
		    buf[m++] = colorgradient[j][2];
		    buf[m++] = rmass[j];
		    buf[m++] = e[j];
		    if (mask[i] & deform_groupbit) {
		      buf[m++] = vest[j][0] + dvx;
		      buf[m++] = vest[j][1] + dvy;
		      buf[m++] = vest[j][2] + dvz;
		    } else {
		      buf[m++] = vest[j][0];
		      buf[m++] = vest[j][1];
		      buf[m++] = vest[j][2];
		    }
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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		  }
		}
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::unpack_comm(int n, int first, double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_comm\n");
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		x[i][0] = buf[m++];
		x[i][1] = buf[m++];
		x[i][2] = buf[m++];
		rho[i] = buf[m++];
		colorgradient[i][0] = buf[m++];
		colorgradient[i][1] = buf[m++];
		colorgradient[i][2] = buf[m++];
		rmass[i] = buf[m++];
		e[i] = buf[m++];
		vest[i][0] = buf[m++];
		vest[i][1] = buf[m++];
		vest[i][2] = buf[m++];

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

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

      p_[i] = buf[m++];
      viscosity_[i] = buf[m++];
	}
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::unpack_comm_vel(int n, int first, double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_comm_vel\n");
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		x[i][0] = buf[m++];
		x[i][1] = buf[m++];
		x[i][2] = buf[m++];
		v[i][0] = buf[m++];
		v[i][1] = buf[m++];
		v[i][2] = buf[m++];
		rho[i] = buf[m++];
		colorgradient[i][0] = buf[m++];
		colorgradient[i][1] = buf[m++];
		colorgradient[i][2] = buf[m++];
		rmass[i] = buf[m++];
		e[i] = buf[m++];
		vest[i][0] = buf[m++];
		vest[i][1] = buf[m++];
		vest[i][2] = buf[m++];

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

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

      p_[i] = buf[m++];
      viscosity_[i] = buf[m++];
	}
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_reverse(int n, int first, double *buf) {
  //printf("in AtomVecMesoMultiPhase::pack_reverse\n");
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = f[i][0];
    buf[m++] = f[i][1];
    buf[m++] = f[i][2];
    buf[m++] = drho[i];
    buf[m++] = de[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::unpack_reverse(int n, int *list, double *buf) {
  //printf("in AtomVecMesoMultiPhase::unpack_reverse\n");
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    f[j][0] += buf[m++];
    f[j][1] += buf[m++];
    f[j][2] += buf[m++];
    drho[j] += buf[m++];
    de[j] += buf[m++];
  }
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_border(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
	//printf("in AtomVecMesoMultiPhase::pack_border\n");
	int i, j, m;
	double dx, dy, dz;

	m = 0;
	if (pbc_flag == 0) {
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = x[j][0];
			buf[m++] = x[j][1];
			buf[m++] = x[j][2];
			buf[m++] = tag[j];
			buf[m++] = type[j];
			buf[m++] = mask[j];
			buf[m++] = rho[j];
			buf[m++] = colorgradient[j][0];
			buf[m++] = colorgradient[j][1];
			buf[m++] = colorgradient[j][2];
			buf[m++] = rmass[j];
			buf[m++] = e[j];
			buf[m++] = cv[j];
			buf[m++] = vest[j][0];
			buf[m++] = vest[j][1];
			buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		}
	} else {
		if (domain->triclinic == 0) {
			dx = pbc[0] * domain->xprd;
			dy = pbc[1] * domain->yprd;
			dz = pbc[2] * domain->zprd;
		} else {
			dx = pbc[0];
			dy = pbc[1];
			dz = pbc[2];
		}
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = x[j][0] + dx;
			buf[m++] = x[j][1] + dy;
			buf[m++] = x[j][2] + dz;
			buf[m++] = tag[j];
			buf[m++] = type[j];
			buf[m++] = mask[j];
			buf[m++] = rho[j];
			buf[m++] = colorgradient[j][0];
			buf[m++] = colorgradient[j][1];
			buf[m++] = colorgradient[j][2];
			buf[m++] = rmass[j];
			buf[m++] = e[j];
			buf[m++] = cv[j];
			buf[m++] = vest[j][0];
			buf[m++] = vest[j][1];
			buf[m++] = vest[j][2];
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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		}
	}
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_border_vel(int n, int *list, double *buf, int pbc_flag,
		int *pbc) {
	//printf("in AtomVecMesoMultiPhase::pack_border_vel\n");
	int i, j, m;
	double dx, dy, dz;

	m = 0;
	if (pbc_flag == 0) {
		for (i = 0; i < n; i++) {
			j = list[i];
			buf[m++] = x[j][0];
			buf[m++] = x[j][1];
			buf[m++] = x[j][2];
			buf[m++] = tag[j];
			buf[m++] = type[j];
			buf[m++] = mask[j];
			buf[m++] = v[j][0];
			buf[m++] = v[j][1];
			buf[m++] = v[j][2];
			buf[m++] = rho[j];
			buf[m++] = colorgradient[j][0];
			buf[m++] = colorgradient[j][1];
			buf[m++] = colorgradient[j][2];
			buf[m++] = rmass[j];
			buf[m++] = e[j];
			buf[m++] = cv[j];
			buf[m++] = vest[j][0];
			buf[m++] = vest[j][1];
			buf[m++] = vest[j][2];

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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];

		}
	} else {
		if (domain->triclinic == 0) {
			dx = pbc[0] * domain->xprd;
			dy = pbc[1] * domain->yprd;
			dz = pbc[2] * domain->zprd;
		} else {
			dx = pbc[0];
			dy = pbc[1];
			dz = pbc[2];
		}
		if (!deform_vremap) {
		  for (i = 0; i < n; i++) {
		    j = list[i];
		    buf[m++] = x[j][0] + dx;
		    buf[m++] = x[j][1] + dy;
		    buf[m++] = x[j][2] + dz;
		    buf[m++] = tag[j];
		    buf[m++] = type[j];
		    buf[m++] = mask[j];
		    buf[m++] = v[j][0];
		    buf[m++] = v[j][1];
		    buf[m++] = v[j][2];
		    buf[m++] = rho[j];
		    buf[m++] = colorgradient[j][0];
		    buf[m++] = colorgradient[j][1];
		    buf[m++] = colorgradient[j][2];
		    buf[m++] = rmass[j];
		    buf[m++] = e[j];
		    buf[m++] = cv[j];
		    buf[m++] = vest[j][0];
		    buf[m++] = vest[j][1];
		    buf[m++] = vest[j][2];
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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		  }
		} else {
		  double dvx = pbc[0]*h_rate[0] + pbc[5]*h_rate[5] + pbc[4]*h_rate[4];
		  double dvy = pbc[1]*h_rate[1] + pbc[3]*h_rate[3];
		  double dvz = pbc[2]*h_rate[2];
		  for (i = 0; i < n; i++) {
		    j = list[i];
		    buf[m++] = x[j][0] + dx;
		    buf[m++] = x[j][1] + dy;
		    buf[m++] = x[j][2] + dz;
		    buf[m++] = tag[j];
		    buf[m++] = type[j];
		    buf[m++] = mask[j];
		    if (mask[i] & deform_groupbit) {
		      buf[m++] = v[j][0] + dvx;
		      buf[m++] = v[j][1] + dvy;
		      buf[m++] = v[j][2] + dvz;
		    } else {
		      buf[m++] = v[j][0];
		      buf[m++] = v[j][1];
		      buf[m++] = v[j][2];
		    }
		    buf[m++] = rho[j];
		    buf[m++] = colorgradient[j][0];
		    buf[m++] = colorgradient[j][1];
		    buf[m++] = colorgradient[j][2];
		    buf[m++] = rmass[j];
		    buf[m++] = e[j];
		    buf[m++] = cv[j];
		    if (mask[i] & deform_groupbit) {
		      buf[m++] = vest[j][0] + dvx;
		      buf[m++] = vest[j][1] + dvy;
		      buf[m++] = vest[j][2] + dvz;
		    } else {
		      buf[m++] = vest[j][0];
		      buf[m++] = vest[j][1];
		      buf[m++] = vest[j][2];
		    }
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

      buf[m++] = epsilonBar_[j][0][0];
      buf[m++] = epsilonBar_[j][0][1];
      buf[m++] = epsilonBar_[j][0][2];
      buf[m++] = epsilonBar_[j][1][0];
      buf[m++] = epsilonBar_[j][1][1];
      buf[m++] = epsilonBar_[j][1][2];
      buf[m++] = epsilonBar_[j][2][0];
      buf[m++] = epsilonBar_[j][2][1];
      buf[m++] = epsilonBar_[j][2][2];

      buf[m++] = R_[j][0][0];
      buf[m++] = R_[j][0][1];
      buf[m++] = R_[j][0][2];
      buf[m++] = R_[j][1][0];
      buf[m++] = R_[j][1][1];
      buf[m++] = R_[j][1][2];
      buf[m++] = R_[j][2][0];
      buf[m++] = R_[j][2][1];
      buf[m++] = R_[j][2][2];

      buf[m++] = tau_[j][0][0];
      buf[m++] = tau_[j][0][1];
      buf[m++] = tau_[j][0][2];
      buf[m++] = tau_[j][1][0];
      buf[m++] = tau_[j][1][1];
      buf[m++] = tau_[j][1][2];
      buf[m++] = tau_[j][2][0];
      buf[m++] = tau_[j][2][1];
      buf[m++] = tau_[j][2][2];

      buf[m++] = p_[j];
      buf[m++] = viscosity_[j];
		  }
		}
	}
	return m;
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::unpack_border(int n, int first, double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_border\n");
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		if (i == nmax)
			grow(0);
		x[i][0] = buf[m++];
		x[i][1] = buf[m++];
		x[i][2] = buf[m++];
		tag[i] = static_cast<int> (buf[m++]);
		type[i] = static_cast<int> (buf[m++]);
		mask[i] = static_cast<int> (buf[m++]);
		rho[i] = buf[m++];
		colorgradient[i][0] = buf[m++];
		colorgradient[i][1] = buf[m++];
		colorgradient[i][2] = buf[m++];
		rmass[i] = buf[m++];
		e[i] = buf[m++];
		cv[i] = buf[m++];
		vest[i][0] = buf[m++];
		vest[i][1] = buf[m++];
		vest[i][2] = buf[m++];

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

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

      p_[i] = buf[m++];
      viscosity_[i] = buf[m++];
	}
}

/* ---------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::unpack_border_vel(int n, int first, double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_border_vel\n");
	int i, m, last;

	m = 0;
	last = first + n;
	for (i = first; i < last; i++) {
		if (i == nmax)
			grow(0);
		x[i][0] = buf[m++];
		x[i][1] = buf[m++];
		x[i][2] = buf[m++];
		tag[i] = static_cast<int> (buf[m++]);
		type[i] = static_cast<int> (buf[m++]);
		mask[i] = static_cast<int> (buf[m++]);
		v[i][0] = buf[m++];
		v[i][1] = buf[m++];
		v[i][2] = buf[m++];
		rho[i] = buf[m++];
		colorgradient[i][0] = buf[m++];
		colorgradient[i][1] = buf[m++];
		colorgradient[i][2] = buf[m++];
		rmass[i] = buf[m++];
		e[i] = buf[m++];
		cv[i] = buf[m++];
		vest[i][0] = buf[m++];
		vest[i][1] = buf[m++];
		vest[i][2] = buf[m++];

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

      epsilonBar_[i][0][0] = buf[m++];
      epsilonBar_[i][0][1] = buf[m++];
      epsilonBar_[i][0][2] = buf[m++];
      epsilonBar_[i][1][0] = buf[m++];
      epsilonBar_[i][1][1] = buf[m++];
      epsilonBar_[i][1][2] = buf[m++];
      epsilonBar_[i][2][0] = buf[m++];
      epsilonBar_[i][2][1] = buf[m++];
      epsilonBar_[i][2][2] = buf[m++];

      R_[i][0][0] = buf[m++];
      R_[i][0][1] = buf[m++];
      R_[i][0][2] = buf[m++];
      R_[i][1][0] = buf[m++];
      R_[i][1][1] = buf[m++];
      R_[i][1][2] = buf[m++];
      R_[i][2][0] = buf[m++];
      R_[i][2][1] = buf[m++];
      R_[i][2][2] = buf[m++];

      tau_[i][0][0] = buf[m++];
      tau_[i][0][1] = buf[m++];
      tau_[i][0][2] = buf[m++];
      tau_[i][1][0] = buf[m++];
      tau_[i][1][1] = buf[m++];
      tau_[i][1][2] = buf[m++];
      tau_[i][2][0] = buf[m++];
      tau_[i][2][1] = buf[m++];
      tau_[i][2][2] = buf[m++];

      p_[i] = buf[m++];
      viscosity_[i] = buf[m++];
	}
}

/* ----------------------------------------------------------------------
   pack data for atom I for sending to another proc
   xyz must be 1st 3 values, so comm::exchange() can test on them
   ------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_exchange(int i, double *buf) {
	//printf("in AtomVecMesoMultiPhase::pack_exchange\n");
	int m = 1;
	buf[m++] = x[i][0];
	buf[m++] = x[i][1];
	buf[m++] = x[i][2];
	buf[m++] = v[i][0];
	buf[m++] = v[i][1];
	buf[m++] = v[i][2];
	buf[m++] = tag[i];
	buf[m++] = type[i];
	buf[m++] = mask[i];
	buf[m++] = image[i];
	buf[m++] = rho[i];
	buf[m++] = colorgradient[i][0];
	buf[m++] = colorgradient[i][1];
	buf[m++] = colorgradient[i][2];
	buf[m++] = rmass[i];
	buf[m++] = e[i];
	buf[m++] = cv[i];
	buf[m++] = vest[i][0];
	buf[m++] = vest[i][1];
	buf[m++] = vest[i][2];

      buf[m++] = epsilon_[i][0][0];
      buf[m++] = epsilon_[i][0][1];
      buf[m++] = epsilon_[i][0][2];
      buf[m++] = epsilon_[i][1][0];
      buf[m++] = epsilon_[i][1][1];
      buf[m++] = epsilon_[i][1][2];
      buf[m++] = epsilon_[i][2][0];
      buf[m++] = epsilon_[i][2][1];
      buf[m++] = epsilon_[i][2][2];

      buf[m++] = sigma_[i][0][0];
      buf[m++] = sigma_[i][0][1];
      buf[m++] = sigma_[i][0][2];
      buf[m++] = sigma_[i][1][0];
      buf[m++] = sigma_[i][1][1];
      buf[m++] = sigma_[i][1][2];
      buf[m++] = sigma_[i][2][0];
      buf[m++] = sigma_[i][2][1];
      buf[m++] = sigma_[i][2][2];

      buf[m++] = epsilonBar_[i][0][0];
      buf[m++] = epsilonBar_[i][0][1];
      buf[m++] = epsilonBar_[i][0][2];
      buf[m++] = epsilonBar_[i][1][0];
      buf[m++] = epsilonBar_[i][1][1];
      buf[m++] = epsilonBar_[i][1][2];
      buf[m++] = epsilonBar_[i][2][0];
      buf[m++] = epsilonBar_[i][2][1];
      buf[m++] = epsilonBar_[i][2][2];

      buf[m++] = R_[i][0][0];
      buf[m++] = R_[i][0][1];
      buf[m++] = R_[i][0][2];
      buf[m++] = R_[i][1][0];
      buf[m++] = R_[i][1][1];
      buf[m++] = R_[i][1][2];
      buf[m++] = R_[i][2][0];
      buf[m++] = R_[i][2][1];
      buf[m++] = R_[i][2][2];

      buf[m++] = tau_[i][0][0];
      buf[m++] = tau_[i][0][1];
      buf[m++] = tau_[i][0][2];
      buf[m++] = tau_[i][1][0];
      buf[m++] = tau_[i][1][1];
      buf[m++] = tau_[i][1][2];
      buf[m++] = tau_[i][2][0];
      buf[m++] = tau_[i][2][1];
      buf[m++] = tau_[i][2][2];

      buf[m++] = p_[i];
      buf[m++] = viscosity_[i];

	if (atom->nextra_grow)
		for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
			m += modify->fix[atom->extra_grow[iextra]]->pack_exchange(i, &buf[m]);

	buf[0] = m;
	return m;
}

/* ---------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::unpack_exchange(double *buf) {
	//printf("in AtomVecMesoMultiPhase::unpack_exchange\n");
	int nlocal = atom->nlocal;
	if (nlocal == nmax)
		grow(0);

	int m = 1;
	x[nlocal][0] = buf[m++];
	x[nlocal][1] = buf[m++];
	x[nlocal][2] = buf[m++];
	v[nlocal][0] = buf[m++];
	v[nlocal][1] = buf[m++];
	v[nlocal][2] = buf[m++];
	tag[nlocal] = static_cast<int> (buf[m++]);
	type[nlocal] = static_cast<int> (buf[m++]);
	mask[nlocal] = static_cast<int> (buf[m++]);
	image[nlocal] = static_cast<int> (buf[m++]);
	rho[nlocal] = buf[m++];
	colorgradient[nlocal][0] = buf[m++];
	colorgradient[nlocal][1] = buf[m++];
	colorgradient[nlocal][2] = buf[m++];
	rmass[nlocal] = buf[m++];
	e[nlocal] = buf[m++];
	cv[nlocal] = buf[m++];
	vest[nlocal][0] = buf[m++];
	vest[nlocal][1] = buf[m++];
	vest[nlocal][2] = buf[m++];

      epsilon_[nlocal][0][0] = buf[m++];
      epsilon_[nlocal][0][1] = buf[m++];
      epsilon_[nlocal][0][2] = buf[m++];
      epsilon_[nlocal][1][0] = buf[m++];
      epsilon_[nlocal][1][1] = buf[m++];
      epsilon_[nlocal][1][2] = buf[m++];
      epsilon_[nlocal][2][0] = buf[m++];
      epsilon_[nlocal][2][1] = buf[m++];
      epsilon_[nlocal][2][2] = buf[m++];

      sigma_[nlocal][0][0] = buf[m++];
      sigma_[nlocal][0][1] = buf[m++];
      sigma_[nlocal][0][2] = buf[m++];
      sigma_[nlocal][1][0] = buf[m++];
      sigma_[nlocal][1][1] = buf[m++];
      sigma_[nlocal][1][2] = buf[m++];
      sigma_[nlocal][2][0] = buf[m++];
      sigma_[nlocal][2][1] = buf[m++];
      sigma_[nlocal][2][2] = buf[m++];

      epsilonBar_[nlocal][0][0] = buf[m++];
      epsilonBar_[nlocal][0][1] = buf[m++];
      epsilonBar_[nlocal][0][2] = buf[m++];
      epsilonBar_[nlocal][1][0] = buf[m++];
      epsilonBar_[nlocal][1][1] = buf[m++];
      epsilonBar_[nlocal][1][2] = buf[m++];
      epsilonBar_[nlocal][2][0] = buf[m++];
      epsilonBar_[nlocal][2][1] = buf[m++];
      epsilonBar_[nlocal][2][2] = buf[m++];

      R_[nlocal][0][0] = buf[m++];
      R_[nlocal][0][1] = buf[m++];
      R_[nlocal][0][2] = buf[m++];
      R_[nlocal][1][0] = buf[m++];
      R_[nlocal][1][1] = buf[m++];
      R_[nlocal][1][2] = buf[m++];
      R_[nlocal][2][0] = buf[m++];
      R_[nlocal][2][1] = buf[m++];
      R_[nlocal][2][2] = buf[m++];

      tau_[nlocal][0][0] = buf[m++];
      tau_[nlocal][0][1] = buf[m++];
      tau_[nlocal][0][2] = buf[m++];
      tau_[nlocal][1][0] = buf[m++];
      tau_[nlocal][1][1] = buf[m++];
      tau_[nlocal][1][2] = buf[m++];
      tau_[nlocal][2][0] = buf[m++];
      tau_[nlocal][2][1] = buf[m++];
      tau_[nlocal][2][2] = buf[m++];

      p_[nlocal] = buf[m++];
      viscosity_[nlocal] = buf[m++];

	if (atom->nextra_grow)
		for (int iextra = 0; iextra < atom->nextra_grow; iextra++)
			m += modify->fix[atom->extra_grow[iextra]]-> unpack_exchange(nlocal,
					&buf[m]);

	atom->nlocal++;
	return m;
}

/* ----------------------------------------------------------------------
   size of restart data for all atoms owned by this proc
   include extra data stored by fixes
   ------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::size_restart() {
        int i;

	int nlocal = atom->nlocal;
	int n = 22 * nlocal; // 11 + rmass + rho + colorgradient[3] + e + cv + vest[3] + p

        if (atom->nextra_restart)
                for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
                        for (i = 0; i < nlocal; i++)
                                n += modify->fix[atom->extra_restart[iextra]]->size_restart(i);

        return n;
}

/* ----------------------------------------------------------------------
   pack atom I's data for restart file including extra quantities
   xyz must be 1st 3 values, so that read_restart can test on them
   molecular types may be negative, but write as positive
   ------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_restart(int i, double *buf) {
	int m = 1;
	buf[m++] = x[i][0];
	buf[m++] = x[i][1];
	buf[m++] = x[i][2];
	buf[m++] = tag[i];
	buf[m++] = type[i];
	buf[m++] = mask[i];
	buf[m++] = image[i];
	buf[m++] = v[i][0];
	buf[m++] = v[i][1];
	buf[m++] = v[i][2];
	buf[m++] = rho[i];
	buf[m++] = colorgradient[i][0];
	buf[m++] = colorgradient[i][1];
	buf[m++] = colorgradient[i][2];
	buf[m++] = rmass[i];
	buf[m++] = e[i];
	buf[m++] = cv[i];
	buf[m++] = vest[i][0];
	buf[m++] = vest[i][1];
	buf[m++] = vest[i][2];
	buf[m++] = p_[i];
	buf[m++] = viscosity_[i];

	if (atom->nextra_restart)
		for (int iextra = 0; iextra < atom->nextra_restart; iextra++)
			m += modify->fix[atom->extra_restart[iextra]]->pack_restart(i, &buf[m]);

	buf[0] = m;
	return m;
}

/* ----------------------------------------------------------------------
   unpack data for one atom from restart file including extra quantities
   ------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::unpack_restart(double *buf) {
	int nlocal = atom->nlocal;
	if (nlocal == nmax) {
		grow(0);
		if (atom->nextra_store)
			memory->grow(atom->extra, nmax, atom->nextra_store, "atom:extra");
	}

	int m = 1;
	x[nlocal][0] = buf[m++];
	x[nlocal][1] = buf[m++];
	x[nlocal][2] = buf[m++];
	tag[nlocal] = static_cast<int> (buf[m++]);
	type[nlocal] = static_cast<int> (buf[m++]);
	mask[nlocal] = static_cast<int> (buf[m++]);
	image[nlocal] = static_cast<int> (buf[m++]);
	v[nlocal][0] = buf[m++];
	v[nlocal][1] = buf[m++];
	v[nlocal][2] = buf[m++];
	rho[nlocal] = buf[m++];
	colorgradient[nlocal][0] = buf[m++];
	colorgradient[nlocal][1] = buf[m++];
	colorgradient[nlocal][2] = buf[m++];
	rmass[nlocal] = buf[m++];
	e[nlocal] = buf[m++];
	cv[nlocal] = buf[m++];
	vest[nlocal][0] = buf[m++];
	vest[nlocal][1] = buf[m++];
	vest[nlocal][2] = buf[m++];
	p_[nlocal] = buf[m++];
	viscosity_[nlocal] = buf[m++];

	double **extra = atom->extra;
	if (atom->nextra_store) {
		int size = static_cast<int> (buf[0]) - m;
		for (int i = 0; i < size; i++)
			extra[nlocal][i] = buf[m++];
	}

	atom->nlocal++;
	return m;
}

/* ----------------------------------------------------------------------
   create one atom of itype at coord
   set other values to defaults
   ------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::create_atom(int itype, double *coord) {
	int nlocal = atom->nlocal;
	if (nlocal == nmax)
		grow(0);

	tag[nlocal] = 0;
	type[nlocal] = itype;
	x[nlocal][0] = coord[0];
	x[nlocal][1] = coord[1];
	x[nlocal][2] = coord[2];
	mask[nlocal] = 1;
	image[nlocal] = (512 << 20) | (512 << 10) | 512;
	v[nlocal][0] = 0.0;
	v[nlocal][1] = 0.0;
	v[nlocal][2] = 0.0;
	rho[nlocal] = 0.0;
	colorgradient[nlocal][0] = 0.0;
	colorgradient[nlocal][1] = 0.0;
	colorgradient[nlocal][2] = 0.0;
	rmass[nlocal] = 0.0;
	e[nlocal] = 0.0;
	cv[nlocal] = 1.0;
	vest[nlocal][0] = 0.0;
	vest[nlocal][1] = 0.0;
	vest[nlocal][2] = 0.0;
	de[nlocal] = 0.0;
	drho[nlocal] = 0.0;

  epsilon_[nlocal][0][0] = 0.0;
  epsilon_[nlocal][0][1] = 0.0;
  epsilon_[nlocal][0][2] = 0.0;
  epsilon_[nlocal][1][0] = 0.0;
  epsilon_[nlocal][1][1] = 0.0;
  epsilon_[nlocal][1][2] = 0.0;
  epsilon_[nlocal][2][0] = 0.0;
  epsilon_[nlocal][2][1] = 0.0;
  epsilon_[nlocal][2][2] = 0.0;

  sigma_[nlocal][0][0] = 0.0;
  sigma_[nlocal][0][1] = 0.0;
  sigma_[nlocal][0][2] = 0.0;
  sigma_[nlocal][1][0] = 0.0;
  sigma_[nlocal][1][1] = 0.0;
  sigma_[nlocal][1][2] = 0.0;
  sigma_[nlocal][2][0] = 0.0;
  sigma_[nlocal][2][1] = 0.0;
  sigma_[nlocal][2][2] = 0.0;

  epsilonBar_[nlocal][0][0] = 0.0;
  epsilonBar_[nlocal][0][1] = 0.0;
  epsilonBar_[nlocal][0][2] = 0.0;
  epsilonBar_[nlocal][1][0] = 0.0;
  epsilonBar_[nlocal][1][1] = 0.0;
  epsilonBar_[nlocal][1][2] = 0.0;
  epsilonBar_[nlocal][2][0] = 0.0;
  epsilonBar_[nlocal][2][1] = 0.0;
  epsilonBar_[nlocal][2][2] = 0.0;

  R_[nlocal][0][0] = 0.0;
  R_[nlocal][0][1] = 0.0;
  R_[nlocal][0][2] = 0.0;
  R_[nlocal][1][0] = 0.0;
  R_[nlocal][1][1] = 0.0;
  R_[nlocal][1][2] = 0.0;
  R_[nlocal][2][0] = 0.0;
  R_[nlocal][2][1] = 0.0;
  R_[nlocal][2][2] = 0.0;

  tau_[nlocal][0][0] = 0.0;
  tau_[nlocal][0][1] = 0.0;
  tau_[nlocal][0][2] = 0.0;
  tau_[nlocal][1][0] = 0.0;
  tau_[nlocal][1][1] = 0.0;
  tau_[nlocal][1][2] = 0.0;
  tau_[nlocal][2][0] = 0.0;
  tau_[nlocal][2][1] = 0.0;
  tau_[nlocal][2][2] = 0.0;

  p_[nlocal] = 0.0;
  viscosity_[nlocal] = 0.0;

	atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack one line from Atoms section of data file
   initialize other atom quantities
   ------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::data_atom(double *coord, tagint imagetmp, char **values) {
        int nlocal = atom->nlocal;
        if (nlocal == nmax)
                grow(0);

        tag[nlocal] = atoi(values[0]);
        if (tag[nlocal] <= 0)
                error->one(FLERR,"Invalid atom ID in Atoms section of data file");

        type[nlocal] = atoi(values[1]);
        if (type[nlocal] <= 0 || type[nlocal] > atom->ntypes)
                error->one(FLERR,"Invalid atom type in Atoms section of data file");

        rho[nlocal] = atof(values[2]);
        rmass[nlocal] = atof(values[3]);
        e[nlocal] = atof(values[4]);
        cv[nlocal] = atof(values[5]);

        x[nlocal][0] = coord[0];
        x[nlocal][1] = coord[1];
        x[nlocal][2] = coord[2];

        //printf("rho=%f, e=%f, cv=%f, x=%f\n", rho[nlocal], e[nlocal], cv[nlocal], x[nlocal][0]);

        image[nlocal] = imagetmp;

        mask[nlocal] = 1;
        v[nlocal][0] = 0.0;
        v[nlocal][1] = 0.0;
        v[nlocal][2] = 0.0;

	colorgradient[nlocal][0] = 0.0;
	colorgradient[nlocal][1] = 0.0;
	colorgradient[nlocal][2] = 0.0;

	vest[nlocal][0] = 0.0;
	vest[nlocal][1] = 0.0;
	vest[nlocal][2] = 0.0;

      epsilon_[nlocal][0][0] = 0.0;
      epsilon_[nlocal][0][1] = 0.0;
      epsilon_[nlocal][0][2] = 0.0;
      epsilon_[nlocal][1][0] = 0.0;
      epsilon_[nlocal][1][1] = 0.0;
      epsilon_[nlocal][1][2] = 0.0;
      epsilon_[nlocal][2][0] = 0.0;
      epsilon_[nlocal][2][1] = 0.0;
      epsilon_[nlocal][2][2] = 0.0;

      sigma_[nlocal][0][0] = 0.0;
      sigma_[nlocal][0][1] = 0.0;
      sigma_[nlocal][0][2] = 0.0;
      sigma_[nlocal][1][0] = 0.0;
      sigma_[nlocal][1][1] = 0.0;
      sigma_[nlocal][1][2] = 0.0;
      sigma_[nlocal][2][0] = 0.0;
      sigma_[nlocal][2][1] = 0.0;
      sigma_[nlocal][2][2] = 0.0;

      epsilonBar_[nlocal][0][0] = 0.0;
      epsilonBar_[nlocal][0][1] = 0.0;
      epsilonBar_[nlocal][0][2] = 0.0;
      epsilonBar_[nlocal][1][0] = 0.0;
      epsilonBar_[nlocal][1][1] = 0.0;
      epsilonBar_[nlocal][1][2] = 0.0;
      epsilonBar_[nlocal][2][0] = 0.0;
      epsilonBar_[nlocal][2][1] = 0.0;
      epsilonBar_[nlocal][2][2] = 0.0;

      R_[nlocal][0][0] = 0.0;
      R_[nlocal][0][1] = 0.0;
      R_[nlocal][0][2] = 0.0;
      R_[nlocal][1][0] = 0.0;
      R_[nlocal][1][1] = 0.0;
      R_[nlocal][1][2] = 0.0;
      R_[nlocal][2][0] = 0.0;
      R_[nlocal][2][1] = 0.0;
      R_[nlocal][2][2] = 0.0;

      tau_[nlocal][0][0] = 0.0;
      tau_[nlocal][0][1] = 0.0;
      tau_[nlocal][0][2] = 0.0;
      tau_[nlocal][1][0] = 0.0;
      tau_[nlocal][1][1] = 0.0;
      tau_[nlocal][1][2] = 0.0;
      tau_[nlocal][2][0] = 0.0;
      tau_[nlocal][2][1] = 0.0;
      tau_[nlocal][2][2] = 0.0;
	  p_[nlocal] = 0.0;
	  viscosity_[nlocal] = 0.0;

        de[nlocal] = 0.0;
        drho[nlocal] = 0.0;

        atom->nlocal++;
}

/* ----------------------------------------------------------------------
   unpack hybrid quantities from one line in Atoms section of data file
   initialize other atom quantities for this sub-style
   ------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::data_atom_hybrid(int nlocal, char **values) {

  rho[nlocal] = atof(values[0]);
  rmass[nlocal] = atof(values[1]);
  e[nlocal] = atof(values[2]);
  cv[nlocal] = atof(values[3]);
  p_[nlocal] = atof(values[4]);
  viscosity_[nlocal] = atof(values[5]);

  return 4+1+1;
}

/* ----------------------------------------------------------------------
   pack atom info for data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::pack_data(double **buf)
{
  int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++) {
    buf[i][0] = ubuf(tag[i]).d;
    buf[i][1] = ubuf(type[i]).d;
    buf[i][2] = rho[i];
    buf[i][3] = e[i];
    buf[i][4] = cv[i];
    buf[i][5] = p_[i];
    buf[i][6] = viscosity_[i];
    buf[i][6+1] = x[i][0];
    buf[i][7+1] = x[i][1];
    buf[i][8+1] = x[i][2];
    buf[i][9+1] = ubuf((image[i] & IMGMASK) - IMGMAX).d;
    buf[i][10+1] = ubuf((image[i] >> IMGBITS & IMGMASK) - IMGMAX).d;
    buf[i][11+1] = ubuf((image[i] >> IMG2BITS) - IMGMAX).d;
  }
}

/* ----------------------------------------------------------------------
   pack hybrid atom info for data file
------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::pack_data_hybrid(int i, double *buf)
{
  buf[0] = rho[i];
  buf[1] = e[i];
  buf[2] = cv[i];
  buf[3] = p_[i];
  buf[4] = viscosity_[i];
  return 3+1+1;
}

/* ----------------------------------------------------------------------
   write atom info to data file including 3 image flags
------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::write_data(FILE *fp, int n, double **buf)
{
  for (int i = 0; i < n; i++)
    fprintf(fp,TAGINT_FORMAT 
            " %d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e "
            "%d %d %d\n",
            (tagint) ubuf(buf[i][0]).i,(int) ubuf(buf[i][1]).i,
            buf[i][2],buf[i][3],buf[i][4],buf[i][5],buf[i][6],
            buf[i][5+2],buf[i][6+2],buf[i][7+2],
            (int) ubuf(buf[i][8+2]).i,(int) ubuf(buf[i][9+2]).i,
            (int) ubuf(buf[i][10+2]).i);
}

/* ----------------------------------------------------------------------
   write hybrid atom info to data file
------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::write_data_hybrid(FILE *fp, double *buf)
{
  fprintf(fp," %-1.16e %-1.16e %-1.16e %-1.16e",buf[0],buf[1],buf[2],buf[3]);
  return 3+1;
}

/* ----------------------------------------------------------------------
   assign an index to named atom property and return index
   return -1 if name is unknown to this atom style
------------------------------------------------------------------------- */

int AtomVecMesoMultiPhase::property_atom(char *name)
{
  if (strcmp(name,"rho") == 0) return 0;
  if (strcmp(name,"drho") == 0) return 1;
  if (strcmp(name,"e") == 0) return 2;
  if (strcmp(name,"de") == 0) return 3;
  if (strcmp(name,"cv") == 0) return 4;
  if (strcmp(name,"p") == 0) return 5;
  return -1;
}

/* ----------------------------------------------------------------------
   pack per-atom data into buf for ComputePropertyAtom
   index maps to data specific to this atom style
------------------------------------------------------------------------- */

void AtomVecMesoMultiPhase::pack_property_atom(int index, double *buf, 
                                     int nvalues, int groupbit)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int n = 0;

  if (index == 0) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = rho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 1) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = drho[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 2) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = e[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 3) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = de[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 4) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = cv[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 5) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = p_[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  } else if (index == 6) {
    for (int i = 0; i < nlocal; i++) {
      if (mask[i] & groupbit) buf[n] = viscosity_[i];
      else buf[n] = 0.0;
      n += nvalues;
    }
  }
}

/* ----------------------------------------------------------------------
   return # of bytes of allocated memory
   ------------------------------------------------------------------------- */

bigint AtomVecMesoMultiPhase::memory_usage() {
	bigint bytes = 0;

	if (atom->memcheck("tag"))
		bytes += memory->usage(tag, nmax);
	if (atom->memcheck("type"))
		bytes += memory->usage(type, nmax);
	if (atom->memcheck("mask"))
		bytes += memory->usage(mask, nmax);
	if (atom->memcheck("image"))
		bytes += memory->usage(image, nmax);
	if (atom->memcheck("x"))
		bytes += memory->usage(x, nmax, 3);
	if (atom->memcheck("v"))
		bytes += memory->usage(v, nmax, 3);
	if (atom->memcheck("f"))
		bytes += memory->usage(f, nmax*comm->nthreads, 3);
	if (atom->memcheck("rmass"))
		bytes += memory->usage(rmass, nmax);
	if (atom->memcheck("rho"))
		bytes += memory->usage(rho, nmax);
	if (atom->memcheck("colorgradient"))
	        bytes += memory->usage(colorgradient, nmax, 3);
	if (atom->memcheck("drho"))
		bytes += memory->usage(drho, nmax*comm->nthreads);
	if (atom->memcheck("e"))
		bytes += memory->usage(e, nmax);
	if (atom->memcheck("de"))
		bytes += memory->usage(de, nmax*comm->nthreads);
	if (atom->memcheck("cv"))
		bytes += memory->usage(cv, nmax);
	if (atom->memcheck("vest"))
		bytes += memory->usage(vest, nmax);

  if (atom->memcheck("epsilon")) bytes += memory->usage(epsilon_,nmax,3,3);
  if (atom->memcheck("sigma")) bytes += memory->usage(sigma_,nmax,3,3);
  if (atom->memcheck("epsilonBar")) bytes += memory->usage(epsilonBar_,nmax,3,3);
  if (atom->memcheck("R")) bytes += memory->usage(R_,nmax,3,3);
  if (atom->memcheck("tau")) bytes += memory->usage(tau_,nmax,3,3);
  if (atom->memcheck("p")) bytes += memory->usage(p_,nmax);
  if (atom->memcheck("viscosity")) bytes += memory->usage(viscosity_,nmax);

	return bytes;
}
