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

#include "stdio.h"
#include "string.h"
#include "fix_meso_solidMech2.h"
#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "memory.h"
#include "error.h"
#include "pair.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMesoSolidMech2::FixMesoSolidMech2(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) {

  if ((atom->e_flag != 1) || (atom->rho_flag != 1))
    error->all(FLERR,
        "fix meso/solidMech2 command requires atom_style with both energy and density");

  if (narg != 3)
    error->all(FLERR,"Illegal number of arguments for fix meso/solidMech2 command");

  time_integrate = 1;
}

/* ---------------------------------------------------------------------- */

FixMesoSolidMech2::~FixMesoSolidMech2()
{
}

/* ---------------------------------------------------------------------- */

int FixMesoSolidMech2::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMesoSolidMech2::init() {
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

void FixMesoSolidMech2::setup_pre_force(int vflag)
{

  // set vest equal to v 
  double **v = atom->v;
  double **x = atom->x;
  double ***tau_ = atom->tau_;
  double **vest = atom->vest;
  int *mask = atom->mask;
  double *rho = atom->rho;
  double *e = atom->e;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (int i = 0; i < nlocal + atom->nghost; i++) {
    if (mask[i] & groupbit) {
      vest[i][0] = v[i][0];
      vest[i][1] = v[i][1];
      vest[i][2] = v[i][2];
    }
  }
}

/* ----------------------------------------------------------------------
 allow for both per-type and per-atom mass
 ------------------------------------------------------------------------- */

void FixMesoSolidMech2::initial_integrate(int vflag) {
  // update v and x and rho and e of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **vXSPH_ = atom->vXSPH_;
  double **f = atom->f;
  double **vest = atom->vest;
  double *rho = atom->rho;
  double *drho = atom->drho;
  double ***tau_ = atom->tau_;
  double ***dTau_ = atom->dTau_;
  double *e = atom->e;
  double *de = atom->de;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int rmass_flag = atom->rmass_flag;

  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int i;
  double dtfm;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }
	  
      e[i] = e[i] + dtf * de[i]; // half-step update of particle internal energy
      rho[i] = rho[i] + dtf * drho[i]; // ... and density

	  tau_[i][0][0] = tau_[i][0][0] + 2.0 * dtf * dTau_[i][0][0];
	  tau_[i][0][1] = tau_[i][0][1] + 2.0 * dtf * dTau_[i][0][1];
	  tau_[i][0][2] = tau_[i][0][2] + 2.0 * dtf * dTau_[i][0][2];
	  tau_[i][1][0] = tau_[i][1][0] + 2.0 * dtf * dTau_[i][1][0];
	  tau_[i][1][1] = tau_[i][1][1] + 2.0 * dtf * dTau_[i][1][1];
	  tau_[i][1][2] = tau_[i][1][2] + 2.0 * dtf * dTau_[i][1][2];
	  tau_[i][2][0] = tau_[i][2][0] + 2.0 * dtf * dTau_[i][2][0];
	  tau_[i][2][1] = tau_[i][2][1] + 2.0 * dtf * dTau_[i][2][1];
	  tau_[i][2][2] = tau_[i][2][2] + 2.0 * dtf * dTau_[i][2][2];

      vest[i][0] = v[i][0] + 2.0 * dtfm * f[i][0];
      vest[i][1] = v[i][1] + 2.0 * dtfm * f[i][1];
      vest[i][2] = v[i][2] + 2.0 * dtfm * f[i][2];

      v[i][0] = v[i][0] + dtfm * f[i][0];
      v[i][1] = v[i][1] + dtfm * f[i][1];
      v[i][2] = v[i][2] + dtfm * f[i][2];

      x[i][0] = x[i][0] + dtv * (v[i][0] + vXSPH_[i][0]);
      x[i][1] = x[i][1] + dtv * (v[i][1] + vXSPH_[i][1]);
      x[i][2] = x[i][2] + dtv * (v[i][2] + vXSPH_[i][2]);

    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMesoSolidMech2::final_integrate() {

  // update v, rho, and e of atoms in group

  double **v = atom->v;
  double **x = atom->x;
  double **vXSPH_ = atom->vXSPH_;
  double **f = atom->f;
  double *e = atom->e;
  double *de = atom->de;
  double *rho = atom->rho;
  double *drho = atom->drho;
  double ***tau_ = atom->tau_;
  double ***dTau_ = atom->dTau_;
  int *type = atom->type;
  int *mask = atom->mask;
  double *mass = atom->mass;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;
  double dtfm;
  double *rmass = atom->rmass;
  int rmass_flag = atom->rmass_flag;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }
      e[i] = e[i] + dtf * de[i]; // half-step update of particle internal energy
      rho[i] = rho[i] + dtf * drho[i]; // ... and density

      v[i][0] = v[i][0] + dtfm * f[i][0];
      v[i][1] = v[i][1] + dtfm * f[i][1];
      v[i][2] = v[i][2] + dtfm * f[i][2];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMesoSolidMech2::reset_dt() {
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

