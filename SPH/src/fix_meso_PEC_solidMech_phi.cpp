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
#include "fix_meso_PEC_solidMech_phi.h"
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
#include "modify.h"

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMesoPECSolidMechPhi::FixMesoPECSolidMechPhi(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg) {

  if ((atom->e_flag != 1) || (atom->rho_flag != 1))
    error->all(FLERR,
        "fix meso/PEC/solidMech/phi command requires atom_style with both energy and density");

  if (narg != 3)
    error->all(FLERR,"Illegal number of arguments for fix meso/PEC/solidMech/phi command");

  time_integrate = 1;

	xOld_ = NULL;
	vOld_ = NULL;
	rhoOld_ = NULL;
	phiOld_ = NULL;
	eOld_ = NULL;
	tauOld_ = NULL;
}

/* ---------------------------------------------------------------------- */

FixMesoPECSolidMechPhi::~FixMesoPECSolidMechPhi()
{
}

/* ---------------------------------------------------------------------- */

int FixMesoPECSolidMechPhi::setmask() {
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= PRE_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMesoPECSolidMechPhi::init() {
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
}

void FixMesoPECSolidMechPhi::setup_pre_force(int vflag)
{
  // set vest equal to v 
  double **v = atom->v;
  double **x = atom->x;
  double ***tau_ = atom->tau_;
  double **vest = atom->vest;
  int *mask = atom->mask;
  double *rho = atom->rho;
  double *phi = atom->phi;
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

void FixMesoPECSolidMechPhi::initial_integrate(int vflag) {
  // update v and x and rho and phi and e of atoms in group

  double **x = atom->x;
  double **v = atom->v;
  double **vXSPH_ = atom->vXSPH_;
  double **f = atom->f;
  double **vest = atom->vest;
  double *rho = atom->rho;
  double *phi = atom->phi;
  double *drho = atom->drho;
  double *dphi = atom->dphi;
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

  vOld_ = atom->vOld_;
  xOld_ = atom->xOld_;
  rhoOld_ = atom->rhoOld_;
  phiOld_ = atom->phiOld_;
  eOld_ = atom->eOld_;
  tauOld_ = atom->tauOld_;

  if (igroup == atom->firstgroup)
    nlocal = atom->nfirst;

  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

//- store the old values

      xOld_[i][0] = x[i][0];
      xOld_[i][1] = x[i][1];
      xOld_[i][2] = x[i][2];

      vOld_[i][0] = v[i][0];
      vOld_[i][1] = v[i][1];
      vOld_[i][2] = v[i][2];

      rhoOld_[i] = rho[i];
      phiOld_[i] = phi[i];
      eOld_[i] = e[i];

	  tauOld_[i][0][0] = tau_[i][0][0];
	  tauOld_[i][0][1] = tau_[i][0][1];
	  tauOld_[i][0][2] = tau_[i][0][2];

	  tauOld_[i][1][0] = tau_[i][1][0];
	  tauOld_[i][1][1] = tau_[i][1][1];
	  tauOld_[i][1][2] = tau_[i][1][2];

	  tauOld_[i][2][0] = tau_[i][2][0];
	  tauOld_[i][2][1] = tau_[i][2][1];
	  tauOld_[i][2][2] = tau_[i][2][2];

//- start the 1st integration	  

      e[i] = eOld_[i] + dtf * de[i]; // half-step update of particle internal energy
      rho[i] = rhoOld_[i] + dtf * drho[i]; // ... and density
      phi[i] = phiOld_[i] + dtf * dphi[i]; // ... and phi

	  tau_[i][0][0] = tauOld_[i][0][0] + dtf * dTau_[i][0][0];
	  tau_[i][0][1] = tauOld_[i][0][1] + dtf * dTau_[i][0][1];
	  tau_[i][0][2] = tauOld_[i][0][2] + dtf * dTau_[i][0][2];
	  tau_[i][1][0] = tauOld_[i][1][0] + dtf * dTau_[i][1][0];
	  tau_[i][1][1] = tauOld_[i][1][1] + dtf * dTau_[i][1][1];
	  tau_[i][1][2] = tauOld_[i][1][2] + dtf * dTau_[i][1][2];
	  tau_[i][2][0] = tauOld_[i][2][0] + dtf * dTau_[i][2][0];
	  tau_[i][2][1] = tauOld_[i][2][1] + dtf * dTau_[i][2][1];
	  tau_[i][2][2] = tauOld_[i][2][2] + dtf * dTau_[i][2][2];

      x[i][0] = xOld_[i][0] + 0.5 * dtv * (vOld_[i][0] + vXSPH_[i][0]);
      x[i][1] = xOld_[i][1] + 0.5 * dtv * (vOld_[i][1] + vXSPH_[i][1]);
      x[i][2] = xOld_[i][2] + 0.5 * dtv * (vOld_[i][2] + vXSPH_[i][2]);

      v[i][0] = vOld_[i][0] + dtfm * f[i][0]; // + vXSPH_[i][0];
      v[i][1] = vOld_[i][1] + dtfm * f[i][1]; // + vXSPH_[i][1];
      v[i][2] = vOld_[i][2] + dtfm * f[i][2]; // + vXSPH_[i][2];

      vest[i][0] = vOld_[i][0] + dtfm * f[i][0]; // + vXSPH_[i][0];
      vest[i][1] = vOld_[i][1] + dtfm * f[i][1]; // + vXSPH_[i][1];
      vest[i][2] = vOld_[i][2] + dtfm * f[i][2]; // + vXSPH_[i][2];

    }
  }

}

/* ---------------------------------------------------------------------- */

void FixMesoPECSolidMechPhi::final_integrate() {

  // update v, rho, phi and e of atoms in group

  double **v = atom->v;
  double **vest = atom->vest;
  double **x = atom->x;
  double **vXSPH_ = atom->vXSPH_;
  double **f = atom->f;
  double *e = atom->e;
  double *de = atom->de;
  double *rho = atom->rho;
  double *phi = atom->phi;
  double *drho = atom->drho;
  double *dphi = atom->dphi;
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

  vOld_ = atom->vOld_;
  xOld_ = atom->xOld_;
  rhoOld_ = atom->rhoOld_;
  phiOld_ = atom->phiOld_;
  eOld_ = atom->eOld_;
  tauOld_ = atom->tauOld_;

  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {

      if (rmass_flag) {
        dtfm = dtf / rmass[i];
      } else {
        dtfm = dtf / mass[type[i]];
      }

      e[i] = eOld_[i] + 2.0 * dtf * de[i];
      rho[i] = rhoOld_[i] + 2.0 * dtf * drho[i];
      phi[i] = phiOld_[i] + 2.0 * dtf * dphi[i];

	  tau_[i][0][0] = tauOld_[i][0][0] + 2.0 * dtf * dTau_[i][0][0];
	  tau_[i][0][1] = tauOld_[i][0][1] + 2.0 * dtf * dTau_[i][0][1];
	  tau_[i][0][2] = tauOld_[i][0][2] + 2.0 * dtf * dTau_[i][0][2];
	  tau_[i][1][0] = tauOld_[i][1][0] + 2.0 * dtf * dTau_[i][1][0];
	  tau_[i][1][1] = tauOld_[i][1][1] + 2.0 * dtf * dTau_[i][1][1];
	  tau_[i][1][2] = tauOld_[i][1][2] + 2.0 * dtf * dTau_[i][1][2];
	  tau_[i][2][0] = tauOld_[i][2][0] + 2.0 * dtf * dTau_[i][2][0];
	  tau_[i][2][1] = tauOld_[i][2][1] + 2.0 * dtf * dTau_[i][2][1];
	  tau_[i][2][2] = tauOld_[i][2][2] + 2.0 * dtf * dTau_[i][2][2];
	  
      x[i][0] = xOld_[i][0] + dtv * (v[i][0] + vXSPH_[i][0]);
      x[i][1] = xOld_[i][1] + dtv * (v[i][1] + vXSPH_[i][1]);
      x[i][2] = xOld_[i][2] + dtv * (v[i][2] + vXSPH_[i][2]);

      v[i][0] = vOld_[i][0] + 2.0 * dtfm * f[i][0];// + vXSPH_[i][0];
      v[i][1] = vOld_[i][1] + 2.0 * dtfm * f[i][1];// + vXSPH_[i][1];
      v[i][2] = vOld_[i][2] + 2.0 * dtfm * f[i][2]; + vXSPH_[i][2];

      vest[i][0] = vOld_[i][0] + 2.0 * dtfm * f[i][0]; // + vXSPH_[i][0];
      vest[i][1] = vOld_[i][1] + 2.0 * dtfm * f[i][1]; // + vXSPH_[i][1];
      vest[i][2] = vOld_[i][2] + 2.0 * dtfm * f[i][2]; // + vXSPH_[i][2];

    }
  }
}

/* ---------------------------------------------------------------------- */

void FixMesoPECSolidMechPhi::reset_dt() {
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
//  dtdTau = 0.5 * update->dt * force->ftm2v;
}

