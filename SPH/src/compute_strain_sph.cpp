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

#include <stdlib.h>
#include <string.h>
#include "compute_strain_sph.h"
#include "atom.h"
#include "update.h"
#include "comm.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "modify.h"
#include "fix.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeStrainSph::ComputeStrainSph(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal compute strain/sph command");

  peratom_flag = 1;
  size_peratom_cols = 6;
  pressatomflag = 1;
  timeflag = 1;
  comm_reverse = 6;

  // store temperature ID used by strain computation
  // insure it is valid for temperature computation

  nmax = 0;
  strain = NULL;
  firsttimestep = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

ComputeStrainSph::~ComputeStrainSph()
{
  memory->destroy(strain);
}

/* ---------------------------------------------------------------------- */

void ComputeStrainSph::init()
{
}

/* ---------------------------------------------------------------------- */

void ComputeStrainSph::compute_peratom()
{
  int i,j;

  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local strain array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(strain);
    nmax = atom->nmax;
    memory->create(strain,nmax,6,"strain/sph:strain");
    array_atom = strain;
  }

  // npair includes ghosts if either newton flag is set
  //   b/c some bonds/dihedrals call pair::ev_tally with pairwise info
  // nbond includes ghosts if newton_bond is set
  // ntotal includes ghosts if either newton flag is set
  // KSpace includes ghosts if tip4pflag is set

  int nlocal = atom->nlocal;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // clear local strain array

  if(update->ntimestep == firsttimestep) 
	  for (i = 0; i < ntotal; i++)
		for (j = 0; j < 6; j++)
		  strain[i][j] = 0.0;

  // communicate ghost virials between neighbor procs

  if (force->newton)
    comm->reverse_comm_compute(this);

  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;
  double ***epsilon = atom->epsilon_;

  for (i = 0; i < nlocal; i++)
	if (!(mask[i] & groupbit)) {
	  strain[i][0] = 0.0;
	  strain[i][1] = 0.0;
	  strain[i][2] = 0.0;
	  strain[i][3] = 0.0;
	  strain[i][4] = 0.0;
	  strain[i][5] = 0.0;
	}


  double dt = update->dt;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      strain[i][0] += epsilon[i][0][0]*dt;
      strain[i][1] += epsilon[i][1][1]*dt;
      strain[i][2] += epsilon[i][2][2]*dt;
      strain[i][3] += epsilon[i][0][1]*dt;
      strain[i][4] += epsilon[i][0][2]*dt;
      strain[i][5] += epsilon[i][1][2]*dt;
    }
}

/* ---------------------------------------------------------------------- */

int ComputeStrainSph::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = strain[i][0];
    buf[m++] = strain[i][1];
    buf[m++] = strain[i][2];
    buf[m++] = strain[i][3];
    buf[m++] = strain[i][4];
    buf[m++] = strain[i][5];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeStrainSph::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    strain[j][0] += buf[m++];
    strain[j][1] += buf[m++];
    strain[j][2] += buf[m++];
    strain[j][3] += buf[m++];
    strain[j][4] += buf[m++];
    strain[j][5] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeStrainSph::memory_usage()
{
  double bytes = nmax*6 * sizeof(double);
  return bytes;
}
