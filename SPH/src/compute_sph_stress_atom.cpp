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

#include "stdlib.h"
#include "string.h"
#include "compute_sph_stress_atom.h"
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

ComputeSphStressAtom::ComputeSphStressAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph/stress/atom command");

  peratom_flag = 1;
  /*
   * xx xy xz
   * yx yy yz
   * zx zy zz
  */
  size_peratom_cols = 9;
  pressatomflag = 1;
  timeflag = 1;
  comm_reverse = 9;

  nmax = 0;
  stress = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSphStressAtom::~ComputeSphStressAtom()
{
  memory->destroy(stress);
}

/* ---------------------------------------------------------------------- */

void ComputeSphStressAtom::compute_peratom()
{
  int i, j;
  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local stress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(stress);
    nmax = atom->nmax;
    memory->create(stress,nmax, 9,"stress/atom:stress");
    array_atom = stress;
  }

  int nlocal = atom->nlocal;

  // communicate ghost atom virials between neighbor procs

  if (force->newton) comm->reverse_comm_compute(this);


  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (groupbit) {
      stress[i][0] = atom->sigma_[i][0][0];
      stress[i][1] = atom->sigma_[i][0][1];
      stress[i][2] = atom->sigma_[i][0][2];
      stress[i][3] = atom->sigma_[i][1][0];
      stress[i][4] = atom->sigma_[i][1][1];
      stress[i][5] = atom->sigma_[i][1][2];
      stress[i][6] = atom->sigma_[i][2][0];
      stress[i][7] = atom->sigma_[i][2][1];
      stress[i][8] = atom->sigma_[i][2][2];
    }

  // convert to stress*volume units = -pressure*volume
  // unit of stress is Pa

  double angstrom = force->angstrom;
  double nktv2p = force->nktv2p; ///angstrom/angstrom/angstrom;
  for (i = 0; i < nlocal; i++)
    if (groupbit) {
      stress[i][0] *= nktv2p;
      stress[i][1] *= nktv2p;
      stress[i][2] *= nktv2p;
      stress[i][3] *= nktv2p;
      stress[i][4] *= nktv2p;
      stress[i][5] *= nktv2p;
      stress[i][6] *= nktv2p;
      stress[i][7] *= nktv2p;
      stress[i][8] *= nktv2p;
    }
}

/* ---------------------------------------------------------------------- */

int ComputeSphStressAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = stress[i][0];
    buf[m++] = stress[i][1];
    buf[m++] = stress[i][2];
    buf[m++] = stress[i][3];
    buf[m++] = stress[i][4];
    buf[m++] = stress[i][5];
    buf[m++] = stress[i][6];
    buf[m++] = stress[i][7];
    buf[m++] = stress[i][8];
  }
  return 9;
}

/* ---------------------------------------------------------------------- */

void ComputeSphStressAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    stress[j][0] += buf[m++];
    stress[j][1] += buf[m++];
    stress[j][2] += buf[m++];
    stress[j][3] += buf[m++];
    stress[j][4] += buf[m++];
    stress[j][5] += buf[m++];
    stress[j][6] += buf[m++];
    stress[j][7] += buf[m++];
    stress[j][8] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSphStressAtom::memory_usage()
{
  double bytes = nmax*9 * sizeof(double);
  return bytes;
}
