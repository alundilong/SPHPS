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
#include "compute_sph_colorgradient_atom.h"
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

ComputeSphColorgradientAtom::ComputeSphColorgradientAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph/colorgradient/atom command");

  peratom_flag = 1;
  /*
   * x y z
  */
  size_peratom_cols = 3;
  pressatomflag = 1;
  timeflag = 1;
  comm_reverse = 3;

  nmax = 0;
  colorgradient = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSphColorgradientAtom::~ComputeSphColorgradientAtom()
{
  memory->destroy(colorgradient);
}

/* ---------------------------------------------------------------------- */

void ComputeSphColorgradientAtom::compute_peratom()
{
  int i, j;
  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local colorgradient array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(colorgradient);
    nmax = atom->nmax;
    memory->create(colorgradient,nmax, 3,"colorgradient/atom:colorgradient");
    array_atom = colorgradient;
  }

  int nlocal = atom->nlocal;

  // communicate ghost atom virials between neighbor procs

  if (force->newton) comm->reverse_comm_compute(this);


  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (groupbit) {
      colorgradient[i][0] = atom->colorgradient[i][0];
      colorgradient[i][1] = atom->colorgradient[i][1];
      colorgradient[i][2] = atom->colorgradient[i][2];
    }
}

/* ---------------------------------------------------------------------- */

int ComputeSphColorgradientAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = colorgradient[i][0];
    buf[m++] = colorgradient[i][1];
    buf[m++] = colorgradient[i][2];
  }
  return 3;
}

/* ---------------------------------------------------------------------- */

void ComputeSphColorgradientAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    colorgradient[j][0] += buf[m++];
    colorgradient[j][1] += buf[m++];
    colorgradient[j][2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSphColorgradientAtom::memory_usage()
{
  double bytes = nmax*3 * sizeof(double);
  return bytes;
}
