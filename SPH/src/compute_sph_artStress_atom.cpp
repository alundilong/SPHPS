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
#include "compute_sph_artStress_atom.h"
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

ComputeSphArtStressAtom::ComputeSphArtStressAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph/artStress/atom command");

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
  artStress = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSphArtStressAtom::~ComputeSphArtStressAtom()
{
  memory->destroy(artStress);
}

/* ---------------------------------------------------------------------- */

void ComputeSphArtStressAtom::compute_peratom()
{
  int i, j;
  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local artStress array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(artStress);
    nmax = atom->nmax;
    memory->create(artStress,nmax, 9,"artStress/atom:artStress");
    array_atom = artStress;
  }

  int nlocal = atom->nlocal;

  // communicate ghost atom virials between neighbor procs

  if (force->newton) comm->reverse_comm_compute(this);


  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (groupbit) {
      artStress[i][0] = atom->artStress_[i][0][0];
      artStress[i][1] = atom->artStress_[i][0][1];
      artStress[i][2] = atom->artStress_[i][0][2];
      artStress[i][3] = atom->artStress_[i][1][0];
      artStress[i][4] = atom->artStress_[i][1][1];
      artStress[i][5] = atom->artStress_[i][1][2];
      artStress[i][6] = atom->artStress_[i][2][0];
      artStress[i][7] = atom->artStress_[i][2][1];
      artStress[i][8] = atom->artStress_[i][2][2];
    }

  // convert to artStress*volume units = -pressure*volume
  // unit of artStress is Pa

  double angstrom = force->angstrom;
  double nktv2p = force->nktv2p; ///angstrom/angstrom/angstrom;
  for (i = 0; i < nlocal; i++)
    if (groupbit) {
      artStress[i][0] *= nktv2p;
      artStress[i][1] *= nktv2p;
      artStress[i][2] *= nktv2p;
      artStress[i][3] *= nktv2p;
      artStress[i][4] *= nktv2p;
      artStress[i][5] *= nktv2p;
      artStress[i][6] *= nktv2p;
      artStress[i][7] *= nktv2p;
      artStress[i][8] *= nktv2p;
    }
}

/* ---------------------------------------------------------------------- */

int ComputeSphArtStressAtom::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = artStress[i][0];
    buf[m++] = artStress[i][1];
    buf[m++] = artStress[i][2];
    buf[m++] = artStress[i][3];
    buf[m++] = artStress[i][4];
    buf[m++] = artStress[i][5];
    buf[m++] = artStress[i][6];
    buf[m++] = artStress[i][7];
    buf[m++] = artStress[i][8];
  }
  return 9;
}

/* ---------------------------------------------------------------------- */

void ComputeSphArtStressAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    artStress[j][0] += buf[m++];
    artStress[j][1] += buf[m++];
    artStress[j][2] += buf[m++];
    artStress[j][3] += buf[m++];
    artStress[j][4] += buf[m++];
    artStress[j][5] += buf[m++];
    artStress[j][6] += buf[m++];
    artStress[j][7] += buf[m++];
    artStress[j][8] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSphArtStressAtom::memory_usage()
{
  double bytes = nmax*9 * sizeof(double);
  return bytes;
}
