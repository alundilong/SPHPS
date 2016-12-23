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
#include "compute_sph_strain_atom.h"
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

ComputeSphStrainAtom::ComputeSphStrainAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute sph/strain/atom command");

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
  strain = NULL;
}

/* ---------------------------------------------------------------------- */

ComputeSphStrainAtom::~ComputeSphStrainAtom()
{
  memory->destroy(strain);
}

/* ---------------------------------------------------------------------- */

void ComputeSphStrainAtom::compute_peratom()
{
  int i, j;
  invoked_peratom = update->ntimestep;
  if (update->vflag_atom != invoked_peratom)
    error->all(FLERR,"Per-atom virial was not tallied on needed timestep");

  // grow local strain array if necessary
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(strain);
    nmax = atom->nmax;
    memory->create(strain,nmax, 9,"strain/atom:strain");
    array_atom = strain;
  }

  int nlocal = atom->nlocal;

  // communicate ghost atom virials between neighbor procs

  if (force->newton) comm->reverse_comm_compute(this);


  // zero virial of atoms not in group
  // only do this after comm since ghost contributions must be included

  int *mask = atom->mask;

  for (i = 0; i < nlocal; i++)
    if (groupbit) {
      strain[i][0] = atom->epsilon_[i][0][0];
      strain[i][1] = atom->epsilon_[i][0][1];
      strain[i][2] = atom->epsilon_[i][0][2];
      strain[i][3] = atom->epsilon_[i][1][0];
      strain[i][4] = atom->epsilon_[i][1][1];
      strain[i][5] = atom->epsilon_[i][1][2];
      strain[i][6] = atom->epsilon_[i][2][0];
      strain[i][7] = atom->epsilon_[i][2][1];
      strain[i][8] = atom->epsilon_[i][2][2];
    }
}

/* ---------------------------------------------------------------------- */

int ComputeSphStrainAtom::pack_reverse_comm(int n, int first, double *buf)
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
    buf[m++] = strain[i][6];
    buf[m++] = strain[i][7];
    buf[m++] = strain[i][8];
  }
  return 9;
}

/* ---------------------------------------------------------------------- */

void ComputeSphStrainAtom::unpack_reverse_comm(int n, int *list, double *buf)
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
    strain[j][6] += buf[m++];
    strain[j][7] += buf[m++];
    strain[j][8] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSphStrainAtom::memory_usage()
{
  double bytes = nmax*9 * sizeof(double);
  return bytes;
}
