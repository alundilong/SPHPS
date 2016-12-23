/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(number/control,FixNumberControl)

#else

#ifndef LMP_FIX_NUMBER_CONTROL_H
#define LMP_FIX_NUMBER_CONTROL_H

#include "stdio.h"
#include "fix.h"

namespace LAMMPS_NS {

class FixNumberControl : public Fix {
 public:
  FixNumberControl(class LAMMPS *, int, char **);
  ~FixNumberControl();
  int setmask();
  void init();
  virtual void pre_exchange();
  void write_restart(FILE *);
  void restart(char *);
  void *extract(const char *, int &);

 private:
  static int nfix_fluxBC_random;
  int fix_id_local;
  int ninsert,ntype,nfreq,seed;
  int iregion,globalflag,localflag,maxattempt,rateflag,scaleflag,targetflag,ifix;
  int mode,rigidflag,shakeflag,idnext;
  double lo,hi,deltasq,nearsq,rate;
  double vxlo,vxhi,vylo,vyhi,vzlo,vzhi;
  double xlo,xhi,ylo,yhi,zlo,zhi;
  double tx,ty,tz;
  char *idregion,*idfix;
  char *idrigid,*idshake;

  class Molecule **onemols;
  int nmol,natom_max;
  double *molfrac;
  double **coords;
  imageint *imageflags;
  class Fix *fixrigid,*fixshake;
  double oneradius;

  int nfirst,ninserted;
  tagint maxtag_all,maxmol_all;
  class RanPark *random;

  void find_maxid();
  void options(int, char **);

 private:
  // number of atoms in the region
  int num, numfixed;
  int nparticles;

  void insert_atom();
  void insert_molecule();

  int molflag;
  int ndeleted;
  int nflux;
  int nmax;
  int *list, *mark;
  void cleanRegion();
  void remove_atom();
  void remove_molecule();

  double vbias[3];
  double T0;

  void compute_num();

  int xstyle, ystyle, zstyle;
  int estyle;

  int num_atoms;
  int molecules_in_region(int&, int&);

};

}

#endif
#endif

/* ERROR/WARNING messages:

*/
