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

FixStyle(wall/solidcontact,FixWallSolidContact)

#else

#ifndef LMP_FIX_WALL_SPH_H
#define LMP_FIX_WALL_SPH_H

#include "fix_wall.h"

namespace LAMMPS_NS {

class FixWallSolidContact : public Fix {
 public:
  FixWallSolidContact(class LAMMPS *, int, char **);

  int setmask();
  virtual void init();
  void setup(int);
  void min_setup(int);
  void pre_force(int);
  void post_force(int);
  void post_force_respa(int, int, int);
  void min_post_force(int);
  double compute_scalar();
  double compute_vector(int);

  void wall_particle(int, int, double, double);
  void coeff(int, char **);

 private:
  double cutoff[6];
  int eflag;                  // per-wall flag for energy summation
  int xflag;
  double dt;
  int nlevels_respa;
  int nwall;
  int wallwhich[6];
  double xscale,yscale,zscale;
  double coord0[6];
  double vel0[6];
  char *xstr[6];
  int xindex[6];
  int xstyle[6];
  char *vxstr[6];
  int vxindex[6];
  int vxstyle[6];
  int varflag;                // 1 if any wall position,epsilon,sigma is a var
  double ewall[7],ewall_all[7];
//  char *vxvarstr,*vyvarstr,*vzvarstr;
//  int vxvar, vyvar, vzvar;
  double h, Ei, Ej, d0, thetac;
  int vxflag,vyflag,vzflag;
  double vx,vy,vz;
//  int vxvarstyle,vyvarstyle,vzvarstyle;
  double **velocity;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Particle on or inside fix wall surface

Particles must be "exterior" to the wall in order for energy/force to
be calculated.

*/
