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

FixStyle(meso/solidMech,FixMesoSolidMech)

#else

#ifndef LMP_FIX_MESO_SOLID_MECH_H
#define LMP_FIX_MESO_SOLID_MECH_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMesoSolidMech : public Fix {
 public:
  FixMesoSolidMech(class LAMMPS *, int, char **);
  virtual ~FixMesoSolidMech();
  int setmask();
  virtual void init();
  virtual void setup_pre_force(int);
  virtual void initial_integrate(int);
  virtual void final_integrate();
  void reset_dt();

 private:
  class NeighList *list;

  double **xOld_;
  double **vOld_;
  double *rhoOld_;
  double *eOld_;
  double ***tauOld_;

 protected:
  double dtv,dtf;
  double *step_respa;
  int mass_require;

  class Pair *pair;

};

}

#endif
#endif
