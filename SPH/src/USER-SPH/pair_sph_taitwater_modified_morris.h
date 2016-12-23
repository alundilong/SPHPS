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

#ifdef PAIR_CLASS

PairStyle(sph/taitwater/modified/morris,PairSPHTaitwaterModifiedMorris)

#else

#ifndef LMP_PAIR_TAITWATER_MODIFIED_PHI_H
#define LMP_PAIR_TAITWATER_MODIFIED_PHI_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHTaitwaterModifiedMorris : public Pair {
 public:
  PairSPHTaitwaterModifiedMorris(class LAMMPS *);
  virtual ~PairSPHTaitwaterModifiedMorris();
  void init_style();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);

 protected:
  double h_;
  double **cut,**viscosity;
  int first;

  void allocate();
 private:
  int liquid_type;
  double rmass_liquid;
  double phi0_one;
  double b_one;
  double c_one;
};

}

#endif
#endif
