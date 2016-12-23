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

PairStyle(sph/rhosum/shephard,PairSPHRhoSumShephard)

#else

#ifndef LMP_PAIR_SPH_RHOSUM_SHEPHARD_H
#define LMP_PAIR_SPH_RHOSUM_SHEPHARD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHRhoSumShephard : public Pair {
 public:
  PairSPHRhoSumShephard(class LAMMPS *);
  virtual ~PairSPHRhoSumShephard();
  void init_style();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  /// Reports the memory usage of this pair style to LAMMPS.
  double memory_usage();

 protected:
  double **cut, *rho_old, *wf_corr;
  int nstep, first;
  int nmax;                   // allocated size of per-atom arrays

  void allocate();
};

}

#endif
#endif
