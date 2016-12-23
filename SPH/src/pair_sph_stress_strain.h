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

#ifdef PAIR_CLASS

PairStyle(sph/stress/strain,PairSPHStressStrain)

#else

#ifndef LMP_PAIR_SPH_STRESS_STRAIN_H
#define LMP_PAIR_SPH_STRESS_STRAIN_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHStressStrain : public Pair {
 public:
  PairSPHStressStrain(class LAMMPS *);
  virtual ~PairSPHStressStrain();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);

 protected:
  double **cut;

  void allocate();
 
private:

//  void computeStrainAndShearStress();
  void solve6By6(double &, double &, double &, double &, double &, double &, double A[6][6], double b[6]);
  double determinant(double f[6][6], int x);

  double G_; // shear modulus
  double Gamma_;
  double Cs_;
  double S_; // slope ?
  double a0_, b0_, c0_; // constants
  double rho0_; // reference density
  double Y0_; // Yield Stress

};

}

#endif
#endif
