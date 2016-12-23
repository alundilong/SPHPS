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

PairStyle(sph/solidMechanics/LowVel,PairSPHSolidMechanicsLowVel)

#else

#ifndef LMP_PAIR_SOLIDMECHANICS_LOWVEL_H
#define LMP_PAIR_SOLIDMECHANICS_LOWVEL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHSolidMechanicsLowVel : public Pair {
 public:
  PairSPHSolidMechanicsLowVel(class LAMMPS *);
  virtual ~PairSPHSolidMechanicsLowVel();
  void init_style();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  virtual double init_one(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);
//  int pack_comm(int, int *, double *, int, int *);
//  void unpack_comm(int, int, double *);

 protected:
  double **cut,**viscosity;

  void allocate();

 
private:

//  void computeStrainAndShearStress();
  void compute_no_drho(int, int);
  void compute_drho(int, int);

  double Tmelt;
  double n;
  double Tref;
  double c_; // sound speed
  double B;
  double C;
  double epsilonDot0;
  double sigma0;
  double m; // temp index
  double epsiloneqv;

  bool update_drho_flag_;
  double eps_;
  double epsXSPH_;
  double deltap_;
  double **beta_;
  double h_;

  // - 
  bool elastic_flag;
  double fyield; // yield surface
  double lambda_dot;
  double J2, J2max;
  double Y0; // Yield Stress
  double G; // shear modulus
  double K; // bulk modulus

  double tolerance;
  int iterMax;

  double tmp_sigma[9];

  bool check_f();
  void find_sigma(int, int);
  void find_lambda_dot(int);
  void find_properties(int);

};

}

#endif
#endif
