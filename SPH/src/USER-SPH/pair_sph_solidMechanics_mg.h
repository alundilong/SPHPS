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

PairStyle(sph/solidMechanics/mg,PairSPHSolidMechanicsMG)

#else

#ifndef LMP_PAIR_SOLIDMECHANICS_MG_H
#define LMP_PAIR_SOLIDMECHANICS_MG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSPHSolidMechanicsMG : public Pair {
 public:
  PairSPHSolidMechanicsMG(class LAMMPS *);
  virtual ~PairSPHSolidMechanicsMG();
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

  double G_; // shear modulus
  double Gamma_;
  double Cs_;
  double c_; // sound speed
  double S_; // slope ?
  double a0_, b0_, c0_; // constants
  double rho0_; // reference density
  double Y0_; // Yield Stress

  bool update_drho_flag_;
  double eps_;
  double epsXSPH_;
  double deltap_;
  double **beta_;
  double h_;

};

}

#endif
#endif
