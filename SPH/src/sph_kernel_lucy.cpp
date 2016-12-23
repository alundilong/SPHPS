/* ----------------------------------------------------------------------
 LAMMPS-Large-scale Atomic/Molecular Massively Parallel Simulator
 http://lammps.sandia.gov, Sandia National Laboratories
 Steve Plimpton, sjplimp@sandia.gov

 Copyright (2003) Sandia Corporation.  Under the terms of Contract
 DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 certain rights in this software.  This software is distributed under
 the GNU General Public License.

 See the README file in the top-level LAMMPS directory.
 ------------------------------------------------------------------------- */

#include "math.h"
#include "sph_kernel_lucy.h"

double LAMMPS_NS::sph_kernel_lucy3d(double s, double h, double hinv) {
  const double norm3d = 105./(16*M_PI); //0.0716197243913529;
  if (s<1.0) {
    return norm3d*hinv*hinv*hinv*(1+3*s)*(1-s)*(1-s)*(1-s);
  }
  return 0.0;
}

double LAMMPS_NS::sph_kernel_lucy2d(double s, double h, double hinv) {
  const double norm3d = 5./(M_PI); //0.0716197243913529;
  if (s<1.0) {
    return norm3d*hinv*hinv*(1+3*s)*(1-s)*(1-s)*(1-s);
  }
  return 0.0;
}

double LAMMPS_NS::sph_dw_lucy3d(double s, double h, double hinv) {
  const double norm3d = -12*105./(16*M_PI); //0.0716197243913529;
  if (s<1.0) {
    return norm3d*hinv*hinv*hinv*s*(1-s)*(1-s)*hinv;
  }
  return 0.0;
}

double LAMMPS_NS::sph_dw_lucy2d(double s, double h, double hinv) {
  const double norm3d = -12*5./(M_PI); //0.0716197243913529;
  if (s<1.0) {
    return norm3d*hinv*hinv*s*(1-s)*(1-s)*hinv;
  }
  return 0.0;
}

