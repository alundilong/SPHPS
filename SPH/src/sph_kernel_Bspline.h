/* ----------------------------------------------------------------------
   LIGGGHTS - LAMMPS Improved for General Granular and Granular Heat
   Transfer Simulations

   LIGGGHTS is part of the CFDEMproject
   www.liggghts.com | www.cfdem.com

   Christoph Kloss, christoph.kloss@cfdem.com
   Copyright 2009-2012 JKU Linz
   Copyright 2012-     DCS Computing GmbH, Linz

   LIGGGHTS is based on LAMMPS
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   This software is distributed under the GNU General Public License.

   See the README file in the top-level directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
Contributing author for SPH:
Andreas Aigner (CD Lab Particulate Flow Modelling, JKU)
andreas.aigner@jku.at
------------------------------------------------------------------------- */

#ifdef SPH_KERNEL_CLASS

    // kernel identifier (a unique integer >= 0)
    // a name for the kernel
    // name of the functions for the kernel, its derivative, and the cutoff are defined
    SPHKernel
    (
        7,
        Bspline,
        sph_kernel_Bspline,
        sph_kernel_Bspline_der,
        sph_kernel_Bspline_cut
    )

#else

#ifndef LMP_SPH_KERNEL_BSPLINE
#define LMP_SPH_KERNEL_BSPLINE

namespace SPH_KERNEL_NS {
  inline double sph_kernel_Bspline(double s, double h, double hinv);
  inline double sph_kernel_Bspline_der(double s, double h, double hinv);
  inline double sph_kernel_Bspline_der2(double s, double h, double hinv);
  inline double sph_kernel_Bspline_cut();
}

/* ----------------------------------------------------------------------
   B spline SPH kernel
   h is kernel parameter
   s is distance normalized by h
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_Bspline(double s, double h, double hinv)
{
    if (s < 1.)
    {
        return (15./7. * ((2.-s)*(2.-s)*(2.-s) - 4.*(1.-s)*(1.-s)*(1.-s)));
    }
    else
    {
        return (5./14. * ((2.-s)*(2.-s)*(2.-s)));
    }
}

/* ----------------------------------------------------------------------
   Derivative of B spline SPH kernel
   is equal to grad W if multiplied with radial unit vector
   h is kernel parameter
   s is distance normalized by h
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_Bspline_der(double s,double h, double hinv)
{
    if (s < 1.)
    {
        return (15./7.*hinv * (-3.*(2.-s)*(2.-s) + 12.*(1.-s)*(1.-s)));
    }
    else
    {
        return (5./14.*hinv * (-3.*(2.-s)*(2.-s)));
    }
}

/* ----------------------------------------------------------------------
   2nd Derivative of B spline SPH kernel
   is equal to grad W if multiplied with radial unit vector
   h is kernel parameter
   s is distance normalized by h
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_Bspline_der2(double s,double h, double hinv)
{
    if (s < 1.)
    {
        return (15./7.*hinv*hinv * (6.*(2.-s) - 24.*(1.-s)));
    }
    else
    {
        return (5./14.*hinv*hinv * (6.*(2.-s)));
    }
}

/* ----------------------------------------------------------------------
   Definition of B spline SPH kernel cutoff in terms of s
   s is normalized distance
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_Bspline_cut()
{
    return 2.;
}

#endif
#endif

