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

/*-----------------------------------------------------------------------
cubicspline_2D implementation by Markus Schörgenhumer, mkschoe@gmail.com
------------------------------------------------------------------------- */

#ifdef SPH_KERNEL_CLASS

    // kernel identifier (a unique integer >= 0)
    // a name for the kernel
    // name of the functions for the kernel, its derivative, and the cutoff are defined
    SPHKernel
    (
        1,
        cubicspline2d,
        sph_kernel_cubicspline2d,
        sph_kernel_cubicspline2d_der,
        sph_kernel_cubicspline2d_cut
    )

#else

#ifndef LMP_SPH_KERNEL_CUBICSPLINE2D
#define LMP_SPH_KERNEL_CUBICSPLINE2D

namespace SPH_KERNEL_NS {
  inline double sph_kernel_cubicspline2d(double s, double h, double hinv);
  inline double sph_kernel_cubicspline2d_der(double s, double h, double hinv);
  inline double sph_kernel_cubicspline2d_cut();
}

/* ----------------------------------------------------------------------
   Cubic spline SPH kernel
   h is kernel parameter
   s is distance normalized by h
   0.1136821 is 5 over 14pi
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_cubicspline2d(double s, double h, double hinv)
{
    if (s > 2.)
	{
		return 0.0;
	}
	else if (s > 1.)
    {
        return (0.1136821*hinv*hinv * ((2.-s)*(2.-s)*(2.-s)));
    }
    else 
    {
        return (0.1136821*hinv*hinv * ((2.-s)*(2.-s)*(2.-s) - 4.*(1.-s)*(1.-s)*(1.-s)));
    }
}

/* ----------------------------------------------------------------------
   Derivative of cubic spline SPH kernel
   is equal to grad W if multiplied with radial unit vector
   h is kernel parameter
   s is distance normalized by h
   0.1136821 is 5 over 14pi
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_cubicspline2d_der(double s,double h, double hinv)
{
	if(s > 2.0)
	{
		return 0.0;
	}
	else if (s > 1.)
    {
        return (0.1136821*hinv*hinv*hinv * (-3.*(2.-s)*(2.-s)));
    }
    else
    {
        return (0.1136821*hinv*hinv*hinv * (-3.*(2.-s)*(2.-s) + 12.*(1.-s)*(1.-s)));
    }
}

/* ----------------------------------------------------------------------
   Definition of cubic spline SPH kernel cutoff in terms of s
   s is normalized distance
------------------------------------------------------------------------- */

inline double SPH_KERNEL_NS::sph_kernel_cubicspline2d_cut()
{
    return 2.;
}

#endif
#endif

