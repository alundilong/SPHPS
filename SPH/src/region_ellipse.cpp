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

#include "math.h"
#include "stdlib.h"
#include "string.h"
#include "region_ellipse.h"
#include "update.h"
#include "input.h"
#include "variable.h"
#include "error.h"
#include "force.h"

using namespace LAMMPS_NS;

enum{CONSTANT,VARIABLE};

/* ---------------------------------------------------------------------- */

RegEllipse::RegEllipse(LAMMPS *lmp, int narg, char **arg) :
  Region(lmp, narg, arg)
{
  options(narg-8,&arg[8]);

  xc = xscale*force->numeric(FLERR,arg[2]);
  yc = yscale*force->numeric(FLERR,arg[3]);
  zc = zscale*force->numeric(FLERR,arg[4]);

  rstr = NULL;
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    rstr = new char[n];
    strcpy(rstr,&arg[5][2]);
    radiusa = 0.0;
    rstylea = VARIABLE;
    varshape = 1;
    variable_check();
    shape_update();
  } else {
    radiusa = xscale*force->numeric(FLERR,arg[5]);
    rstylea = CONSTANT;
  }
  if (strstr(arg[6],"v_") == arg[6]) {
    int n = strlen(&arg[6][2]) + 1;
    rstr = new char[n];
    strcpy(rstr,&arg[6][2]);
    radiusb = 0.0;
    rstyleb = VARIABLE;
    varshape = 1;
    variable_check();
    shape_update();
  } else {
    radiusb = yscale*force->numeric(FLERR,arg[6]);
    rstyleb = CONSTANT;
  }
  if (strstr(arg[7],"v_") == arg[7]) {
    int n = strlen(&arg[7][2]) + 1;
    rstr = new char[n];
    strcpy(rstr,&arg[7][2]);
    radiusc = 0.0;
    rstylec = VARIABLE;
    varshape = 1;
    variable_check();
    shape_update();
  } else {
    radiusc = zscale*force->numeric(FLERR,arg[7]);
    rstylec = CONSTANT;
  }

  // error check

  if (radiusa < 0.0 || radiusb < 0.0 || radiusc < 0.0) error->all(FLERR,"Illegal region ellipse command! Negative radius is not allowed!");

  // extent of ellipse
  // for variable radius, uses initial radius

  if (interior) {
    bboxflag = 1;
    extent_xlo = xc - radiusa;
    extent_xhi = xc + radiusa;
    extent_ylo = yc - radiusb;
    extent_yhi = yc + radiusb;
    extent_zlo = zc - radiusc;
    extent_zhi = zc + radiusc;
  } else bboxflag = 0;

  cmax = 1;
  contact = new Contact[cmax];
}

/* ---------------------------------------------------------------------- */

RegEllipse::~RegEllipse()
{
  delete [] rstr;
  delete [] contact;
}

/* ---------------------------------------------------------------------- */

void RegEllipse::init()
{
  Region::init();
  if (rstr) variable_check();
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is inside or on surface
   inside = 0 if x,y,z is outside and not on surface
------------------------------------------------------------------------- */

int RegEllipse::inside(double x, double y, double z)
{
  double delx = x - xc;
  double dely = y - yc;
  double delz = z - zc;
  double rsq = delx*delx/radiusa/radiusa + dely*dely/radiusb/radiusb + delz*delz/radiusc/radiusc;

  if (rsq <= 1.0) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from inner surface of ellipse
   no contact if outside (possible if called from union/intersect)
   delxyz = vector from nearest point on ellipse to x
   special case: no contact if x is at center of ellipse
------------------------------------------------------------------------- */

int RegEllipse::surface_interior(double *x, double cutoff)
{
  double delx = x[0] - xc;
  double dely = x[1] - yc;
  double delz = x[2] - zc;
  double rsq = delx*delx/radiusa/radiusa + dely*dely/radiusb/radiusb + delz*delz/radiusc/radiusc;
  if (rsq > 1.0 || rsq == 0.0) return 0;

  double r = sqrt(delx*delx + dely*dely + delz*delz);
  double radius = sqrt((delx*delx + dely*dely + delz*delz)/rsq);
  double delta = radius - r;
  if (delta < cutoff) {
    contact[0].r = delta;
    contact[0].delx = delx*(1.0-radius/r);
    contact[0].dely = dely*(1.0-radius/r);
    contact[0].delz = delz*(1.0-radius/r);
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from outer surface of ellipse
   no contact if inside (possible if called from union/intersect)
   delxyz = vector from nearest point on ellipse to x
------------------------------------------------------------------------- */

int RegEllipse::surface_exterior(double *x, double cutoff)
{
  double delx = x[0] - xc;
  double dely = x[1] - yc;
  double delz = x[2] - zc;
  double rsq = delx*delx/radiusa/radiusa + dely*dely/radiusb/radiusb + delz*delz/radiusc/radiusc;
  if (rsq < 1) return 0;

  double r = sqrt(delx*delx + dely*dely + delz*delz);
  double radius = sqrt((delx*delx + dely*dely + delz*delz)/rsq);
  double delta = rsq - radius;
  if (delta < cutoff) {
    contact[0].r = delta;
    contact[0].delx = delx*(1.0-radius/r);
    contact[0].dely = dely*(1.0-radius/r);
    contact[0].delz = delz*(1.0-radius/r);
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   change region shape via variable evaluation
------------------------------------------------------------------------- */

void RegEllipse::shape_update()
{
  radiusa = xscale * input->variable->compute_equal(rvar);
  radiusb = yscale * input->variable->compute_equal(rvar);
  radiusc = zscale * input->variable->compute_equal(rvar);
  if (radiusa < 0.0 || radiusb < 0.0 || radiusc < 0.0)
    error->one(FLERR,"Variable evaluation in region gave bad value");
}

/* ----------------------------------------------------------------------
   error check on existence of variable
------------------------------------------------------------------------- */

void RegEllipse::variable_check()
{
  rvar = input->variable->find(rstr);
  if (rvar < 0)
    error->all(FLERR,"Variable name for region ellipse does not exist");
  if (!input->variable->equalstyle(rvar))
    error->all(FLERR,"Variable for region ellipse is invalid style");
}
