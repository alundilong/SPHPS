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

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "fix_setvelocity.h"
#include "atom.h"
#include "force.h"
#include "group.h"
#include "update.h"
#include "domain.h"
#include "region.h"
#include "comm.h"
#include "input.h"
#include "variable.h"
#include "modify.h"
#include "compute.h"
#include "error.h"
#include "memory.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL,ATOM};

/* ---------------------------------------------------------------------- */

FixSetVelocity::FixSetVelocity(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 7) error->all(FLERR,"Illegal fix temp/rescale command");

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix temp/rescale command");

  global_freq = nevery;
  extscalar = 1;

  xstr = ystr = zstr = NULL;

  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[4][2]);
  } else if (strcmp(arg[4],"NULL") == 0) {
    xstyle = NONE;
  } else {
    xvalue = force->numeric(FLERR,arg[4]);
    xstyle = CONSTANT;
  }
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[5][2]);
  } else if (strcmp(arg[5],"NULL") == 0) {
    ystyle = NONE;
  } else {
    yvalue = force->numeric(FLERR,arg[5]);
    ystyle = CONSTANT;
  }
  if (strstr(arg[6],"v_") == arg[6]) {
    int n = strlen(&arg[6][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[6][2]);
  } else if (strcmp(arg[6],"NULL") == 0) {
    zstyle = NONE;
  } else {
    zvalue = force->numeric(FLERR,arg[6]);
    zstyle = CONSTANT;
  }

  // optional args

  iregion = -1;
  idregion = NULL;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix setvelocity command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix setvelocity does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else error->all(FLERR,"Illegal fix setvelocity command");
  }

  maxatom = 1;
  memory->create(svel,maxatom,3,"setvelocity:svel");
}

/* ---------------------------------------------------------------------- */

FixSetVelocity::~FixSetVelocity()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
  delete [] idregion;
  memory->destroy(svel);
}

/* ---------------------------------------------------------------------- */

int FixSetVelocity::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixSetVelocity::init()
{
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix setvelocity does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else if (input->variable->atomstyle(xvar)) xstyle = ATOM;
    else error->all(FLERR,"Variable for fix setvelocity is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix setvelocity does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else if (input->variable->atomstyle(yvar)) ystyle = ATOM;
    else error->all(FLERR,"Variable for fix setvelocity is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix setvelocity does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else if (input->variable->atomstyle(zvar)) zstyle = ATOM;
    else error->all(FLERR,"Variable for fix setvelocity is invalid style");
  }

  // set index and check validity of region

  if (iregion >= 0) {
    iregion = domain->find_region(idregion);
    if (iregion == -1)
      error->all(FLERR,"Region ID for fix setvelocity does not exist");
  }

  if (xstyle == ATOM || ystyle == ATOM || zstyle == ATOM)
    varflag = ATOM;
  else if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL)
    varflag = EQUAL;
  else varflag = CONSTANT;

}

/* ---------------------------------------------------------------------- */

void FixSetVelocity::end_of_step()
{
  double **v = atom->v;
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  // update region if necessary

  Region *region = NULL;
  if (iregion >= 0) {
    region = domain->regions[iregion];
    region->prematch();
  }

  // reallocate sforce array if necessary

  if (varflag == ATOM && nlocal > maxatom) {
    maxatom = atom->nmax;
    memory->destroy(svel);
    memory->create(svel,maxatom,3,"setvelocity:svel");
  }

  if (varflag == CONSTANT) {
	  for (int i = 0; i < nlocal; i++) {
		if (mask[i] & groupbit) {
			if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
			if(xstyle) v[i][0] = xvalue;
			if(ystyle) v[i][1] = yvalue;
			if(zstyle) v[i][2] = zvalue;
		}
	  }
  } else {
    modify->clearstep_compute();

    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    else if (xstyle == ATOM)
      input->variable->compute_atom(xvar,igroup,&svel[0][0],3,0);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    else if (ystyle == ATOM)
      input->variable->compute_atom(yvar,igroup,&svel[0][1],3,0);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    else if (zstyle == ATOM)
      input->variable->compute_atom(zvar,igroup,&svel[0][2],3,0);

    modify->addstep_compute(update->ntimestep + 1);

	for (int i = 0; i < nlocal; i++) {
	  if (mask[i] & groupbit) {
	  	if (region && !region->match(x[i][0],x[i][1],x[i][2])) continue;
	  	if(xstyle == ATOM) v[i][0] = svel[i][0];
	  	else if(xstyle) v[i][0] = xvalue;
	  	if(ystyle == ATOM) v[i][1] = svel[i][1];
	  	else if(ystyle) v[i][1] = yvalue;
	  	if(zstyle == ATOM) v[i][2] = svel[i][2];
	  	else if(zstyle) v[i][2] = zvalue;
	  }
	}
  }
}

/* ---------------------------------------------------------------------- */

//int FixSetVelocity::modify_param(int narg, char **arg)
//{
//  if (strcmp(arg[0],"temp") == 0) {
//    if (narg < 2) error->all(FLERR,"Illegal fix_modify command");
//    if (tflag) {
//      modify->delete_compute(id_temp);
//      tflag = 0;
//    }
//    delete [] id_temp;
//    int n = strlen(arg[1]) + 1;
//    id_temp = new char[n];
//    strcpy(id_temp,arg[1]);
//
//    int icompute = modify->find_compute(id_temp);
//    if (icompute < 0)
//      error->all(FLERR,"Could not find fix_modify temperature ID");
//    temperature = modify->compute[icompute];
//
//    if (temperature->tempflag == 0)
//      error->all(FLERR,
//                 "Fix_modify temperature ID does not compute temperature");
//    if (temperature->igroup != igroup && comm->me == 0)
//      error->warning(FLERR,"Group for fix_modify temp != fix group");
//    return 2;
//  }
//  return 0;
//}
//
///* ---------------------------------------------------------------------- */
//
//void FixSetVelocity::reset_target(double t_new)
//{
//  t_target = t_start = t_stop = t_new;
//}
//
///* ---------------------------------------------------------------------- */
//
//double FixSetVelocity::compute_scalar()
//{
//  return energy;
//}
//
///* ----------------------------------------------------------------------
//   extract thermostat properties
//------------------------------------------------------------------------- */
//
//void *FixSetVelocity::extract(const char *str, int &dim)
//{
//  if (strcmp(str,"t_target") == 0) {
//    dim = 0;
//    return &t_target;
//  }
//  return NULL;
//}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double FixSetVelocity::memory_usage()
{
  double bytes = 0.0;
  if (varflag == ATOM) bytes = maxatom*3 * sizeof(double);
  return bytes;
}
