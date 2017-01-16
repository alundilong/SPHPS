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

#ifdef FIX_CLASS

FixStyle(3D/printing,Fix3DPrinting)

#else

#ifndef LMP_FIX_3D_PRINTING_H
#define LMP_FIX_3D_PRINTING_H

#include "stdio.h"
#include "fix.h"
#include <vector>

namespace LAMMPS_NS {

class Fix3DPrinting : public Fix {
 public:
  Fix3DPrinting(class LAMMPS *, int, char **);
  ~Fix3DPrinting();
  int setmask();
  void init();
  void pre_exchange();
  void post_force(int);
  void write_restart(FILE *);
  void restart(char *);

 private:
  char * fileName;
  double fx, fy, fz;
  int ninsert,ntype,nfreq;
  int iregion,globalflag,localflag,maxattempt,rateflag,scaleflag,targetflag;
  int mode,rigidflag,shakeflag,idnext;
  double lo,hi,deltasq,nearsq,rate;
  double vxlo,vxhi,vylo,vyhi,vzlo,vzhi;
  double xlo,xhi,ylo,yhi,zlo,zhi;
  double tx,ty,tz;

  int natom_max;
  double **coords;
  imageint *imageflags;

  int nfirst,ninserted;
  tagint maxtag_all,maxmol_all;

  struct ele {
      int pType;
      double px;
      double py;
      double pz;
  };

  std::vector<std::vector<ele> > sphElements;
  std::vector<std::vector<ele> >::const_iterator it;
  std::vector<int> insertedIds;


  void find_maxid();
  void loadSPHElements(int&, int&, char*);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Invalid atom type in fix 3D/printing command

Self-explanatory.

E: Must specify a region in fix 3D/printing

The region keyword must be specified with this fix.

E: Fix 3D/printing region does not support a bounding box

Not all regions represent bounded volumes.  You cannot use
such a region with the fix 3D/printing command.

E: Fix 3D/printing region cannot be dynamic

Only static regions can be used with fix 3D/printing.

E: Deposition region extends outside simulation box

Self-explanatory.

E: Cannot use fix_3D/printing unless atoms have IDs

Self-explanatory.

E: Fix 3D/printing molecule must have coordinates

The defined molecule does not specify coordinates.

E: Fix 3D/printing molecule must have atom types

The defined molecule does not specify atom types.

E: Invalid atom type in fix 3D/printing mol command

The atom types in the defined molecule are added to the value
specified in the create_atoms command, as an offset.  The final value
for each atom must be between 1 to N, where N is the number of atom
types.

E: Fix 3D/printing molecule template ID must be same as atom_style template ID

When using atom_style template, you cannot 3D/printing molecules that are
not in that template.

E: Cannot use fix 3D/printing rigid and not molecule

Self-explanatory.

E: Cannot use fix 3D/printing shake and not molecule

Self-explanatory.

E: Cannot use fix 3D/printing rigid and shake

These two attributes are conflicting.

E: Region ID for fix 3D/printing does not exist

Self-explanatory.

E: Fix pour rigid fix does not exist

Self-explanatory.

E: Fix 3D/printing and fix rigid/small not using same molecule template ID

Self-explanatory.

E: Fix 3D/printing shake fix does not exist

Self-explanatory.

E: Fix 3D/printing and fix shake not using same molecule template ID

Self-explanatory.

W: Particle 3D/printingion was unsuccessful

The fix 3D/printing command was not able to insert as many atoms as
needed.  The requested volume fraction may be too high, or other atoms
may be in the insertion region.

E: Too many total atoms

See the setting for bigint in the src/lmptype.h file.

E: New atom IDs exceed maximum allowed ID

See the setting for tagint in the src/lmptype.h file.

E: Molecule template ID for fix 3D/printing does not exist

Self-explanatory.

W: Molecule template for fix 3D/printing has multiple molecules

The fix 3D/printing command will only create molecules of a single type,
i.e. the first molecule in the template.

*/
