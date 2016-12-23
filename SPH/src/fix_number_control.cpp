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
#include "fix_number_control.h"
#include "atom.h"
#include "atom_vec.h"
#include "molecule.h"
#include "force.h"
#include "update.h"
#include "modify.h"
#include "fix.h"
#include "comm.h"
#include "domain.h"
#include "lattice.h"
#include "region.h"
#include "random_park.h"
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "compute.h"
#include "group.h"
#include "force.h"
#include "pair.h"
//#include "math_extra_liggghts.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{ATOM,MOLECULE};
enum{NONE};

#define EPSILON 1.0e6
#define SMALL 1.0e-15
#define MAXIT 2e5
#define BIG MAXTAGINT
#define INVOKED_PERATOM 8

FixNumberControl::FixNumberControl(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 8) error->all(FLERR,"Illegal fix number/control command");

  restart_global = 1;
  time_depend = 1;

  // required args

  numfixed = force->inumeric(FLERR, arg[3]);
  ntype = force->inumeric(FLERR,arg[4]);
  nfreq = force->inumeric(FLERR,arg[5]);
  seed = force->inumeric(FLERR,arg[6]);
  T0 = force->numeric(FLERR,arg[7]);

  if (seed <= 0) error->all(FLERR,"Illegal fix number/control command");

  // read options from end of input line

  options(narg-8,&arg[8]);

  // error check on type

  if (mode == ATOM && (ntype <= 0 || ntype > atom->ntypes))
    error->all(FLERR,"Invalid atom type in fix number/control command");

  // error checks on region and its extent being inside simulation box

  if (iregion == -1) error->all(FLERR,"Must specify a region in fix number/control");
  if (domain->regions[iregion]->bboxflag == 0)
    error->all(FLERR,"Fix fluxBC region does not support a bounding box");
  if (domain->regions[iregion]->dynamic_check())
    error->all(FLERR,"Fix fluxBC region cannot be dynamic");

  xlo = domain->regions[iregion]->extent_xlo;
  xhi = domain->regions[iregion]->extent_xhi;
  ylo = domain->regions[iregion]->extent_ylo;
  yhi = domain->regions[iregion]->extent_yhi;
  zlo = domain->regions[iregion]->extent_zlo;
  zhi = domain->regions[iregion]->extent_zhi;

  if (domain->triclinic == 0) {
    if (xlo < domain->boxlo[0] || xhi > domain->boxhi[0] ||
        ylo < domain->boxlo[1] || yhi > domain->boxhi[1] ||
        zlo < domain->boxlo[2] || zhi > domain->boxhi[2])
      error->all(FLERR,"FluxBCion region extends outside simulation box");
  } else {
    if (xlo < domain->boxlo_bound[0] || xhi > domain->boxhi_bound[0] ||
        ylo < domain->boxlo_bound[1] || yhi > domain->boxhi_bound[1] ||
        zlo < domain->boxlo_bound[2] || zhi > domain->boxhi_bound[2])
      error->all(FLERR,"FluxBCion region extends outside simulation box");
  }

  // error check and further setup for mode = MOLECULE

  if (atom->tag_enable == 0)
    error->all(FLERR,"Cannot use fix_fluxBC unless atoms have IDs");

  if (mode == MOLECULE) {
    for (int i = 0; i < nmol; i++) {
      if (onemols[i]->xflag == 0)
        error->all(FLERR,"Fix fluxBC molecule must have coordinates");
      if (onemols[i]->typeflag == 0)
        error->all(FLERR,"Fix fluxBC molecule must have atom types");
      if (ntype+onemols[i]->ntypes <= 0 || 
          ntype+onemols[i]->ntypes > atom->ntypes) {
		  printf("ntype = %d, molntypes = %d\n", ntype, onemols[i]->ntypes);
        error->all(FLERR,"Invalid atom type in fix number/control mol command");
	  }
      
      if (atom->molecular == 2 && onemols != atom->avec->onemols)
        error->all(FLERR,"Fix fluxBC molecule template ID must be same "
                   "as atom_style template ID");
      onemols[i]->check_attributes(0);

      // fix number/control uses geoemetric center of molecule for insertion
      
      onemols[i]->compute_center();
	  onemols[i]->compute_mass();
    }
  }

  if (rigidflag && mode == ATOM)
    error->all(FLERR,"Cannot use fix number/control rigid and not molecule");
  if (shakeflag && mode == ATOM)
    error->all(FLERR,"Cannot use fix number/control shake and not molecule");
  if (rigidflag && shakeflag)
    error->all(FLERR,"Cannot use fix number/control rigid and shake");

  // setup of coords and imageflags array

  if (mode == ATOM) natom_max = 1;
  else {
    natom_max = 0;
    for (int i = 0; i < nmol; i++)
      natom_max = MAX(natom_max,onemols[i]->natoms);
  }
  memory->create(coords,natom_max,3,"fluxBC:coords");
  memory->create(imageflags,natom_max,"fluxBC:imageflags");

  // setup scaling

  double xscale,yscale,zscale;
  if (scaleflag) {
    xscale = domain->lattice->xlattice;
    yscale = domain->lattice->ylattice;
    zscale = domain->lattice->zlattice;
  }
  else xscale = yscale = zscale = 1.0;

  // apply scaling to all input parameters with dist/vel units

  if (domain->dimension == 2) {
    lo *= yscale;
    hi *= yscale;
    rate *= yscale;
  } else {
    lo *= zscale;
    hi *= zscale;
    rate *= zscale;
  }
  deltasq *= xscale*xscale;
  nearsq *= xscale*xscale;
  vxlo *= xscale;
  vxhi *= xscale;
  vylo *= yscale;
  vyhi *= yscale;
  vzlo *= zscale;
  vzhi *= zscale;
  tx *= xscale;
  ty *= yscale;
  tz *= zscale;

  // find current max atom and molecule IDs if necessary

  if (idnext) find_maxid();

  // random number generator, same for all procs

  random = new RanPark(lmp,seed);

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  nfirst = next_reneighbor;
  ninserted = 0;

  // atom deletion
  nmax = 0;
  list = NULL;
  mark = NULL;
  int nper = 0;
  if(mode == ATOM) nper = 1;
  else nper = onemols[0]->natoms;
}

/* ---------------------------------------------------------------------- */

FixNumberControl::~FixNumberControl()
{
  delete random;
  delete [] molfrac;
  delete [] idrigid;
  delete [] idshake;
  delete [] idregion;
  memory->destroy(coords);
  memory->destroy(imageflags);
  memory->destroy(list);
  memory->destroy(mark);
}

/* ---------------------------------------------------------------------- */

int FixNumberControl::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNumberControl::init()
{
  // set index and check validity of region

  iregion = domain->find_region(idregion);
  if (iregion == -1)
    error->all(FLERR,"Region ID for fix number/control does not exist");

  // if rigidflag defined, check for rigid/small fix
  // its molecule template must be same as this one

  fixrigid = NULL;
  if (rigidflag) {
    int ifix = modify->find_fix(idrigid);
    if (ifix < 0) error->all(FLERR,"Fix pour rigid fix does not exist");
    fixrigid = modify->fix[ifix];
    int tmp;
    if (onemols != (Molecule **) fixrigid->extract("onemol",tmp))
      error->all(FLERR,
                 "Fix fluxBC and fix rigid/small not using "
                 "same molecule template ID");
  }

  // if shakeflag defined, check for SHAKE fix
  // its molecule template must be same as this one

  fixshake = NULL;
  if (shakeflag) {
    int ifix = modify->find_fix(idshake);
    if (ifix < 0) error->all(FLERR,"Fix fluxBC shake fix does not exist");
    fixshake = modify->fix[ifix];
    int tmp;
    if (onemols != (Molecule **) fixshake->extract("onemol",tmp))
      error->all(FLERR,"Fix fluxBC and fix shake not using "
                 "same molecule template ID");
  }

}

/* ----------------------------------------------------------------------
   compute the number of atom/molecule in the region
------------------------------------------------------------------------- */

void FixNumberControl::compute_num()
{
	if(mode == ATOM) {
	  double **x = atom->x;
	  int *mask = atom->mask;
	  int nlocal = atom->nlocal;
	  int *type = atom->type;

	  Region *region = domain->regions[iregion];
	  int count = 0;

	  for (int i = 0; i < nlocal; i++)
		if (mask[i] & groupbit && region->match(x[i][0],x[i][1],x[i][2])) {
			count++;
		}

	  int tarray,tarray_all;
	  tarray = count;
	  MPI_Allreduce(&tarray,&tarray_all,1,MPI_INT,MPI_SUM,world);
	  num = tarray_all;
	} else
	{
		int idlo, idhi;
		idlo = idhi = 0;
		num = molecules_in_region(idlo, idhi);

		//printf("num of moleculse in region = %d, idlo = %d, idhi = %d\n", num, idlo, idhi);
	}
}

/* ----------------------------------------------------------------------
   // similar to compute::molecules_in_region()
   identify molecule IDs with atoms in group
   warn if any atom in group has molecule ID = 0
   warn if any molecule has only some atoms in group
   return Ncount = # of molecules with atoms in group
   set molmap to NULL if molecule IDs include all in range from 1 to Ncount
   else: molecule IDs range from idlo to idhi
         set molmap to vector of length idhi-idlo+1
         molmap[id-idlo] = index from 0 to Ncount-1
         return idlo and idhi
------------------------------------------------------------------------- */

int FixNumberControl::molecules_in_region(int& idlo, int &idhi)
{
  int i;
  int num_atoms_all = 0;
  num_atoms = 0;

  //memory->destroy(molmap);
  int *molmap;
  molmap = NULL;

  Region *region = domain->regions[iregion];
  // find lo/hi molecule ID for any atom in group
  // warn if atom in group has ID = 0

  tagint *molecule = atom->molecule;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  tagint lo = BIG;
  tagint hi = -BIG;
  int flag = 0;

  double **x = atom->x;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit && region->match(x[i][0],x[i][1],x[i][2])) {
      if (molecule[i] == 0) flag = 1;
      lo = MIN(lo,molecule[i]);
      hi = MAX(hi,molecule[i]);
	  num_atoms++;
    }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  MPI_Allreduce(&num_atoms,&num_atoms_all,1,MPI_INT,MPI_SUM,world);
  num_atoms = num_atoms_all;
  if (flagall && comm->me == 0)
    error->warning(FLERR,"Atom with molecule ID = 0 included in "
                   "compute molecule group");

  MPI_Allreduce(&lo,&idlo,1,MPI_LMP_TAGINT,MPI_MIN,world);
  MPI_Allreduce(&hi,&idhi,1,MPI_LMP_TAGINT,MPI_MAX,world);
  if (idlo == BIG) return 0;

  // molmap = vector of length nlen
  // set to 1 for IDs that appear in group across all procs, else 0

  tagint nlen_tag = idhi-idlo+1;
  if (nlen_tag > MAXSMALLINT) 
    error->all(FLERR,"Too many molecules for compute");
  int nlen = (int) nlen_tag;

  memory->create(molmap,nlen,"compute:molmap");
  for (i = 0; i < nlen; i++) molmap[i] = 0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit && region->match(x[i][0],x[i][1],x[i][2]))
      molmap[molecule[i]-idlo] = 1;

  int *molmapall;
  memory->create(molmapall,nlen,"compute:molmapall");
  MPI_Allreduce(molmap,molmapall,nlen,MPI_INT,MPI_MAX,world);

  // nmolecules = # of non-zero IDs in molmap
  // molmap[i] = index of molecule, skipping molecules not in group with -1

  int nmolecules = 0;
  for (i = 0; i < nlen; i++)
    if (molmapall[i]) molmap[i] = nmolecules++;
    else molmap[i] = -1;
  memory->destroy(molmapall);

  /*
  // warn if any molecule has some atoms in group and some not in group

  flag = 0;
  for (i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit && region->match(x[i][0],x[i][1],x[i][2])) continue;
    if (molecule[i] < idlo || molecule[i] > idhi) continue;
    if (molmap[molecule[i]-idlo] >= 0) flag = 1;
  }

  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  //if (flagall && omm->me == 0)
  //  error->warning(FLERR,
  //                "One or more compute molecules has atoms not in group");

  // if molmap simply stores 1 to Nmolecules, then free it

  if (idlo == 1 && idhi == nmolecules && nlen == nmolecules) {
    memory->destroy(molmap);
    molmap = NULL;
  }
  */
  memory->destroy(molmap);
  return nmolecules;

}

/* ----------------------------------------------------------------------
   perform particle insertion
------------------------------------------------------------------------- */

void FixNumberControl::pre_exchange()
{
  // compute_num_momentum();
  compute_num();

  // just return if should not be called on this timestep
  if (next_reneighbor != update->ntimestep) return;

 // printf("next_reneighbor = %d, ntimestep = %d\n", next_reneighbor, update->ntimestep);

  next_reneighbor += nfreq;

  if (num == numfixed) return;
  else if (num < numfixed)
  {
	  // insert particle
	  // usher algorithm
	  // grow insertedAtomID list
	  nparticles = numfixed - num;
	  // nparticles = std::min(nparticles, 5);

	  ninserted = 0;
	  if(molflag && atom->molecule_flag) {
		  // printf("pId = %d, inserting %d molecules\n", comm->me, nparticles);
		  insert_molecule();
	  }
	  else {
		  insert_atom();
	  }
  }
  else if(num > numfixed){
	if(molflag && atom->molecule_flag) 
	{
		int nremove = num - numfixed;

		// if(comm->me == 0) printf("removing %d molecules\n", nremove);

		nflux = onemols[0]->natoms;
		for(int i = 0; i < nremove; i++)
			remove_molecule();
	}
	else 
	{
		int nremove = num - numfixed;
		nflux = 1;
		for(int i = 0; i < nremove; i++)
			remove_atom();
	}
  }
}


/* ----------------------------------------------------------------------
   perform particle insertion
   random algorithm
------------------------------------------------------------------------- */

void FixNumberControl::insert_atom()
{
  int i,j,m,n,nlocalprev,imol,natom,flag,flagall;
  double coord[3],lamda[3],delx,dely,delz,rsq;
  double r[3],vnew[3],rotmat[3][3],quat[4];
  double *newcoord;
  ninserted = 0;

  // following two lines is to compute vbias
  // it will be used to determine the inintial velocity
  // of new added particles
  // find mass total in region
  double masstotal = group->mass(igroup, iregion);
  // compute vcm in region
  group->vcm(igroup,masstotal,vbias,iregion);

  while (ninserted < nparticles) {
	  // just return if should not be called on this timestep

	  // compute current offset = bottom of insertion volume

	  double offset = 0.0;
	  if (rateflag) offset = (update->ntimestep - nfirst) * update->dt * rate;

	  double *sublo,*subhi;
	  if (domain->triclinic == 0) {
		sublo = domain->sublo;
		subhi = domain->subhi;
	  } else {
		sublo = domain->sublo_lamda;
		subhi = domain->subhi_lamda;
	  }

	  // find current max atom and molecule IDs if necessary

	  if (!idnext) find_maxid();

	  // attempt an insertion until successful

	  int dimension = domain->dimension;
	  int nfix = modify->nfix;
	  Fix **fix = modify->fix;

	  int success = 0;
	  int attempt = 0;
	  while (attempt < maxattempt) {
		attempt++;

		// choose random position for new particle within region

		coord[0] = xlo + random->uniform() * (xhi-xlo);
		coord[1] = ylo + random->uniform() * (yhi-ylo);
		coord[2] = zlo + random->uniform() * (zhi-zlo);
		while (domain->regions[iregion]->match(coord[0],coord[1],coord[2]) == 0) {
		  coord[0] = xlo + random->uniform() * (xhi-xlo);
		  coord[1] = ylo + random->uniform() * (yhi-ylo);
		  coord[2] = zlo + random->uniform() * (zhi-zlo);
		}

		// adjust vertical coord by offset

		if (dimension == 2) coord[1] += offset;
		else coord[2] += offset;

		// if global, reset vertical coord to be lo-hi above highest atom
		// if local, reset vertical coord to be lo-hi above highest "nearby" atom
		// local computation computes lateral distance between 2 particles w/ PBC
		// when done, have final coord of atom or center pt of molecule

		if (globalflag || localflag) {
		  int dim;
		  double max,maxall,delx,dely,delz,rsq;

		  if (dimension == 2) {
			dim = 1;
			max = domain->boxlo[1];
		  } else {
			dim = 2;
			max = domain->boxlo[2];
		  }

		  double **x = atom->x;
		  int nlocal = atom->nlocal;
		  for (i = 0; i < nlocal; i++) {
			if (localflag) {
			  delx = coord[0] - x[i][0];
			  dely = coord[1] - x[i][1];
			  delz = 0.0;
			  domain->minimum_image(delx,dely,delz);
			  if (dimension == 2) rsq = delx*delx;
			  else rsq = delx*delx + dely*dely;
			  if (rsq > deltasq) continue;
			}
			if (x[i][dim] > max) max = x[i][dim];
		  }

		  MPI_Allreduce(&max,&maxall,1,MPI_DOUBLE,MPI_MAX,world);
		  if (dimension == 2)
			coord[1] = maxall + lo + random->uniform()*(hi-lo);
		  else
			coord[2] = maxall + lo + random->uniform()*(hi-lo);
		}

		// coords = coords of all atoms
		// for molecule, perform random rotation around center pt
		// apply PBC so final coords are inside box
		// also modify image flags due to PBC

		if (mode == ATOM) {
		  natom = 1;
		  coords[0][0] = coord[0];
		  coords[0][1] = coord[1];
		  coords[0][2] = coord[2];
		  imageflags[0] = ((imageint) IMGMAX << IMG2BITS) |
			((imageint) IMGMAX << IMGBITS) | IMGMAX;
		} else {
		  double rng = random->uniform();
		  imol = 0;
		  while (rng > molfrac[imol]) imol++;
		  natom = onemols[imol]->natoms;
		  if (dimension == 3) {
			r[0] = random->uniform() - 0.5;
			r[1] = random->uniform() - 0.5;
			r[2] = random->uniform() - 0.5;
		  } else {
			r[0] = r[1] = 0.0;
			r[2] = 1.0;
		  }
		  double theta = random->uniform() * MY_2PI;
		  MathExtra::norm3(r);
		  MathExtra::axisangle_to_quat(r,theta,quat);
		  MathExtra::quat_to_mat(quat,rotmat);
		  for (i = 0; i < natom; i++) {
			MathExtra::matvec(rotmat,onemols[imol]->dx[i],coords[i]);
			coords[i][0] += coord[0];
			coords[i][1] += coord[1];
			coords[i][2] += coord[2];

			imageflags[i] = ((imageint) IMGMAX << IMG2BITS) |
			  ((imageint) IMGMAX << IMGBITS) | IMGMAX;
			domain->remap(coords[i],imageflags[i]);
		  }
		}

		// if distance to any inserted atom is less than near, try again
		// use minimum_image() to account for PBC

		double **x = atom->x;
		int nlocal = atom->nlocal;

		flag = 0;
		for (m = 0; m < natom; m++) {
		  for (i = 0; i < nlocal; i++) {
			delx = coords[m][0] - x[i][0];
			dely = coords[m][1] - x[i][1];
			delz = coords[m][2] - x[i][2];
			domain->minimum_image(delx,dely,delz);
			rsq = delx*delx + dely*dely + delz*delz;
			if (rsq < nearsq) flag = 1;
		  }
		}
		MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_MAX,world);
		if (flagall) continue;

		// proceed with insertion

		nlocalprev = atom->nlocal;

		vnew[0] = vxlo + random->uniform() * (vxhi-vxlo);
		vnew[1] = vylo + random->uniform() * (vyhi-vylo);
		vnew[2] = vzlo + random->uniform() * (vzhi-vzlo);

		// if target specified, change velocity vector accordingly

		if (targetflag) {
		  double vel = sqrt(vnew[0]*vnew[0] + vnew[1]*vnew[1] + vnew[2]*vnew[2]);
		  delx = tx - coord[0];
		  dely = ty - coord[1];
		  delz = tz - coord[2];
		  double rsq = delx*delx + dely*dely + delz*delz;
		  if (rsq > 0.0) {
			double rinv = sqrt(1.0/rsq);
			vnew[0] = delx*rinv*vel;
			vnew[1] = dely*rinv*vel;
			vnew[2] = delz*rinv*vel;
		  }
		}

		// check if new atoms are in my sub-box or above it if I am highest proc
		// if so, add atom to my list via create_atom()
		// initialize additional info about the atoms
		// set group mask to "all" plus fix group

		for (m = 0; m < natom; m++) {
		  if (domain->triclinic) {
			domain->x2lamda(coords[m],lamda);
			newcoord = lamda;
		  } else newcoord = coords[m];
		  
		  flag = 0;
		  if (newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
			  newcoord[1] >= sublo[1] && newcoord[1] < subhi[1] &&
			  newcoord[2] >= sublo[2] && newcoord[2] < subhi[2]) flag = 1;
		  else if (dimension == 3 && newcoord[2] >= domain->boxhi[2] &&
				   comm->myloc[2] == comm->procgrid[2]-1 &&
				   newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
				   newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
		  else if (dimension == 2 && newcoord[1] >= domain->boxhi[1] &&
				   comm->myloc[1] == comm->procgrid[1]-1 &&
				   newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;

		  if (flag) {
			if (mode == ATOM) {
				atom->avec->create_atom(ntype,coords[m]);
			}
			else {
				atom->avec->create_atom(ntype+onemols[imol]->type[m],coords[m]);
			}
			n = atom->nlocal - 1;
			atom->tag[n] = maxtag_all + m+1;
			if (mode == MOLECULE) {
			  if (atom->molecule_flag) atom->molecule[n] = maxmol_all+1;
			  if (atom->molecular == 2) {
				atom->molindex[n] = 0;
				atom->molatom[n] = m;
			  }
			}
			atom->mask[n] = 1 | groupbit;
			atom->image[n] = imageflags[m];
			atom->v[n][0] = vnew[0];
			atom->v[n][1] = vnew[1];
			atom->v[n][2] = vnew[2];

			atom->rho[n] = 1000;
			atom->phi[n] = 1000;
			atom->e[n] = 0;
			atom->cv[n] = 1;
			atom->viscosity_[n] = 1;
			atom->rmass[n] = atom->mass[ntype];
			atom->csound[n] = 1000;

			if (mode == MOLECULE) 
			  atom->add_molecule_atom(onemols[imol],m,n,maxtag_all);
			for (j = 0; j < nfix; j++)
			  if (fix[j]->create_attribute) fix[j]->set_arrays(n);

		  }
		}

		// FixRigidSmall::set_molecule stores rigid body attributes
		//   coord is new position of geometric center of mol, not COM
		// FixShake::set_molecule stores shake info for molecule

		if (rigidflag)
		  fixrigid->set_molecule(nlocalprev,maxtag_all,imol,coord,vnew,quat);
		else if (shakeflag)
		  fixshake->set_molecule(nlocalprev,maxtag_all,imol,coord,vnew,quat);

		// old code: unsuccessful if no proc performed insertion of an atom
		// don't think that check is necessary
		// if get this far, should always be succesful
		// would be hard to undo partial insertion for a molecule
		// better to check how many atoms could be inserted (w/out inserting)
		//   then sum to insure all are inserted, before doing actual insertion
		// MPI_Allreduce(&flag,&success,1,MPI_INT,MPI_MAX,world);

		success = 1;
		break;
	  }

	  // warn if not successful b/c too many attempts

	  if (!success && comm->me == 0)
		error->warning(FLERR,"Particle deposition was unsuccessful",0);

	  // reset global natoms,nbonds,etc
	  // increment maxtag_all and maxmol_all if necessary
	  // if global map exists, reset it now instead of waiting for comm
	  // since adding atoms messes up ghosts

	  if (success) {
		atom->natoms += natom;
		 // printf("natom = %d, anatom = %d\n", natom, atom->natoms);
		if (atom->natoms < 0 || atom->natoms > MAXBIGINT)
		  error->all(FLERR,"Too many total atoms");
		if (mode == MOLECULE) {
		  atom->nbonds += onemols[imol]->nbonds;
		  atom->nangles += onemols[imol]->nangles;
		  atom->ndihedrals += onemols[imol]->ndihedrals;
		  atom->nimpropers += onemols[imol]->nimpropers;
		}
		maxtag_all += natom;
		if (maxtag_all >= MAXTAGINT)
		  error->all(FLERR,"New atom IDs exceed maximum allowed ID");
		if (mode == MOLECULE && atom->molecule_flag) maxmol_all++;
		if (atom->map_style) {
		  atom->nghost = 0;
		  atom->map_init();
		  atom->map_set();
		}
	  }

	  // next timestep to insert
	  // next_reneighbor = 0 if done

	  if (success) ninserted++;
	  //if (ninserted < ninsert) next_reneighbor += nfreq;
	  //else next_reneighbor = 0;
  }
}

/* ----------------------------------------------------------------------
   perform particle insertion
   // not done yet
------------------------------------------------------------------------- */

void FixNumberControl::insert_molecule()
{
	insert_atom();
}

/* ----------------------------------------------------------------------
   perform particle removal 
------------------------------------------------------------------------- */

void FixNumberControl::remove_atom()
{
  int i,j,m,iwhichglobal,iwhichlocal;
  int ndel,ndeltopo[4];
  ndel = 0;

  // grow list and mark arrays if necessary

  if (atom->nlocal > nmax) {
    memory->destroy(list);
    memory->destroy(mark);
    nmax = atom->nmax;
    memory->create(list,nmax,"FixNumberControl:list");
    memory->create(mark,nmax,"FixNumberControl:mark");
  }

  // ncount = # of deletable atoms in region that I own
  // nall = # on all procs
  // nbefore = # on procs before me
  // list[ncount] = list of local indices of atoms I can delete

  Region *region = domain->regions[iregion];
  region->prematch();

  double **x = atom->x;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  int ncount = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit)
      if (region->match(x[i][0],x[i][1],x[i][2])) list[ncount++] = i;

  int nall,nbefore;
  MPI_Allreduce(&ncount,&nall,1,MPI_INT,MPI_SUM,world);
  MPI_Scan(&ncount,&nbefore,1,MPI_INT,MPI_SUM,world);
  nbefore -= ncount;

  // ndel = total # of atom deletions, in or out of region
  // ndeltopo[1,2,3,4] = ditto for bonds, angles, dihedrals, impropers
  // mark[] = 1 if deleted

  ndel = 0;
  for (i = 0; i < nlocal; i++) mark[i] = 0;

  // atomic deletions
  // choose atoms randomly across all procs and mark them for deletion
  // shrink eligible list as my atoms get marked
  // keep ndel,ncount,nall,nbefore current after each atom deletion

  if (molflag == 0) {
    while (nall && ndel < nflux) {
      iwhichglobal = static_cast<int> (nall*random->uniform());
      if (iwhichglobal < nbefore) nbefore--;
      else if (iwhichglobal < nbefore + ncount) {
        iwhichlocal = iwhichglobal - nbefore;
        mark[list[iwhichlocal]] = 1;
        list[iwhichlocal] = list[ncount-1];
        ncount--;
      }
      ndel++;
      nall--;
    }

  // molecule deletions
  // choose one atom in one molecule randomly across all procs
  // bcast mol ID and delete all atoms in that molecule on any proc
  // update deletion count by total # of atoms in molecule
  // shrink list of eligible candidates as any of my atoms get marked
  // keep ndel,ndeltopo,ncount,nall,nbefore current after each mol deletion

  } else {
    int me,proc,iatom,ndelone,ndelall,index;
    tagint imolecule;
    tagint *molecule = atom->molecule;
    int *molindex = atom->molindex;
    int *molatom = atom->molatom;
    int molecular = atom->molecular;
    Molecule **onemols = atom->avec->onemols;

    ndeltopo[0] = ndeltopo[1] = ndeltopo[2] = ndeltopo[3] = 0;

    while (nall && ndel < nflux) {

      // pick an iatom,imolecule on proc me to delete

      iwhichglobal = static_cast<int> (nall*random->uniform());
      if (iwhichglobal >= nbefore && iwhichglobal < nbefore + ncount) {
        iwhichlocal = iwhichglobal - nbefore;
        iatom = list[iwhichlocal];
        imolecule = molecule[iatom];
        me = comm->me;
      } else me = -1;

      // bcast mol ID to delete all atoms from
      // if mol ID > 0, delete any atom in molecule and decrement counters
      // if mol ID == 0, delete single iatom
      // logic with ndeltopo is to count # of deleted bonds,angles,etc
      // for atom->molecular = 1, do this for each deleted atom in molecule
      // for atom->molecular = 2, use Molecule counts for just 1st atom in mol

      MPI_Allreduce(&me,&proc,1,MPI_INT,MPI_MAX,world);
      MPI_Bcast(&imolecule,1,MPI_LMP_TAGINT,proc,world);
      ndelone = 0;
      for (i = 0; i < nlocal; i++) {
        if (imolecule && molecule[i] == imolecule) {
          mark[i] = 1;
          ndelone++;

          if (molecular == 1) {
            if (atom->avec->bonds_allow) {
              if (force->newton_bond) ndeltopo[0] += atom->num_bond[i];
              else {
                for (j = 0; j < atom->num_bond[i]; j++) {
                  if (tag[i] < atom->bond_atom[i][j]) ndeltopo[0]++;
                }
              }
            }
            if (atom->avec->angles_allow) {
              if (force->newton_bond) ndeltopo[1] += atom->num_angle[i];
              else {
                for (j = 0; j < atom->num_angle[i]; j++) {
                  m = atom->map(atom->angle_atom2[i][j]);
                  if (m >= 0 && m < nlocal) ndeltopo[1]++;
                }
              }
            }
            if (atom->avec->dihedrals_allow) {
              if (force->newton_bond) ndeltopo[2] += atom->num_dihedral[i];
              else {
                for (j = 0; j < atom->num_dihedral[i]; j++) {
                  m = atom->map(atom->dihedral_atom2[i][j]);
                  if (m >= 0 && m < nlocal) ndeltopo[2]++;
                }
              }
            }
            if (atom->avec->impropers_allow) {
              if (force->newton_bond) ndeltopo[3] += atom->num_improper[i];
              else {
                for (j = 0; j < atom->num_improper[i]; j++) {
                  m = atom->map(atom->improper_atom2[i][j]);
                  if (m >= 0 && m < nlocal) ndeltopo[3]++;
                }
              }
            }

          } else {
            if (molatom[i] == 0) {
              index = molindex[i];
              ndeltopo[0] += onemols[index]->nbonds;
              ndeltopo[1] += onemols[index]->nangles;
              ndeltopo[2] += onemols[index]->ndihedrals;
              ndeltopo[3] += onemols[index]->nimpropers;
            }
          }

        } else if (me == proc && i == iatom) {
          mark[i] = 1;
          ndelone++;
        }
      }

      // remove any atoms marked for deletion from my eligible list

      i = 0;
      while (i < ncount) {
        if (mark[list[i]]) {
          list[i] = list[ncount-1];
          ncount--;
        } else i++;
      }

      // update ndel,ncount,nall,nbefore
      // ndelall is total atoms deleted on this iteration
      // ncount is already correct, so resum to get nall and nbefore

      MPI_Allreduce(&ndelone,&ndelall,1,MPI_INT,MPI_SUM,world);
      ndel += ndelall;
      MPI_Allreduce(&ncount,&nall,1,MPI_INT,MPI_SUM,world);
      MPI_Scan(&ncount,&nbefore,1,MPI_INT,MPI_SUM,world);
      nbefore -= ncount;
    }
  }

  // delete my marked atoms
  // loop in reverse order to avoid copying marked atoms

  AtomVec *avec = atom->avec;
  double **v = atom->vest;

  double removedQ = 0;
  for (i = nlocal-1; i >= 0; i--) {
    if (mark[i]) {

      avec->copy(atom->nlocal-1,i,1);
      atom->nlocal--;
    }
  }
  // printf("nstep = %d, pid = %d, qsum = %g\n", update->ntimestep, comm->me, removedQ);

  // reset global natoms and bonds, angles, etc
  // if global map exists, reset it now instead of waiting for comm
  // since deleting atoms messes up ghosts

  atom->natoms -= ndel;
  if (molflag) {
    int all[4];
    MPI_Allreduce(ndeltopo,all,4,MPI_INT,MPI_SUM,world);
    atom->nbonds -= all[0];
    atom->nangles -= all[1];
    atom->ndihedrals -= all[2];
    atom->nimpropers -= all[3];
  }

  if (ndel && atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }

  // statistics

  ndeleted += ndel;
//  next_reneighbor = update->ntimestep + nevery;
}

/* ----------------------------------------------------------------------
   perform particle removal 
------------------------------------------------------------------------- */

void FixNumberControl::remove_molecule()
{
	remove_atom();
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
   maxmol_all = current max molecule ID for all atoms
------------------------------------------------------------------------- */

void FixNumberControl::find_maxid()
{
  tagint *tag = atom->tag;
  tagint *molecule = atom->molecule;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);

  if (mode == MOLECULE && molecule) {
    max = 0;
    for (int i = 0; i < nlocal; i++) max = MAX(max,molecule[i]);
    MPI_Allreduce(&max,&maxmol_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
  }
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixNumberControl::options(int narg, char **arg)
{
  // defaults

  iregion = -1;
  idregion = NULL;
  ifix = -1;
  idfix = NULL;
  mode = ATOM;
  molfrac = NULL;
  rigidflag = 0;
  idrigid = NULL;
  shakeflag = 0;
  idshake = NULL;
  idnext = 0;
  globalflag = localflag = 0;
  lo = hi = deltasq = 0.0;
  nearsq = 0.0;
  maxattempt = 10;
  rateflag = 0;
  vxlo = vxhi = vylo = vyhi = vzlo = vzhi = 0.0;
  scaleflag = 1;
  targetflag = 0;
  molflag = 0;

  // default
  nparticles = ninsert = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"region") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      iregion = domain->find_region(arg[iarg+1]);
      if (iregion == -1)
        error->all(FLERR,"Region ID for fix number/control does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idregion = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
	} else if (strcmp(arg[iarg],"fix") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      ifix = domain->find_region(arg[iarg+1]);
      if (ifix == -1)
        error->all(FLERR,"Fix ID for fix number/control does not exist");
      int n = strlen(arg[iarg+1]) + 1;
      idfix = new char[n];
      strcpy(idregion,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"mol") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      int imol = atom->find_molecule(arg[iarg+1]);
      if (imol == -1)
        error->all(FLERR,"Molecule template ID for fix number/control does not exist");
	  molflag = 1; // does have molecular tag; necessary for the molecular deletion
      mode = MOLECULE;
      onemols = &atom->molecules[imol];
      nmol = onemols[0]->nset;
      delete [] molfrac;
      molfrac = new double[nmol];
      molfrac[0] = 1.0/nmol;
      for (int i = 1; i < nmol-1; i++) molfrac[i] = molfrac[i-1] + 1.0/nmol;
      molfrac[nmol-1] = 1.0;
      iarg += 2;
    } else if (strcmp(arg[iarg],"molfrac") == 0) {
      if (mode != MOLECULE) error->all(FLERR,"Illegal fix number/control command");
      if (iarg+nmol+1 > narg) error->all(FLERR,"Illegal fix number/control command");
      molfrac[0] = force->numeric(FLERR,arg[iarg+1]);
      for (int i = 1; i < nmol; i++) 
        molfrac[i] = molfrac[i-1] + force->numeric(FLERR,arg[iarg+i+1]);
      if (molfrac[nmol-1] < 1.0-EPSILON || molfrac[nmol-1] > 1.0+EPSILON) 
        error->all(FLERR,"Illegal fix number/control command");
      molfrac[nmol-1] = 1.0;
      iarg += nmol+1;

    } else if (strcmp(arg[iarg],"rigid") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      int n = strlen(arg[iarg+1]) + 1;
      delete [] idrigid;
      idrigid = new char[n];
      strcpy(idrigid,arg[iarg+1]);
      rigidflag = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg],"shake") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      int n = strlen(arg[iarg+1]) + 1;
      delete [] idshake;
      idshake = new char[n];
      strcpy(idshake,arg[iarg+1]);
      shakeflag = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg],"id") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      if (strcmp(arg[iarg+1],"max") == 0) idnext = 0;
      else if (strcmp(arg[iarg+1],"next") == 0) idnext = 1;
      else error->all(FLERR,"Illegal fix number/control command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"global") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix number/control command");
      globalflag = 1;
      localflag = 0;
      lo = force->numeric(FLERR,arg[iarg+1]);
      hi = force->numeric(FLERR,arg[iarg+2]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"local") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix number/control command");
      localflag = 1;
      globalflag = 0;
      lo = force->numeric(FLERR,arg[iarg+1]);
      hi = force->numeric(FLERR,arg[iarg+2]);
      deltasq = force->numeric(FLERR,arg[iarg+3]) * 
        force->numeric(FLERR,arg[iarg+3]);
      iarg += 4;

    } else if (strcmp(arg[iarg],"near") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      nearsq = force->numeric(FLERR,arg[iarg+1]) * 
        force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"attempt") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      maxattempt = force->inumeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"rate") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      rateflag = 1;
      rate = force->numeric(FLERR,arg[iarg+1]);
      iarg += 2;
    } else if (strcmp(arg[iarg],"vx") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix number/control command");
      vxlo = force->numeric(FLERR,arg[iarg+1]);
      vxhi = force->numeric(FLERR,arg[iarg+2]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"vy") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix number/control command");
      vylo = force->numeric(FLERR,arg[iarg+1]);
      vyhi = force->numeric(FLERR,arg[iarg+2]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"vz") == 0) {
      if (iarg+3 > narg) error->all(FLERR,"Illegal fix number/control command");
      vzlo = force->numeric(FLERR,arg[iarg+1]);
      vzhi = force->numeric(FLERR,arg[iarg+2]);
      iarg += 3;
    } else if (strcmp(arg[iarg],"units") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal fix number/control command");
      if (strcmp(arg[iarg+1],"box") == 0) scaleflag = 0;
      else if (strcmp(arg[iarg+1],"lattice") == 0) scaleflag = 1;
      else error->all(FLERR,"Illegal fix number/control command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"target") == 0) {
      if (iarg+4 > narg) error->all(FLERR,"Illegal fix number/control command");
      tx = force->numeric(FLERR,arg[iarg+1]);
      ty = force->numeric(FLERR,arg[iarg+2]);
      tz = force->numeric(FLERR,arg[iarg+3]);
      targetflag = 1;
      iarg += 4;
    } else error->all(FLERR,"Illegal fix number/control command");
  }
}


/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixNumberControl::write_restart(FILE *fp)
{
  int n = 0;
  //double list[4];
  double list[3];
  list[n++] = random->state();
  // list[n++] = ninserted;
  list[n++] = nfirst;
  list[n++] = next_reneighbor;

  if (comm->me == 0) {
    int size = n * sizeof(double);
    fwrite(&size,sizeof(int),1,fp);
    fwrite(list,sizeof(double),n,fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixNumberControl::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  seed = static_cast<int> (list[n++]);
  // ninserted = static_cast<int> (list[n++]);
  nfirst = static_cast<int> (list[n++]);
  next_reneighbor = static_cast<int> (list[n++]);

  random->reset(seed);
}

/* ----------------------------------------------------------------------
   extract particle radius for atom type = itype
------------------------------------------------------------------------- */

void *FixNumberControl::extract(const char *str, int &itype)
{
  if (strcmp(str,"radius") == 0) {
    if (mode == ATOM) {
      if (itype == ntype) oneradius = 0.5;
      else oneradius = 0.0;

    } else {

      // find a molecule in template with matching type

      for (int i = 0; i < nmol; i++) {
        if (itype-ntype > onemols[i]->ntypes) continue;
        double *radius = onemols[i]->radius;
        int *type = onemols[i]->type;
        int natoms = onemols[i]->natoms;

        // check radii of matching types in Molecule
        // default to 0.5, if radii not defined in Molecule
        //   same as atom->avec->create_atom(), invoked in pre_exchange()

        oneradius = 0.0;
        for (int i = 0; i < natoms; i++)
          if (type[i] == itype-ntype) {
            if (radius) oneradius = MAX(oneradius,radius[i]);
            else oneradius = 0.5;
          }
      }
    }
    itype = 0;
    return &oneradius;
  }

  return NULL;
}

