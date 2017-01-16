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
#include "fix_3d_printing.h"
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
#include "math_extra.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

#define EPSILON 1.0e6
enum{LAYOUT_UNIFORM,LAYOUT_NONUNIFORM,LAYOUT_TILED};    // several files

/* ---------------------------------------------------------------------- */

Fix3DPrinting::Fix3DPrinting(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg != 7) error->all(FLERR,"Illegal fix 3D/printing command");

  restart_global = 1;
  time_depend = 1;
  idnext = 0;

  // program will first load file (filename)
  // the file follows a format as:
  // sphElement 5
  // type x y z
  // type x y z
  // type x y z
  // type x y z
  // type x y z
  // sphElement 2
  // type x y z
  // type x y z
  // ....

  // required args

  fileName = arg[3]; // filename
  fx = force->numeric(FLERR, arg[4]); // fx
  fy = force->numeric(FLERR, arg[5]); // fy
  fz = force->numeric(FLERR, arg[6]); // fz

  // error check on type
  // parse the file and do error check
  int maxType, minType;
  loadSPHElements(minType, maxType, fileName);

  //std::cout << comm->me << " Total Number of sphElement :" << sphElements.size() << std::endl;

  if (minType <= 0 || maxType > atom->ntypes)
    error->all(FLERR,"Invalid atom type in fix 3D/printing command");


  //it = sphElements.end();      
  //--it;
  //std::cout << comm->me << " The last sphElement has following : " << std::endl;
  //std::cout << comm->me << " " << (*it).size() << std::endl;

  //std::vector<ele>::const_iterator tmpvec;
  //for(tmpvec = (*it).begin(); tmpvec != (*it).end(); ++tmpvec) {
      //std::cout << comm->me << " " << (*tmpvec).pType << " " << (*tmpvec).px << " " << (*tmpvec).py << " " << (*tmpvec).pz << std::endl;
  //}

  it = sphElements.begin();      

  if(sphElements.size() < 1) 
      error->all(FLERR, " 3D Printing sphElement cannot be zero!");

  // setup of coords and imageflags array

  natom_max = 1;
  memory->create(coords,natom_max,3,"3D/printing:coords");
  memory->create(imageflags,natom_max,"3D/printing:imageflags");

  // find current max atom and molecule IDs if necessary

  if (idnext) find_maxid();

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;
  nfirst = next_reneighbor;
  ninserted = 0;

  //std::cout << comm->me << " ----------------- " << std::endl;
}

/* ---------------------------------------------------------------------- */

void Fix3DPrinting::loadSPHElements(int &minType, int &maxType, char*fileName) {

    //std::cout << comm->me <<  " Loading SPHElements ........... " << std::endl;
    minType = 10000;
    maxType = 0;
    int cSphEle = 0;
    std::string line;
    std::ifstream file(fileName, std::ifstream::in);
//    std::stringstream buffer;
    if(file.is_open()) {
//        buffer << file.rdbuf();

        int nParticles = 0;
        int c = 0;
        int pType = 0;

        double px, py, pz;
        std::vector<ele> vec;

        // not the end 
        // do the loop
        while (std::getline(file,line)) {
            //std::cout << line << std::endl;
            std::istringstream iss(line);

            // skip the empty line
            if(line.empty()) continue;

            // find keyword: sphElement 
            if(line.find("sphElement") == 0) {
                //std::string str1;
                //std::string str2;
                //iss >> str1;
                //iss >> str2;
                std::string val;
                iss >> val;
                iss >> nParticles;
                //std::cout << str1 << std::endl;
                //std::cout << str2 << std::endl;
                cSphEle++;
                continue;
            } else {
                iss >> pType;
                iss >> px;
                iss >> py;
                iss >> pz;
                //std::cout << pType << " " << px << " " << py << " " << pz << std::endl;

                ele e;
                e.pType = pType;
                e.px = px;
                e.py = py;
                e.pz = pz;

                minType = std::min(minType, pType);
                maxType = std::max(maxType, pType);

                vec.push_back(e);
                c++;
            }

            // clear vec
            if(c%nParticles == 0) {
                sphElements.push_back(vec);
                vec.clear();
                c = 0;
            }
        }
    }

    //std::cout << comm->me << " Ending Loading Element .... " << cSphEle << std::endl;
}

/* ---------------------------------------------------------------------- */

Fix3DPrinting::~Fix3DPrinting()
{
  memory->destroy(coords);
  memory->destroy(imageflags);
}

/* ---------------------------------------------------------------------- */

int Fix3DPrinting::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  mask |= POST_FORCE; // post the force to the added particles
  return mask;
}

/* ---------------------------------------------------------------------- */

void Fix3DPrinting::init()
{
  nfreq = update->nsteps/sphElements.size();
  //std::cout << comm->me << " nfreq : " << nfreq << std::endl;
}

/* ----------------------------------------------------------------------
   perform particle insertion
------------------------------------------------------------------------- */

void Fix3DPrinting::pre_exchange()
{
  int i,j,m,n,natom,flag;
  double coord[3];
  double vnew[3];
  double *newcoord;

  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  insertedIds.clear();

  double *sublo,*subhi;
  if (domain->triclinic == 0) {
    sublo = domain->sublo;
    subhi = domain->subhi;
  } 
  // find current max atom and molecule IDs if necessary

  if (!idnext) find_maxid();

  int dimension = domain->dimension;
  int nfix = modify->nfix;
  Fix **fix = modify->fix;

  std::vector<ele>::const_iterator eleIt;
  natom = (*it).size();
  if(natom < 1) return;
  m = 0;
  //std::cout << update->ntimestep << " Num of Particles: " << natom << std::endl;
  for(eleIt = (*it).begin(); eleIt != (*it).end(); ++eleIt) {

    coord[0] = (*eleIt).px;
    coord[1] = (*eleIt).py;
    coord[2] = (*eleIt).pz;
    ntype = (*eleIt).pType;
    //std::cout << (*eleIt).pType <<  " " << (*eleIt).px << " " << (*eleIt).py << " " << (*eleIt).pz << std::endl;

    coords[0][0] = coord[0];
    coords[0][1] = coord[1];
    coords[0][2] = coord[2];

    imageflags[0] = ((imageint) IMGMAX << IMG2BITS) |
        ((imageint) IMGMAX << IMGBITS) | IMGMAX;

    vnew[0] = 0.0;
    vnew[1] = 0.0;
    vnew[2] = 0.0;

    newcoord = coords[0];
    flag = 0;
    if (newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
        newcoord[1] >= sublo[1] && newcoord[1] < subhi[1] &&
        newcoord[2] >= sublo[2] && newcoord[2] < subhi[2]) flag = 1;
    else if (dimension == 3 && newcoord[2] >= domain->boxhi[2]) {
      if (comm->layout != LAYOUT_TILED) {
        if (comm->myloc[2] == comm->procgrid[2]-1 &&
            newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
            newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
      } else {
        if (comm->mysplit[2][1] == 1.0 &&
            newcoord[0] >= sublo[0] && newcoord[0] < subhi[0] &&
            newcoord[1] >= sublo[1] && newcoord[1] < subhi[1]) flag = 1;
      } 
    } else if (dimension == 2 && newcoord[1] >= domain->boxhi[1]) {
      if (comm->layout != LAYOUT_TILED) {
        if (comm->myloc[1] == comm->procgrid[1]-1 &&
            newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;
      } else {
        if (comm->mysplit[1][1] == 1.0 &&
            newcoord[0] >= sublo[0] && newcoord[0] < subhi[0]) flag = 1;
      }
    }

      //std::cout << "m :" << m << " flag : " << flag << " " << ntype << std::endl;

    if (flag) {
      atom->avec->create_atom(ntype,coords[0]);
      n = atom->nlocal - 1;
      atom->tag[n] = maxtag_all + m + 1;
      int globalId = atom->tag[n];
      insertedIds.push_back(globalId);
      atom->mask[n] = 1 | groupbit;
      atom->image[n] = imageflags[0];
      atom->v[n][0] = vnew[0];
      atom->v[n][1] = vnew[1];
      atom->v[n][2] = vnew[2];
      //std::cout << n << " " << atom->rho[n]  << std::endl;
      for (j = 0; j < nfix; j++)
        if (fix[j]->create_attribute) fix[j]->set_arrays(n);
    }
    m++;
  }

  atom->natoms += natom;
  if (atom->natoms < 0 || atom->natoms > MAXBIGINT)
    error->all(FLERR,"Too many total atoms");
  maxtag_all += natom;
  if (maxtag_all >= MAXTAGINT)
    error->all(FLERR,"New atom IDs exceed maximum allowed ID");
  if (atom->map_style) {
    atom->nghost = 0;
    atom->map_init();
    atom->map_set();
  }

  // next timestep to insert
  // next_reneighbor = 0 if done
  next_reneighbor += nfreq;

  // print all coordinates
  //for (int i = 0; i < atom->nlocal; i++) {

      //std::cout << i+1 << " " << atom->x[i][0] << " "\
          //<< atom->x[i][1] << " " \
          //<< atom->x[i][2] << std::endl;
  //}

}

/* ----------------------------------------------------------------------
    add force to the newly added particles
------------------------------------------------------------------------- */

void Fix3DPrinting::post_force(int vflag)
{
  double **f = atom->f;

  if (update->ntimestep % nfreq != 0) return;

  // constant force
  // potential energy = - x dot f in unwrapped coords

  if ((*it).size() < 1) return;

  std::vector<int>::const_iterator idIt;
  //std::cout << update->ntimestep << std::endl;
  for (idIt = insertedIds.begin(); idIt != insertedIds.end(); ++idIt) {
      // use a list to store particle global id
      // global id will be mapped to local id
        int localId = atom->map((*idIt));
        //std::cout << (*idIt) << ' ' << localId << std::endl;
        f[localId][0] += fx;
        f[localId][1] += fy;
        f[localId][2] += fz;
  }

  // system energy would be consider later
  
  // iterator to the next sphelement
  ++it;
}

/* ----------------------------------------------------------------------
   maxtag_all = current max atom ID for all atoms
   maxmol_all = current max molecule ID for all atoms
------------------------------------------------------------------------- */

void Fix3DPrinting::find_maxid()
{
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;

  tagint max = 0;
  for (int i = 0; i < nlocal; i++) max = MAX(max,tag[i]);
  MPI_Allreduce(&max,&maxtag_all,1,MPI_LMP_TAGINT,MPI_MAX,world);
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void Fix3DPrinting::write_restart(FILE *fp)
{
  int n = 0;
  double list[3];
  list[n++] = ninserted;
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

void Fix3DPrinting::restart(char *buf)
{
  int n = 0;
  double *list = (double *) buf;

  ninserted = static_cast<int> (list[n++]);
  nfirst = static_cast<int> (list[n++]);
  next_reneighbor = static_cast<int> (list[n++]);
}

