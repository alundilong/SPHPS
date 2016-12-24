# SPHPS
Smoothed Particle Hydrodynamics Parallel Simulator


Smoothed Particle Hydrodynamics Parallel Simulator (SPHPS) is 
a High-performance computer software package developed 
by Multiscale Thermal Transport Laboratory (MTTL) 
at University of Missouri-Columbia. It is inspired 
and designed based on the structure of the open source 
software LAMMPS (http://lammps.sandia.gov/index.html) developed 
at Sandia National Laboratory. 
Smoothed Particle Hydrodynamics is a promising pure-Lagrangian 
Numerical Method that use particles to reproduce the computational
domain of interest and discretize each term of goverening equations 
using Kernel function. This approach enables us to simulate mass 
destruction, large deformation, crack development and so forth 
where mesh-based method cannot handle appropriately due to large
mesh distortion. More importantly, this provides a more natural 
numerical mean to solve fluid structure interation problem where 
the fluid-solid interaction can be easily realized with simple 
pairwise interation. It is our motivation to provide a general 
platform for scientists interested in cracking down challenging 
computer modeling problems through contributing new models to 
the existing code. With the supervision and support from Prof. 
Yuwen Zhang, Yijin Mao as the lead designer and developer, 
MTTL group aims to conduct all sophosticated computer modeling 
within this framework. For example, fluid-structure interaction. 
Distinguished from the other FSI simulation tool, SPHPS has full 
capability to simulate the interaction between fluid phase and 
solid phase accurately, meanwhile, the stress wave propogation 
within the solid phase can also be preciously captured. It is our 
intention to create a parallel open source code that is easy and 
convenient for researchers to modify and extend to full meet 
their own research needs.

SPHPS is currently written in C++. The parallelism is fulfilled 
by using the domain decomposition technique and managed by the 
message passing interface (MPI) library. It can be run on single 
processor or multiple processors machine which can compile C++ 
and support MPI library. It is a free open source software 
package under the GNU license.

In this document, the instruction of each command is provided 
in terms of category and alphabetic order. Currently, we are 
still adding more details into this manual. For any unclear 
command instruction, user can follow LAMMPS website for more 
details. If you have any question, please feel free to contact 
the lead developer Yijin Mao for further information. 
