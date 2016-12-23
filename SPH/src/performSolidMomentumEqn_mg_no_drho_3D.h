
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      jmass = rmass[j];
	  rhoj = rho[j];

	  if (rsq < cutsq[itype][jtype]) {
		//h = h_;  //cut[itype][jtype];
		h = 0.5*cut[itype][jtype];
        ih = 1. / h;

		double r = sqrt(rsq);
		double rInv = 1./r;
		double s = r/h;

		wdeltap = SPH_KERNEL_NS::sph_kernel_cubicspline(deltap_/h, h, ih);
		wf = SPH_KERNEL_NS::sph_kernel_cubicspline(s, h, ih);
		wfd = SPH_KERNEL_NS::sph_kernel_cubicspline_der(s, h, ih)*rInv;

        // dot product of velocity delta and distance vector
        delVdotDelR = delx * (vxtmp - v[j][0]) + dely * (vytmp - v[j][1])
            + delz * (vztmp - v[j][2]);
		double delvx = vxtmp - v[j][0];
		double delvy = vytmp - v[j][1];
		double delvz = vztmp - v[j][2];


        if (delVdotDelR < 0.) {

          cj = c_;
          mu = h * delVdotDelR / (rsq + 0.01 * h * h);
          fvisc = (-viscosity[itype][jtype] * ((ci + cj)/2) * mu + beta_[itype][jtype]*mu*mu) / ((rhoi + rhoj)/2);
        } else {
          fvisc = 0.;
        }

		/*
		 * Monaghan (1983 paper, shock simulation by the particle method SPH)
		 */

		fix = imass*jmass*wfd*((sigma_[i][0][0]/rhoi/rhoi + sigma_[j][0][0]/rhoj/rhoj)*delx \
							  + (sigma_[i][0][1]/rhoi/rhoi + sigma_[j][0][1]/rhoj/rhoj)*dely \
							  + (sigma_[i][0][2]/rhoi/rhoi + sigma_[j][0][2]/rhoj/rhoj)*delz);
		fiy = imass*jmass*wfd*((sigma_[i][1][0]/rhoi/rhoi + sigma_[j][1][0]/rhoj/rhoj)*delx \
							  + (sigma_[i][1][1]/rhoi/rhoi + sigma_[j][1][1]/rhoj/rhoj)*dely \
							  + (sigma_[i][1][2]/rhoi/rhoi + sigma_[j][1][2]/rhoj/rhoj)*delz);
		fiz = imass*jmass*wfd*((sigma_[i][2][0]/rhoi/rhoi + sigma_[j][2][0]/rhoj/rhoj)*delx \
							  + (sigma_[i][2][1]/rhoi/rhoi + sigma_[j][2][1]/rhoj/rhoj)*dely \
							  + (sigma_[i][2][2]/rhoi/rhoi + sigma_[j][2][2]/rhoj/rhoj)*delz);

		/*
		 * XSPH correction 
		 * Gray, J. P., Monaghan, JJ. Swift, R. P. SPH Elastic Dynamics, Compute Methods
		 * in Applied Mechanics and Engineering.
		 */

		vXSPH[i][0] += epsXSPH_*jmass*wf*(v[j][0] - v[i][0])/(0.5*(rhoi+rhoj));
		vXSPH[i][1] += epsXSPH_*jmass*wf*(v[j][1] - v[i][1])/(0.5*(rhoi+rhoj));
		vXSPH[i][2] += epsXSPH_*jmass*wf*(v[j][2] - v[i][2])/(0.5*(rhoi+rhoj));

		/*
		 * Artifical stress to avoid tensile instability
		 * Gray, J. P., Monaghan, JJ. Swift, R. P. SPH Elastic Dynamics, Compute Methods
		 * in Applied Mechanics and Engineering.
		 */
		
		fab = wf/wdeltap;
		fab = pow(fab, 4);
		if(wdeltap < 0) fab = 0;

		fix += imass*jmass*wfd*fab*((artStress_[i][0][0] + artStress_[j][0][0])*delx \
							  + (artStress_[i][0][1] + artStress_[j][0][1])*dely \
							  + (artStress_[i][0][2] + artStress_[j][0][2])*delz);
		fiy += imass*jmass*wfd*fab*((artStress_[i][1][0] + artStress_[j][1][0])*delx \
							  + (artStress_[i][1][1] + artStress_[j][1][1])*dely \
							  + (artStress_[i][1][2] + artStress_[j][1][2])*delz);
		fiz += imass*jmass*wfd*fab*((artStress_[i][2][0] + artStress_[j][2][0])*delx \
							  + (artStress_[i][2][1] + artStress_[j][2][1])*dely \
							  + (artStress_[i][2][2] + artStress_[j][2][2])*delz);

		/* look at the Idealgas code
		 *
         * fvisc = -viscosity[itype][jtype] * (ci + cj) * mu / (rho[i] + rho[j]);
		 * f = -imass*jmass*(fi + fj + fvisc) * wfd
		 * ||
		 * \/
		 * f = imass*jmass(sigmai+sigmaj - fvisc) * wfd
		 */

		double PIJ = fvisc; // depends ....
		fix -= imass*jmass*wfd*PIJ*delx;
		fiy -= imass*jmass*wfd*PIJ*dely;
		fiz -= imass*jmass*wfd*PIJ*delz;

		double eshear = 0.5*imass*jmass*(p_[i]/rhoi/rhoi + p_[j]/rhoj/rhoj + PIJ)*wfd*(delVdotDelR);

			  eshear += -0.5*imass*jmass*wfd*fab*(tau_[i][0][0]*delvx*delx + tau_[i][0][1]*delvx*dely + tau_[i][0][2]*delvx*delz \
							  + tau_[i][1][0]*delvy*delx + tau_[i][1][1]*delvy*dely + tau_[i][1][2]*delvy*delz \
							  + tau_[i][2][0]*delvz*delx + tau_[i][2][1]*delvz*dely + tau_[i][2][2]*delvz*delz)/rhoi/rhoi;

		 	  eshear += -0.5*imass*jmass*fab*wfd*(tau_[j][0][0]*delvx*delx + tau_[j][0][1]*delvx*dely + tau_[j][0][2]*delvx*delz \
							  + tau_[j][1][0]*delvy*delx + tau_[j][1][1]*delvy*dely + tau_[j][1][2]*delvy*delz \
							  + tau_[j][2][0]*delvz*delx + tau_[j][2][1]*delvz*dely + tau_[j][2][2]*delvz*delz)/rhoj/rhoj;

			  eshear += -0.5*imass*jmass*wfd*fab*(artStress_[i][0][0]*delvx*delx + artStress_[i][0][1]*delvx*dely + artStress_[i][0][2]*delvx*delz \
							  + artStress_[i][1][0]*delvy*delx + artStress_[i][1][1]*delvy*dely + artStress_[i][1][2]*delvy*delz \
							  + artStress_[i][2][0]*delvz*delx + artStress_[i][2][1]*delvz*dely + artStress_[i][2][2]*delvz*delz);

		 	  eshear += -0.5*imass*jmass*fab*wfd*(artStress_[j][0][0]*delvx*delx + artStress_[j][0][1]*delvx*dely + artStress_[j][0][2]*delvx*delz \
							  + artStress_[j][1][0]*delvy*delx + artStress_[j][1][1]*delvy*dely + artStress_[j][1][2]*delvy*delz \
							  + artStress_[j][2][0]*delvz*delx + artStress_[j][2][1]*delvz*dely + artStress_[j][2][2]*delvz*delz);
		deltaE = eshear;

		f[i][0] += fix; 
		f[i][1] += fiy; 
		f[i][2] += fiz; 

        // and change in density
        //drho[i] += jmass * delVdotDelR * wfd;

        // change in thermal energy
        de[i] += deltaE;
/*
        if (newton_pair || j < nlocal) {
          f[j][0] -= fix;
          f[j][1] -= fiy;
          f[j][2] -= fiz;
          de[j] += deltaE;
          //drho[j] += imass * delVdotDelR * wfd;
        }

         if (evflag)
           ev_tally(i, j, nlocal, newton_pair, 0.0, 0.0, fpair, delx, dely, delz);
*/
      }
	  
    }

