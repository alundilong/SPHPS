
	double tauxx = tau_[i][0][0];
	double tauxy = tau_[i][0][1];
	double tauyx = tau_[i][1][0];
	double tauyy = tau_[i][1][1];
	
	double Jsqr = \
			   tauxx*tauxx + tauxy*tauxy \
			 + tauyx*tauyx + tauyy*tauyy;

	Jsqr *= 0.5;
	
	if(Jsqr > Y0_*Y0_)
	{
		//double scale = sqrt(Y0_/3./Jsqr);
		double scale = sqrt(Y0_*Y0_/3./Jsqr);
		tau_[i][0][0] *= scale;
		tau_[i][0][1] *= scale;
		tau_[i][1][1] *= scale;
		tau_[i][1][0] *= scale;
	}

	tauxx = tau_[i][0][0];
	tauxy = tau_[i][0][1];
	tauyx = tau_[i][1][0];
	tauyy = tau_[i][1][1];

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
        h = h_;
        ih = 1. / h;

		double r = sqrt(rsq);
		double rInv = 1./r;
		double s = r/h;

		wfd = SPH_KERNEL_NS::sph_kernel_cubicspline2d_der(s, h, ih)*rInv;

		// - compute strain rate
		double vjix = v[j][0] - v[i][0];
		double vjiy = v[j][1] - v[i][1];
		double vjiz = v[j][2] - v[i][2];

		epsilon_[i][0][0] += jmass/rhoj*vjix*wfd*delx;

		epsilon_[i][1][1] += jmass/rhoj*vjiy*wfd*dely;
		epsilon_[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely + vjiy*delx);

		R_[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely - vjiy*delx);
    }

  }

	epsilon_[i][1][0] = epsilon_[i][0][1];

	R_[i][1][0] = -1.0*R_[i][0][1];

	double trace = 1./2.*(epsilon_[i][0][0] + epsilon_[i][1][1]);

	epsilonBar_[i][0][0] = epsilon_[i][0][0] - trace;
	epsilonBar_[i][0][1] = epsilon_[i][0][1];
	epsilonBar_[i][1][0] = epsilon_[i][0][1];
	epsilonBar_[i][1][1] = epsilon_[i][1][1] - trace;

	double & dTauxx = dTau_[i][0][0];
	double & dTauxy = dTau_[i][0][1];
	double & dTauyx = dTau_[i][1][0];
	double & dTauyy = dTau_[i][1][1];

	double Rxx = R_[i][0][0];
	double Rxy = R_[i][0][1];
	double Ryx = R_[i][1][0];
	double Ryy = R_[i][1][1];

	double epsBarxx = epsilonBar_[i][0][0];
	double epsBarxy = epsilonBar_[i][0][1];
	double epsBaryx = epsilonBar_[i][1][0];
	double epsBaryy = epsilonBar_[i][1][1];

	double epsxx = epsilon_[i][0][0];
	double epsxy = epsilon_[i][0][1];
	double epsyx = epsilon_[i][1][0];
	double epsyy = epsilon_[i][1][1];

	//- Hooke Deviatoric Stress
	
	dTauxx = 2*G_*epsBarxx + ( tauxx*Rxx + tauyx*Rxy ) + ( tauxx*Rxx + tauxy*Rxy );
	dTauxy = 2*G_*epsBarxy + ( tauxy*Rxx + tauyy*Rxy ) + ( tauxx*Ryx + tauxy*Ryy );
	dTauyx = 2*G_*epsBaryx + ( tauxx*Ryx + tauyx*Ryy ) + ( tauyx*Rxx + tauyy*Rxy );
	dTauyy = 2*G_*epsBaryy + ( tauxy*Ryx + tauyy*Ryy ) + ( tauyx*Ryx + tauyy*Ryy );

	/*
	 * EOS 
	 * Mie-Gruneisen Equation
	 * Liu's Book, Page 297
	 */ 
	double eta = rho[i]/rho0_ - 1.0;
	a0_ = rho0_*Cs_*Cs_;
	b0_ = a0_*(1. + 2.*(S_ - 1.));
	c0_ = a0_*(2*(S_ - 1.) + 3*(S_ - 1.)*(S_ - 1.));
	double pH = eta > 0 ? (a0_*eta + b0_*eta*eta + c0_*eta*eta*eta) : a0_*eta;
//	p_[i] =  (1.0 - 0.5*Gamma_*eta)*pH + Gamma_*rho[i]*e[i]; 

	p_[i] = Cs_*Cs_*(rho[i] - rho0_);

	sigma_[i][0][0] = -p_[i] + tauxx; sigma_[i][0][1] = tauxy;
	sigma_[i][1][0] = tauyx; sigma_[i][1][1] = -p_[i] + tauyy;


	// - Artificial Stress

	  double S[2][2];
	  S[0][0] = sigma_[i][0][0];
	  S[0][1] = sigma_[i][0][1];

	  S[1][0] = sigma_[i][1][0];
	  S[1][1] = sigma_[i][1][1];

	  //- to avoid too small value
	  //- that will create difficulty
	  //- in decomposition_gsl
	  for(int ll = 0; ll < 2; ll++)
		{
	    	for(int mm = 0; mm < 2; mm++)
				{
					if(fabs(S[ll][mm]) < TOLERANCE) S[ll][mm] = 0; 
				}
		}

	  double R[2][2], Rab[2][2];
	  double V[2], rd[2];
	  R[0][0] = R[0][1] = 0.0;
	  R[1][0] = R[1][1] = 0.0;

	  Rab[0][0] = Rab[0][1] = 0.0;
	  Rab[1][0] = Rab[1][1] = 0.0;
	  V[0] = V[1] = rd[0] = rd[1] = 0.0;

	  MathExtraPysph::eigen_decomposition_gsl_2D(S, R, V);

	  double rhoi2Inv = 1./rho[i]/rho[i];
	  if(V[0] > 0) rd[0] = -eps_*V[0]*rhoi2Inv;
	  else rd[0] = 0;

	  if(V[1] > 0) rd[1] = -eps_*V[1]*rhoi2Inv;
	  else rd[1] = 0;

	  MathExtraPysph::transform_diag_inv_2D(rd, R, Rab);
	  
	  artStress_[i][0][0] = Rab[0][0];
	  artStress_[i][0][1] = Rab[0][1];

	  artStress_[i][1][0] = Rab[1][0];
	  artStress_[i][1][1] = Rab[1][1];

