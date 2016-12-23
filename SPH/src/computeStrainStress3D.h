
	double tauxx = tau_[i][0][0];
	double tauxy = tau_[i][0][1];
	double tauxz = tau_[i][0][2];
	double tauyx = tau_[i][1][0];
	double tauyy = tau_[i][1][1];
	double tauyz = tau_[i][1][2];
	double tauzx = tau_[i][2][0];
	double tauzy = tau_[i][2][1];
	double tauzz = tau_[i][2][2];

	double Jsqr = \
			   tauxx*tauxx + tauxy*tauxy+ tauxz*tauxz \
			 + tauyx*tauyx + tauyy*tauyy+ tauyz*tauyz \
			 + tauzx*tauzx + tauzy*tauzy+ tauzz*tauzz;

	Jsqr *= 0.5;

	if(Jsqr > Y0_*Y0_)
	{
		//double scale = sqrt(Y0_/3./Jsqr);
		double scale = sqrt(Y0_*Y0_/3./Jsqr);
		tau_[i][0][0] *= scale;
		tau_[i][0][1] *= scale;
		tau_[i][0][2] *= scale;
		tau_[i][1][0] *= scale;
		tau_[i][1][1] *= scale;
		tau_[i][1][2] *= scale;
		tau_[i][2][0] *= scale;
		tau_[i][2][1] *= scale;
		tau_[i][2][2] *= scale;
	}

	tauxx = tau_[i][0][0];
	tauxy = tau_[i][0][1];
	tauxz = tau_[i][0][2];
	tauyx = tau_[i][1][0];
	tauyy = tau_[i][1][1];
	tauyz = tau_[i][1][2];
	tauzx = tau_[i][2][0];
	tauzy = tau_[i][2][1];
	tauzz = tau_[i][2][2];

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

		wfd = SPH_KERNEL_NS::sph_kernel_cubicspline_der(s, h, ih)*rInv;

		// - compute strains

		double vjix = v[j][0] - v[i][0];
		double vjiy = v[j][1] - v[i][1];
		double vjiz = v[j][2] - v[i][2];
		
		epsilon_[i][0][0] += jmass/rhoj*vjix*wfd*delx;
		epsilon_[i][1][1] += jmass/rhoj*vjiy*wfd*dely;
		epsilon_[i][2][2] += jmass/rhoj*vjiz*wfd*delz;
		epsilon_[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely + vjiy*delx);
		epsilon_[i][0][2] += 0.5*jmass/rhoj*wfd*(vjix*delz + vjiz*delx);
		epsilon_[i][1][2] += 0.5*jmass/rhoj*wfd*(vjiy*delz + vjiz*dely);

		R_[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely - vjiy*delx);
		R_[i][0][2] += 0.5*jmass/rhoj*wfd*(vjix*delz - vjiz*delx);
		R_[i][1][2] += 0.5*jmass/rhoj*wfd*(vjiy*delz - vjiz*dely);
		
    }

  }

	epsilon_[i][1][0] = epsilon_[i][0][1];
	epsilon_[i][2][0] = epsilon_[i][0][2];
	epsilon_[i][2][1] = epsilon_[i][1][2];

	R_[i][1][0] = -1.0*R_[i][0][1];
	R_[i][2][0] = -1.0*R_[i][0][2];
	R_[i][2][1] = -1.0*R_[i][1][2];

	double trace = 1./3.*(epsilon_[i][0][0] + epsilon_[i][1][1] + epsilon_[i][2][2]);

	epsilonBar_[i][0][0] = epsilon_[i][0][0] - trace;
	epsilonBar_[i][1][1] = epsilon_[i][1][1] - trace;
	epsilonBar_[i][2][2] = epsilon_[i][2][2] - trace;
	epsilonBar_[i][0][1] = epsilon_[i][0][1];
	epsilonBar_[i][0][2] = epsilon_[i][0][2];
	epsilonBar_[i][1][2] = epsilon_[i][1][2];
	epsilonBar_[i][1][0] = epsilon_[i][0][1];
	epsilonBar_[i][2][0] = epsilon_[i][0][2];
	epsilonBar_[i][2][1] = epsilon_[i][1][2];

	double & dTauxx = dTau_[i][0][0];
	double & dTauyy = dTau_[i][1][1];
	double & dTauzz = dTau_[i][2][2];
	double & dTauxy = dTau_[i][0][1];
	double & dTauxz = dTau_[i][0][2];
	double & dTauyz = dTau_[i][1][2];
	double & dTauyx = dTau_[i][1][0];
	double & dTauzx = dTau_[i][2][0];
	double & dTauzy = dTau_[i][2][1];

	double Rxx = R_[i][0][0];
	double Ryy = R_[i][1][1];
	double Rzz = R_[i][2][2];
	double Rxy = R_[i][0][1];
	double Rxz = R_[i][0][2];
	double Ryz = R_[i][1][2];
	double Ryx = R_[i][1][0];
	double Rzx = R_[i][2][0];
	double Rzy = R_[i][2][1];

	double epsBarxx = epsilonBar_[i][0][0];
	double epsBaryy = epsilonBar_[i][1][1];
	double epsBarzz = epsilonBar_[i][2][2];
	double epsBarxy = epsilonBar_[i][0][1];
	double epsBarxz = epsilonBar_[i][0][2];
	double epsBaryz = epsilonBar_[i][1][2];
	double epsBaryx = epsilonBar_[i][1][0];
	double epsBarzx = epsilonBar_[i][2][0];
	double epsBarzy = epsilonBar_[i][2][1];

	//- Hooke Deviatoric Stress
	
	dTauxx = 2*G_*epsBarxx + ( tauxx*Rxx + tauyx*Rxy + tauzx*Rxz ) + ( tauxx*Rxx + tauxy*Rxy + tauxz*Rxz );
	dTauyy = 2*G_*epsBaryy + ( tauxy*Ryx + tauyy*Ryy + tauzy*Ryz ) + ( tauyx*Ryx + tauyy*Ryy + tauyz*Ryz );
	dTauzz = 2*G_*epsBarzz + ( tauxz*Rzx + tauyz*Rzy + tauzz*Rzz ) + ( tauzx*Rzx + tauzy*Rzy + tauzz*Rzz );
	dTauxy = 2*G_*epsBarxy + ( tauxy*Rxx + tauyy*Rxy + tauzy*Rxz ) + ( tauxx*Ryx + tauxy*Ryy + tauxz*Ryz );
	dTauxz = 2*G_*epsBarxz + ( tauxz*Rxx + tauyz*Rxy + tauzz*Rxz ) + ( tauxx*Rzx + tauxy*Rzy + tauxz*Rzz );
	dTauyz = 2*G_*epsBaryz + ( tauxz*Ryx + tauyz*Ryy + tauzz*Ryz ) + ( tauyx*Rzx + tauyy*Rzy + tauyz*Rzz );

	dTauyx = dTauxy;
	dTauzx = dTauxz;
	dTauzy = dTauyz;

	// This is WRONG way in solving deviatoric stress
	/*
	A[0][0] = 1.0/dt; 		A[0][1] = 0.0;    			A[0][2] = 0.0;    			A[0][3] = -2.0*R_[i][0][1]; A[0][4] = -2.0*R_[i][0][2]; 	A[0][5] = 0.0;
	A[1][0] = 0.0;    		A[1][1] = 1.0/dt; 			A[1][2] = 0.0;    			A[1][3] = 2.0*R_[i][0][1];  A[1][4] =  0.0; 			  	A[1][5] = -2.0*R_[i][1][2];
	A[2][0] = 0.0;    		A[2][1] = 0.0;    			A[2][2] = 1.0/dt; 			A[2][3] = 0.0; 			  	A[2][4] =  2.0*R_[i][0][2];  	A[2][5] = 2.0*R_[i][1][2];
 	A[3][0] = R_[i][0][1];  A[3][1] = -1.0*R_[i][0][1]; A[3][2] = 0.0;    			A[3][3] = 1.0/dt; 			A[3][4] = -1.0*R_[i][1][2]; 	A[3][5] = -1.0*R_[i][0][2];
	A[4][0] = R_[i][0][2];  A[4][1] = 0.0; 				A[4][2] = -1.0*R_[i][0][2]; A[4][3] = R_[i][1][2]; 		A[4][4] =  1.0/dt;  			A[4][5] = -1.0*R_[i][0][1];
	A[5][0] = 0.0;  		A[5][1] = R_[i][1][2]; 		A[5][2] = -1.0*R_[i][1][2]; A[5][3] = R_[i][0][2]; 		A[5][4] =  R_[i][0][1];  		A[5][5] = 1.0/dt;

	b[0] = 2*G_*epsilonBar_[i][0][0] + oldTauxx/dt;
	b[1] = 2*G_*epsilonBar_[i][1][1] + oldTauyy/dt;
	b[2] = 2*G_*epsilonBar_[i][2][2] + oldTauzz/dt;
	b[3] = 2*G_*epsilonBar_[i][0][1] + oldTauxy/dt;
	b[4] = 2*G_*epsilonBar_[i][0][2] + oldTauxz/dt;
	b[5] = 2*G_*epsilonBar_[i][1][2] + oldTauyz/dt;

//	solve6By6(tauxx, tauyy, tauzz, tauxy, tauxz, tauyz, A, b);
//	double my_x[6] = {0, 0, 0, 0, 0, 0};
	double my_x[6];
	MathExtraPysph::solve6By6_gsl(A, b, my_x);

	tauxx = my_x[0]; tauyy = my_x[1]; tauzz = my_x[2];
	tauxy = my_x[3]; tauxz = my_x[4]; tauyz = my_x[5];
	*/

	/*
	 * EOS 
	 * MIe-Gruneisen Equation
	 * Liu's Book, Page 297
	 */ 
	double eta = rho[i]/rho0_ - 1.0;
	a0_ = rho0_*Cs_*Cs_;
	b0_ = a0_*(1. + 2.*(S_ - 1.));
	c0_ = a0_*(2*(S_ - 1.) + 3*(S_ - 1.)*(S_ - 1.));
	double pH = eta > 0 ? (a0_*eta + b0_*eta*eta + c0_*eta*eta*eta) : a0_*eta;
//	p_[i] =  (1.0 - 0.5*Gamma_*eta)*pH + Gamma_*rho[i]*e[i]; 

	p_[i] = Cs_*Cs_*(rho[i] - rho0_);

	sigma_[i][0][0] = -p_[i] + tau_[i][0][0]; sigma_[i][0][1] = tau_[i][0][1]; sigma_[i][0][2] = tau_[i][0][2];
	sigma_[i][1][0] = tau_[i][1][0]; sigma_[i][1][1] = -p_[i] + tau_[i][1][1]; sigma_[i][1][2] = tau_[i][1][2];
	sigma_[i][2][0] = tau_[i][2][0]; sigma_[i][2][1] = tau_[i][2][1]; sigma_[i][2][2] = -p_[i] + tau_[i][2][2];

	// - Artificial Stress

	  double S[3][3];
	  S[0][0] = sigma_[i][0][0];
	  S[0][1] = sigma_[i][0][1];
	  S[0][2] = sigma_[i][0][2];

	  S[1][0] = sigma_[i][1][0];
	  S[1][1] = sigma_[i][1][1];
	  S[1][2] = sigma_[i][1][2];

	  S[2][0] = sigma_[i][2][0];
	  S[2][1] = sigma_[i][2][1];
	  S[2][2] = sigma_[i][2][2];

	  //- to avoid too small value
	  //- that will create difficulty
	  //- in decomposition_gsl
	  for(int ll = 0; ll < 3; ll++)
		{
	    	for(int mm = 0; mm < 3; mm++)
				{
					if(fabs(S[ll][mm]) < TOLERANCE) S[ll][mm] = 0; 
				}
		}

	  double R[3][3], Rab[3][3];
	  double V[3], rd[3];
	  R[0][0] = R[0][1] = R[0][2] = 0.0;
	  R[1][0] = R[1][1] = R[1][2] = 0.0;
	  R[2][0] = R[2][1] = R[2][2] = 0.0;

	  Rab[0][0] = Rab[0][1] = Rab[0][2] = 0.0;
	  Rab[1][0] = Rab[1][1] = Rab[1][2] = 0.0;
	  Rab[2][0] = Rab[2][1] = Rab[2][2] = 0.0;
	  V[0] = V[1] = V[2] = rd[0] = rd[1] = rd[2] = 0.0;

	  MathExtraPysph::eigen_decomposition_gsl_3D(S, R, V);

	  double rhoi2Inv = 1./rho[i]/rho[i];
	  if(V[0] > 0) rd[0] = -eps_*V[0]*rhoi2Inv;
	  else rd[0] = 0;

	  if(V[1] > 0) rd[1] = -eps_*V[1]*rhoi2Inv;
	  else rd[1] = 0;

	  if(V[2] > 0) rd[2] = -eps_*V[2]*rhoi2Inv;
	  else rd[2] = 0;

	  MathExtraPysph::transform_diag_inv_3D(rd, R, Rab);
	  
	  artStress_[i][0][0] = Rab[0][0];
	  artStress_[i][0][1] = Rab[0][1];
	  artStress_[i][0][2] = Rab[0][2];

	  artStress_[i][1][0] = Rab[1][0];
	  artStress_[i][1][1] = Rab[1][1];
	  artStress_[i][1][2] = Rab[1][2];

	  artStress_[i][2][0] = Rab[2][0];
	  artStress_[i][2][1] = Rab[2][1];
	  artStress_[i][2][2] = Rab[2][2];

