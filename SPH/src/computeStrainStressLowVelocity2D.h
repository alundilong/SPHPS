
	double tauxx = tau[i][0][0];
	double tauxy = tau[i][0][1];
	double tauyx = tau[i][1][0];
	double tauyy = tau[i][1][1];

	double sigmaxx = sigma[i][0][0];
	double sigmaxy = sigma[i][0][1];
	double sigmayx = sigma[i][1][0];
	double sigmayy = sigma[i][1][1];

	double Jsqr = \
			   tauxx*tauxx + tauxy*tauxy \
			 + tauyx*tauyx + tauyy*tauyy;

	Jsqr *= 0.5;
	J2 = Jsqr+SMALL;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      rsq = delx * delx + dely * dely;
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

		// - compute strains

		double vjix = v[j][0] - v[i][0];
		double vjiy = v[j][1] - v[i][1];
		double vjiz = v[j][2] - v[i][2];
		
		epsilon[i][0][0] += jmass/rhoj*vjix*wfd*delx;
		epsilon[i][1][1] += jmass/rhoj*vjiy*wfd*dely;
		epsilon[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely + vjiy*delx);

		R[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely - vjiy*delx);
//		printf("iter = %d, i = %d, j = %d, wfd = %f, vix = %f, viy = %f,  vjx = %f, vjy = %f, rsq = %f, rsqijtype = %f\n", iter, i, j, wfd, v[i][0], v[i][1], v[j][0], v[j][1], rsq, cutsq[itype][jtype]);
    }
  }

	epsilon[i][1][0] = epsilon[i][0][1];

	R[i][1][0] = -1.0*R[i][0][1];

	double trace = 1./2.*(epsilon[i][0][0] + epsilon[i][1][1]);
	double threeTrace = 2.0*trace;
	double tautrace = (tauxx + tauyy);

	epsilonBar[i][0][0] = epsilon[i][0][0] - trace;
	epsilonBar[i][1][1] = epsilon[i][1][1] - trace;
	epsilonBar[i][0][1] = epsilon[i][0][1];
	epsilonBar[i][1][0] = epsilon[i][0][1];

	double & dSigmaxx = dSigma[i][0][0];
	double & dSigmayy = dSigma[i][1][1];
	double & dSigmaxy = dSigma[i][0][1];
	double & dSigmayx = dSigma[i][1][0];

	double Rxx = R[i][0][0];
	double Ryy = R[i][1][1];
	double Rxy = R[i][0][1];
	double Ryx = R[i][1][0];

	double epsBarxx = epsilonBar[i][0][0];
	double epsBaryy = epsilonBar[i][1][1];
	double epsBarxy = epsilonBar[i][0][1];
	double epsBaryx = epsilonBar[i][1][0];

	if(!elastic_flag) find_lambda_dot(i);
  	find_properties(i);
//	printf("lambda_dot = %f\n", lambda_dot);

	//- Hooke Deviatoric Stress
	
	//printf("G = %f, K = %f, lambda_dot = %f\n", G, K, lambda_dot);
	dSigmaxx = 2*G*epsBarxx + ( sigmaxx*Rxx + sigmayx*Rxy ) + ( sigmaxx*Rxx + sigmaxy*Rxy ) + K*threeTrace;
	dSigmayy = 2*G*epsBaryy + ( sigmaxy*Ryx + sigmayy*Ryy ) + ( sigmayx*Ryx + sigmayy*Ryy ) + K*threeTrace;
	dSigmaxy = 2*G*epsBarxy + ( sigmaxy*Rxx + sigmayy*Rxy ) + ( sigmaxx*Ryx + sigmaxy*Ryy );

	double	dSigmapxx = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauxx + (3*K-2*G)*tautrace/3);
	double	dSigmapyy = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauyy + (3*K-2*G)*tautrace/3);
	double	dSigmapxy = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauxy);

	
	dSigmaxx -= dSigmapxx;
	dSigmayy -= dSigmapyy;
	dSigmaxy -= dSigmapxy;
	
	dSigmayx = dSigmaxy;

	tmp_sigma[0] = sigma[i][0][0] + dSigmaxx*dt;
	tmp_sigma[1] = sigma[i][0][1] + dSigmaxy*dt;
	tmp_sigma[2] = sigma[i][1][0] + dSigmayx*dt;
	tmp_sigma[3] = sigma[i][1][1] + dSigmayy*dt;

//	if(!elastic_flag) printf(" syy = %f\n", tmp_sigma[3]);

	double pressureTrace = 1./2.*(tmp_sigma[0] + tmp_sigma[3]);
	p[i] = -pressureTrace;
	// printf("timestep = %d, pressure  = %f, sigmaxx = %f, sigmayy = %f, i = %d\n", update->ntimestep, p[i], sigmaxx, sigmayy, i);

	// printf("pressure = %f, sigma0 = %f, sigma1 = %f, sigma2 = %f, sigma3\n", pressureTrace, tmp_sigma[0], tmp_sigma[1], tmp_sigma[2], tmp_sigma[3]);
	tau[i][0][0] = -pressureTrace + tmp_sigma[0]; tau[i][0][1] = tmp_sigma[1]; 
	tau[i][1][0] = tmp_sigma[2]; tau[i][1][1] = -pressureTrace + tmp_sigma[3];

	tauxx = tau[i][0][0];
	tauxy = tau[i][0][1];
	tauyx = tau[i][1][0];
	tauyy = tau[i][1][1];

	Jsqr = \
		   tauxx*tauxx + tauxy*tauxy \
		 + tauyx*tauyx + tauyy*tauyy;

	Jsqr *= 0.5;
	J2 = Jsqr+SMALL;

	// - Artificial Stress

	  double S[2][2];
	  S[0][0] = tmp_sigma[0];
	  S[0][1] = tmp_sigma[1];

	  S[1][0] = tmp_sigma[2];
	  S[1][1] = tmp_sigma[3];

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
	  V[0] = V[1] = rd[0] = rd[1]  = 0.0;

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

