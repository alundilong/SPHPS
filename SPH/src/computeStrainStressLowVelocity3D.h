
	double tauxx = tau[i][0][0];
	double tauxy = tau[i][0][1];
	double tauxz = tau[i][0][2];
	double tauyx = tau[i][1][0];
	double tauyy = tau[i][1][1];
	double tauyz = tau[i][1][2];
	double tauzx = tau[i][2][0];
	double tauzy = tau[i][2][1];
	double tauzz = tau[i][2][2];

	double sigmaxx = sigma[i][0][0];
	double sigmaxy = sigma[i][0][1];
	double sigmaxz = sigma[i][0][2];
	double sigmayx = sigma[i][1][0];
	double sigmayy = sigma[i][1][1];
	double sigmayz = sigma[i][1][2];
	double sigmazx = sigma[i][2][0];
	double sigmazy = sigma[i][2][1];
	double sigmazz = sigma[i][2][2];

	double Jsqr = \
			   tauxx*tauxx + tauxy*tauxy+ tauxz*tauxz \
			 + tauyx*tauyx + tauyy*tauyy+ tauyz*tauyz \
			 + tauzx*tauzx + tauzy*tauzy+ tauzz*tauzz;

	Jsqr *= 0.5;
	J2 = Jsqr + SMALL;
  	// printf("J2 = %g\n", J2 );

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
		
		epsilon[i][0][0] += jmass/rhoj*vjix*wfd*delx;
		epsilon[i][1][1] += jmass/rhoj*vjiy*wfd*dely;
		epsilon[i][2][2] += jmass/rhoj*vjiz*wfd*delz;
		epsilon[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely + vjiy*delx);
		epsilon[i][0][2] += 0.5*jmass/rhoj*wfd*(vjix*delz + vjiz*delx);
		epsilon[i][1][2] += 0.5*jmass/rhoj*wfd*(vjiy*delz + vjiz*dely);

		R[i][0][1] += 0.5*jmass/rhoj*wfd*(vjix*dely - vjiy*delx);
		R[i][0][2] += 0.5*jmass/rhoj*wfd*(vjix*delz - vjiz*delx);
		R[i][1][2] += 0.5*jmass/rhoj*wfd*(vjiy*delz - vjiz*dely);
    }
  }

	epsilon[i][1][0] = epsilon[i][0][1];
	epsilon[i][2][0] = epsilon[i][0][2];
	epsilon[i][2][1] = epsilon[i][1][2];

	R[i][1][0] = -1.0*R[i][0][1];
	R[i][2][0] = -1.0*R[i][0][2];
	R[i][2][1] = -1.0*R[i][1][2];

	double trace = 1./3.*(epsilon[i][0][0] + epsilon[i][1][1] + epsilon[i][2][2]);
	double threeTrace = 3.0*trace;

	double tautrace = (tauxx + tauyy + tauzz);

	epsilonBar[i][0][0] = epsilon[i][0][0] - trace;
	epsilonBar[i][1][1] = epsilon[i][1][1] - trace;
	epsilonBar[i][2][2] = epsilon[i][2][2] - trace;
	epsilonBar[i][0][1] = epsilon[i][0][1];
	epsilonBar[i][0][2] = epsilon[i][0][2];
	epsilonBar[i][1][2] = epsilon[i][1][2];
	epsilonBar[i][1][0] = epsilon[i][0][1];
	epsilonBar[i][2][0] = epsilon[i][0][2];
	epsilonBar[i][2][1] = epsilon[i][1][2];

	double & dSigmaxx = dSigma[i][0][0];
	double & dSigmayy = dSigma[i][1][1];
	double & dSigmazz = dSigma[i][2][2];
	double & dSigmaxy = dSigma[i][0][1];
	double & dSigmaxz = dSigma[i][0][2];
	double & dSigmayz = dSigma[i][1][2];
	double & dSigmayx = dSigma[i][1][0];
	double & dSigmazx = dSigma[i][2][0];
	double & dSigmazy = dSigma[i][2][1];

	double Rxx = R[i][0][0];
	double Ryy = R[i][1][1];
	double Rzz = R[i][2][2];
	double Rxy = R[i][0][1];
	double Rxz = R[i][0][2];
	double Ryz = R[i][1][2];
	double Ryx = R[i][1][0];
	double Rzx = R[i][2][0];
	double Rzy = R[i][2][1];

	double epsBarxx = epsilonBar[i][0][0];
	double epsBaryy = epsilonBar[i][1][1];
	double epsBarzz = epsilonBar[i][2][2];
	double epsBarxy = epsilonBar[i][0][1];
	double epsBarxz = epsilonBar[i][0][2];
	double epsBaryz = epsilonBar[i][1][2];
	double epsBaryx = epsilonBar[i][1][0];
	double epsBarzx = epsilonBar[i][2][0];
	double epsBarzy = epsilonBar[i][2][1];

	if(!elastic_flag) find_lambda_dot(i);
	find_properties(i);

	//- Hooke Deviatoric Stress
	
	dSigmaxx = 2*G*epsBarxx + ( sigmaxx*Rxx + sigmayx*Rxy + sigmazx*Rxz ) + ( sigmaxx*Rxx + sigmaxy*Rxy + sigmaxz*Rxz ) + K*threeTrace;
	dSigmayy = 2*G*epsBaryy + ( sigmaxy*Ryx + sigmayy*Ryy + sigmazy*Ryz ) + ( sigmayx*Ryx + sigmayy*Ryy + sigmayz*Ryz ) + K*threeTrace;
	dSigmazz = 2*G*epsBarzz + ( sigmaxz*Rzx + sigmayz*Rzy + sigmazz*Rzz ) + ( sigmazx*Rzx + sigmazy*Rzy + sigmazz*Rzz ) + K*threeTrace;
	dSigmaxy = 2*G*epsBarxy + ( sigmaxy*Rxx + sigmayy*Rxy + sigmazy*Rxz ) + ( sigmaxx*Ryx + sigmaxy*Ryy + sigmaxz*Ryz );
	dSigmaxz = 2*G*epsBarxz + ( sigmaxz*Rxx + sigmayz*Rxy + sigmazz*Rxz ) + ( sigmaxx*Rzx + sigmaxy*Rzy + sigmaxz*Rzz );
	dSigmayz = 2*G*epsBaryz + ( sigmaxz*Ryx + sigmayz*Ryy + sigmazz*Ryz ) + ( sigmayx*Rzx + sigmayy*Rzy + sigmayz*Rzz );

	double	dSigmapxx = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauxx + (3*K-2*G)*tautrace/3);
	double	dSigmapyy = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauyy + (3*K-2*G)*tautrace/3);
	double	dSigmapzz = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauzz + (3*K-2*G)*tautrace/3);
	double	dSigmapxy = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauxy);
	double	dSigmapxz = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauxz);
	double	dSigmapyz = 0.5*lambda_dot*sqrt(3/J2)*(2*G*tauyz);

	dSigmaxx -= dSigmapxx;
	dSigmayy -= dSigmapyy;
	dSigmazz -= dSigmapzz;
	dSigmaxy -= dSigmapxy;
	dSigmaxz -= dSigmapxz;
	dSigmayz -= dSigmapyz;

	dSigmayx = dSigmaxy;
	dSigmazx = dSigmaxz;
	dSigmazy = dSigmayz;

	tmp_sigma[0] = sigma[i][0][0] + dSigmaxx*dt;
	tmp_sigma[1] = sigma[i][0][1] + dSigmaxy*dt;
	tmp_sigma[2] = sigma[i][0][2] + dSigmaxz*dt;
	tmp_sigma[3] = sigma[i][1][0] + dSigmayx*dt;
	tmp_sigma[4] = sigma[i][1][1] + dSigmayy*dt;
	tmp_sigma[5] = sigma[i][1][2] + dSigmayz*dt;
	tmp_sigma[6] = sigma[i][2][0] + dSigmazx*dt;
	tmp_sigma[7] = sigma[i][2][1] + dSigmazy*dt;
	tmp_sigma[8] = sigma[i][2][2] + dSigmazz*dt;

	double pressureTrace = 1./3.*(tmp_sigma[0] + tmp_sigma[4] + tmp_sigma[8]);
	p[i] = -pressureTrace;

	tau[i][0][0] = -pressureTrace + tmp_sigma[0]; tau[i][0][1] = tmp_sigma[1]; tau[i][0][2] = tmp_sigma[2];
	tau[i][1][0] = tmp_sigma[3]; tau[i][1][1] = -pressureTrace + tmp_sigma[4]; tau[i][1][2] = tmp_sigma[5];
	tau[i][2][0] = tmp_sigma[6]; tau[i][2][1] = tmp_sigma[7]; tau[i][2][2] = -pressureTrace + tmp_sigma[8];

	tauxx = tau[i][0][0];
	tauxy = tau[i][0][1];
	tauxz = tau[i][0][2];
	tauyx = tau[i][1][0];
	tauyy = tau[i][1][1];
	tauyz = tau[i][1][2];
	tauzx = tau[i][2][0];
	tauzy = tau[i][2][1];
	tauzz = tau[i][2][2];

	Jsqr = \
		   tauxx*tauxx + tauxy*tauxy+ tauxz*tauxz \
		 + tauyx*tauyx + tauyy*tauyy+ tauyz*tauyz \
		 + tauzx*tauzx + tauzy*tauzy+ tauzz*tauzz;

	Jsqr *= 0.5;
	J2 = Jsqr + SMALL;

	// - Artificial Stress

	  double S[3][3];
	  S[0][0] = tmp_sigma[0];
	  S[0][1] = tmp_sigma[1];
	  S[0][2] = tmp_sigma[2];

	  S[1][0] = tmp_sigma[3];
	  S[1][1] = tmp_sigma[4];
	  S[1][2] = tmp_sigma[5];

	  S[2][0] = tmp_sigma[6];
	  S[2][1] = tmp_sigma[7];
	  S[2][2] = tmp_sigma[8];

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

