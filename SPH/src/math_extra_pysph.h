/* ----------------------------------------------------------------------
   functions (most) borrowed from the PYSPH 
------------------------------------------------------------------------- */

#ifndef LMP_MATH_EXTRA_PYSPH_H
#define LMP_MATH_EXTRA_PYSPH_H

#include "math.h"
#include "stdio.h"
#include "string.h"
#include "error.h"
#include "vector_liggghts.h"
#include "math_extra.h"
#include "gsl/gsl_math.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_linalg.h"

#define TOLERANCE_ORTHO 1e-10
#define NN 3

namespace MathExtraPysph {

  inline void transform_diag_inv_2D(double *A, double P[2][2], double res[2][2]);
  inline void transform_diag_inv_3D(double *A, double P[3][3], double res[3][3]);
//  inline void transform_diag_inv2_2D(double *A, double P[2][2], double res[2][2]);
//  inline void transform_diag_inv2_3D(double *A, double P[3][3], double res[3][3]);
//  inline void transform_diag_inv2(double *A, double P[3][3], double res[3][3]);
  inline double * tred2_2D(double V[2][2], double *d, double *e);
  inline double * tred2_3D(double V[3][3], double *d, double *e);
  inline void tql2_2D(double V[2][2], double *d, double *e);
  inline void tql2_3D(double V[3][3], double *d, double *e);
  inline void zero_matrix_case_2D(double V[2][2], double *d);
  inline void zero_matrix_case_3D(double V[3][3], double *d);
  inline void eigen_decomposition_2D(double A[2][2], double V[2][2], double *d);
  inline void eigen_decomposition_3D(double A[3][3], double V[3][3], double *d);
//  inline void eigen_decomposition2(double A[3][3], double V[3][3], double *d, double delta);
  inline void eigen_decomposition_gsl(double A[3][3], double V[3][3], double *d);
  inline void eigen_decomposition_gsl_3D(double A[3][3], double V[3][3], double *d);
  inline void eigen_decomposition_gsl_2D(double A[2][2], double V[2][2], double *d);
  inline double hypot2(double x, double y);
  inline void bubbleSort(double arr[], int n, int inc = 1);
  inline void solve6By6_gsl(double A[6][6], double b[6], double *x);
};

/*
 * Compute the transformation P*A*P.T and set it into res.
 * A is diagonal and contains just the diagonal entries.
 */

void MathExtraPysph::transform_diag_inv_3D(double *A, double P[3][3], double res[3][3])
{
    int i, j, k;
    for (i = 0; i < 3; i++)
    	for (j = 0; j < 3; j++)
            res[i][j] = 0.0;

    for (i = 0; i < 3; i++)
    	for (j = 0; j < 3; j++)
    		for (k = 0; k < 3; k++)
                res[i][j] += P[i][k]*A[k]*P[j][k]; // P*A*P.T
}

/*
void MathExtraPysph::transform_diag_inv(double *A, double P[3][3], double res[3][3])
{
    int i, j, k;
    for (i = 0; i < 3; i++)
    	for (j = 0; j < 3; j++)
            res[i][j] = 0.0;

    for (i = 0; i < 3; i++)
    	for (j = 0; j < 3; j++)
    		for (k = 0; k < 3; k++)
                res[i][j] += P[i][k]*A[k]*P[j][k]; // P*A*P.T
}
*/

void MathExtraPysph::transform_diag_inv_2D(double *A, double P[2][2], double res[2][2])
{
    int i, j, k;
    for (i = 0; i < 2; i++)
    	for (j = 0; j < 2; j++)
            res[i][j] = 0.0;

    for (i = 0; i < 2; i++)
    	for (j = 0; j < 2; j++)
    		for (k = 0; k < 2; k++)
                res[i][j] += P[i][k]*A[k]*P[j][k]; // P*A*P.T
}
/*
void MathExtraPysph::transform_diag_inv2_2D(double *A, double P[2][2], double res[2][2])
{
    int i, j;
    for (i = 0; i < 2; i++)
    	for (j = 0; j < 2; j++)
		{
            res[i][j] = 0.0;
		}

	double l1 = P[0][0]; double m1 = P[0][1];
	double l2 = P[1][0]; double m2 = P[1][1];
	double R1 = A[0]; double R2 = A[1];
	res[0][0] = l1*l1*R1 + m1*m1*R2;
	res[0][1] = l1*l2*R1 + m1*m2*R2;
	res[1][0] = l1*l2*R1 + m1*m2*R2;
	res[1][1] = l2*l2*R1 + m2*m2*R2;
}

void MathExtraPysph::transform_diag_inv2_3D(double *A, double P[3][3], double res[3][3])
{
    int i, j;
    for (i = 0; i < 3; i++)
    	for (j = 0; j < 3; j++)
		{
            res[i][j] = 0.0;
		}

	double l1 = P[0][0]; double m1 = P[0][1]; double n1 = P[0][2];
	double l2 = P[1][0]; double m2 = P[1][1]; double n2 = P[1][2];
	double l3 = P[2][0]; double m3 = P[2][1]; double n3 = P[2][2];
	double R1 = A[0]; double R2 = A[1]; double R3 = A[2];
	res[0][0] = l1*l1*R1 + m1*m1*R2 + n1*n1*R3;
	res[0][1] = l1*l2*R1 + m1*m2*R2 + n1*n2*R3;
	res[0][2] = l1*l3*R1 + m1*m3*R2 + n1*n3*R3;
	res[1][0] = l1*l2*R1 + m1*m2*R2 + n1*n2*R3;
	res[1][1] = l2*l2*R1 + m2*m2*R2 + n2*n2*R3;
	res[1][2] = l2*l3*R1 + m2*m3*R2 + n2*n3*R3;
	res[2][1] = l2*l3*R1 + m2*m3*R2 + n2*n3*R3;
	res[2][0] = l1*l3*R1 + m1*m3*R2 + n1*n3*R3;
	res[2][2] = l3*l3*R1 + m3*m3*R2 + n3*n3*R3;

}
*/
/*
void MathExtraPysph::transform_diag_inv2(double *A, double P[3][3], double res[3][3])
{
    int i, j;
    for (i = 0; i < 3; i++)
    	for (j = 0; j < 3; j++)
		{
            res[i][j] = 0.0;
		}

	double l1 = P[0][0]; double m1 = P[0][1]; double n1 = P[0][2];
	double l2 = P[1][0]; double m2 = P[1][1]; double n2 = P[1][2];
	double l3 = P[2][0]; double m3 = P[2][1]; double n3 = P[2][2];
	double R1 = A[0]; double R2 = A[1]; double R3 = A[2];
	res[0][0] = l1*l1*R1 + m1*m1*R2 + n1*n1*R3;
	res[0][1] = l1*l2*R1 + m1*m2*R2 + n1*n2*R3;
	res[0][2] = l1*l3*R1 + m1*m3*R2 + n1*n3*R3;
	res[1][0] = l1*l2*R1 + m1*m2*R2 + n1*n2*R3;
	res[1][1] = l2*l2*R1 + m2*m2*R2 + n2*n2*R3;
	res[1][2] = l2*l3*R1 + m2*m3*R2 + n2*n3*R3;
	res[2][1] = l2*l3*R1 + m2*m3*R2 + n2*n3*R3;
	res[2][0] = l1*l3*R1 + m1*m3*R2 + n1*n3*R3;
	res[2][2] = l3*l3*R1 + m3*m3*R2 + n3*n3*R3;

}
*/

/*
	Symmetric Householder reduction to tridiagonal form

	This is derived from the Algol procedures tred2 by
	Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	Fortran subroutine in EISPACK.

	d contains the diagonal elements of the tridiagonal matrix.

	e contains the subdiagonal elements of the tridiagonal matrix in its
	last n-1 positions.  e[0] is set to zero.
*/

double * MathExtraPysph::tred2_2D(double V[2][2], double *d, double *e)
{
	int n = 2;
	double scale, f, g, h, hh;
	int i, j, k;

	for (j = 0; j < n; j++)
	{
        d[j] = V[n-1][j];
	}

    // Householder reduction to tridiagonal form.

    for(i = n-1; i > 0 ; i--)
	{
        // Scale to avoid under/overflow.
        scale = 0.0;
        h = 0.0;
		for (k = 0; k < i; k++ )
            scale += fabs(d[k]);

        if (scale == 0.0)
		{
            e[i] = d[i-1];
            for (j=0; j < i; j++)
			{
                d[j] = V[i-1][j];
                V[i][j] = 0.0;
                V[j][i] = 0.0;
			}
		}
        else {
            // Generate Householder vector.
            for (k = 0; k < i; k++)
			{
                d[k] /= scale;
                h += d[k] * d[k];
			}

            f = d[i-1];
            g = sqrt(h);
            if (f > 0)
                g = -g;

            e[i] = scale * g;
            h = h - f * g;
            d[i-1] = f - g;
            for (j = 0; j<i; j++)
                e[j] = 0.0;
		}

        // Apply similarity transformation to remaining columns.

			for (j = 0; j<i; j++)
			{
                f = d[j];
                V[j][i] = f;
                g = e[j] + V[j][j] * f;
                for (k = j+1; k < i; k++)
				{
                    g += V[k][j] * d[k];
                    e[k] += V[k][j] * f;
				}

                e[j] = g;
			}

            f = 0.0;
			for (j = 0; j<i; j++)
			{
                e[j] /= h;
                f += e[j] * d[j];
			}

            hh = f / (h + h);
			for (j = 0; j<i; j++)
                e[j] -= hh * d[j];

			for (j = 0; j<i; j++)
			{
                f = d[j];
                g = e[j];
                for (k = j; k<i; k++)
                    V[k][j] -= (f * e[k] + g * d[k]);

                d[j] = V[i-1][j];
                V[i][j] = 0.0;
			}

        d[i] = h;
	}
    // Accumulate transformations.

    for (i = 0; i < n-1; i++)
	{
        V[n-1][i] = V[i][i];
        V[i][i] = 1.0;
        h = d[i+1];
        if (h != 0.0)
		{
			for (k = 0; k<i+1; k++)
                d[k] = V[k][i+1] / h;

			for (j = 0; j< i+1; j++)
			{
                g = 0.0;
				for (k = 0; k< i+1; k++)
                    g += V[k][i+1] * V[k][j];

				for (k = 0; k< i+1; k++)
                    V[k][j] -= g * d[k];
			}
		}


		for(k = 0; k< i+1; k++)
            V[k][i+1] = 0.0;
	}

	for (j = 0; j<n; j++)
	{
        d[j] = V[n-1][j];
        V[n-1][j] = 0.0;
	}

    V[n-1][n-1] = 1.0;
    e[0] = 0.0;

    return d;
}

double * MathExtraPysph::tred2_3D(double V[3][3], double *d, double *e)
{
	int n = 3;
	double scale, f, g, h, hh;
	int i, j, k;

	for (j = 0; j < n; j++)
	{
        d[j] = V[n-1][j];
	}

    // Householder reduction to tridiagonal form.

    for(i = n-1; i > 0 ; i--)
	{
        // Scale to avoid under/overflow.
        scale = 0.0;
        h = 0.0;
		for (k = 0; k < i; k++ )
            scale += fabs(d[k]);

        if (scale == 0.0)
		{
            e[i] = d[i-1];
            for (j=0; j < i; j++)
			{
                d[j] = V[i-1][j];
                V[i][j] = 0.0;
                V[j][i] = 0.0;
			}
		}
        else {
            // Generate Householder vector.
            for (k = 0; k < i; k++)
			{
                d[k] /= scale;
                h += d[k] * d[k];
			}

            f = d[i-1];
            g = sqrt(h);
            if (f > 0)
                g = -g;

            e[i] = scale * g;
            h = h - f * g;
            d[i-1] = f - g;
            for (j = 0; j<i; j++)
                e[j] = 0.0;
		}

        // Apply similarity transformation to remaining columns.

			for (j = 0; j<i; j++)
			{
                f = d[j];
                V[j][i] = f;
                g = e[j] + V[j][j] * f;
                for (k = j+1; k < i; k++)
				{
                    g += V[k][j] * d[k];
                    e[k] += V[k][j] * f;
				}

                e[j] = g;
			}

            f = 0.0;
			for (j = 0; j<i; j++)
			{
                e[j] /= h;
                f += e[j] * d[j];
			}

            hh = f / (h + h);
			for (j = 0; j<i; j++)
                e[j] -= hh * d[j];

			for (j = 0; j<i; j++)
			{
                f = d[j];
                g = e[j];
                for (k = j; k<i; k++)
                    V[k][j] -= (f * e[k] + g * d[k]);

                d[j] = V[i-1][j];
                V[i][j] = 0.0;
			}

        d[i] = h;
	}
    // Accumulate transformations.

    for (i = 0; i < n-1; i++)
	{
        V[n-1][i] = V[i][i];
        V[i][i] = 1.0;
        h = d[i+1];
        if (h != 0.0)
		{
			for (k = 0; k<i+1; k++)
                d[k] = V[k][i+1] / h;

			for (j = 0; j< i+1; j++)
			{
                g = 0.0;
				for (k = 0; k< i+1; k++)
                    g += V[k][i+1] * V[k][j];

				for (k = 0; k< i+1; k++)
                    V[k][j] -= g * d[k];
			}
		}


		for(k = 0; k< i+1; k++)
            V[k][i+1] = 0.0;
	}

	for (j = 0; j<n; j++)
	{
        d[j] = V[n-1][j];
        V[n-1][j] = 0.0;
	}

    V[n-1][n-1] = 1.0;
    e[0] = 0.0;

    return d;
}

/*
	Symmetric tridiagonal QL algo for eigendecomposition

	This is derived from the Algol procedures tql2, by
	Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	Fortran subroutine in EISPACK.

	d contains the eigenvalues in ascending order.  if an error exit is
	made, the eigenvalues are correct but unordered for indices
	1,2,...,ierr-1.

	e has been destroyed.
*/

void MathExtraPysph::tql2_2D(double V[2][2], double *d, double *e)
{
	int n = 2;
	double f, tst1, eps, g, h, p, r, dl1, c, c2, c3, el1, s, s2;
	int i, j, k, l, m, iter;
	bool cont;

    for (i = 1; i<n; i++)
        e[i-1] = e[i];

    e[n-1] = 0.0;

    f = 0.0;
    tst1 = 0.0;
    eps = 2.220446049250313e-16; // = pow(2.0, -52);
	for (l = 0; l<n; l++)
	{
        // Find small subdiagonal element
        tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
        m = l;
        while (m < n)
		{
            if (fabs(e[m]) <= eps*tst1)
                break;
            m += 1;
		}

        // If m == l, d[l] is an eigenvalue,
        // otherwise, iterate.
        if (m > l)
		{
            iter = 0;
            cont = true;
            while (cont)
			{
                iter = iter + 1;        // (Could check iteration count here.)

                // Compute implicit shift
                g = d[l];
                p = (d[l+1] - g) / (2.0 * e[l]);
                r = hypot2(p,1.0);
                if (p < 0)
                    r = -r;

                d[l] = e[l] / (p + r);
                d[l+1] = e[l] * (p + r);
                dl1 = d[l+1];
                h = g - d[l];
				for(i = l+2; i<n; i++)
                    d[i] -= h;

                f += h;

                // Implicit QL transformation.
                p = d[m];
                c = 1.0;
                c2 = c;
                c3 = c;
                el1 = e[l+1];
                s = 0.0;
                s2 = 0.0;
				for (i = m-1; i< l-1; i--)
				{
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot2(p,e[i]);
                    e[i+1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i+1] = h + s * (c * g + s * d[i]);

                    // Accumulate transformation
					for (k = 0; k<n; k++)
					{
                        h = V[k][i+1];
                        V[k][i+1] = s * V[k][i] + c * h;
                        V[k][i] = c * V[k][i] - s * h;
					}
				}

                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Check for convergence
                cont = bool(fabs(e[l]) > eps*tst1);
			} // end - while
		} // end if
        d[l] += f;
        e[l] = 0.0;
	}

    // Sort eigenvalues and corresponding vectors.
	for (i = 0; i<n-1; i++)
	{
        k = i;
        p = d[i];
		for (j = i+1; j<n; j++)
		{
            if (d[j] < p)
			{
                k = j;
                p = d[j];
			}
		}

        if (k != i)
		{
            d[k] = d[i];
            d[i] = p;
			for (j = 0; j<n; j++)
			{
                p = V[j][i];
                V[j][i] = V[j][k];
                V[j][k] = p;
			}
		}
	}
}

void MathExtraPysph::tql2_3D(double V[3][3], double *d, double *e)
{
	int n = 3;
	double f, tst1, eps, g, h, p, r, dl1, c, c2, c3, el1, s, s2;
	int i, j, k, l, m, iter;
	bool cont;

    for (i = 1; i<n; i++)
        e[i-1] = e[i];

    e[n-1] = 0.0;

    f = 0.0;
    tst1 = 0.0;
    eps = 2.220446049250313e-16; // = pow(2.0, -52);
	for (l = 0; l<n; l++)
	{
        // Find small subdiagonal element
        tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
        m = l;
        while (m < n)
		{
            if (fabs(e[m]) <= eps*tst1)
                break;
            m += 1;
		}

        // If m == l, d[l] is an eigenvalue,
        // otherwise, iterate.
        if (m > l)
		{
            iter = 0;
            cont = true;
            while (cont)
			{
                iter = iter + 1;        // (Could check iteration count here.)

                // Compute implicit shift
                g = d[l];
                p = (d[l+1] - g) / (2.0 * e[l]);
                r = hypot2(p,1.0);
                if (p < 0)
                    r = -r;

                d[l] = e[l] / (p + r);
                d[l+1] = e[l] * (p + r);
                dl1 = d[l+1];
                h = g - d[l];
				for(i = l+2; i<n; i++)
                    d[i] -= h;

                f += h;

                // Implicit QL transformation.
                p = d[m];
                c = 1.0;
                c2 = c;
                c3 = c;
                el1 = e[l+1];
                s = 0.0;
                s2 = 0.0;
				for (i = m-1; i< l-1; i--)
				{
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e[i];
                    h = c * p;
                    r = hypot2(p,e[i]);
                    e[i+1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i+1] = h + s * (c * g + s * d[i]);

                    // Accumulate transformation
					for (k = 0; k<n; k++)
					{
                        h = V[k][i+1];
                        V[k][i+1] = s * V[k][i] + c * h;
                        V[k][i] = c * V[k][i] - s * h;
					}
				}

                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;

                // Check for convergence
                cont = bool(fabs(e[l]) > eps*tst1);
			} // end - while
		} // end if
        d[l] += f;
        e[l] = 0.0;
	}

    // Sort eigenvalues and corresponding vectors.
	for (i = 0; i<n-1; i++)
	{
        k = i;
        p = d[i];
		for (j = i+1; j<n; j++)
		{
            if (d[j] < p)
			{
                k = j;
                p = d[j];
			}
		}

        if (k != i)
		{
            d[k] = d[i];
            d[i] = p;
			for (j = 0; j<n; j++)
			{
                p = V[j][i];
                V[j][i] = V[j][k];
                V[j][k] = p;
			}
		}
	}
}

void MathExtraPysph::zero_matrix_case_2D(double V[2][2], double *d)
{
	int n = 2;
    int i, j;
	for (i = 0; i<2; i++)
	{
        d[i] = 0.0;
		for (j = 0; j<2; j++)
		{
            V[i][j] = (i==j);
		}
	}
}

void MathExtraPysph::zero_matrix_case_3D(double V[3][3], double *d)
{
	int n = 3;
    int i, j;
	for (i = 0; i<3; i++)
	{
        d[i] = 0.0;
		for (j = 0; j<3; j++)
		{
            V[i][j] = (i==j);
		}
	}
}

/*
	Get eigenvalues and eigenvectors of matrix A.
	V is output eigenvectors and d are the eigenvalues.
*/

void MathExtraPysph::eigen_decomposition_2D(double A[2][2], double V[2][2], double *d)
{
	int n = 2;
    double e[n];
    int i, j;
    // Scale the matrix, as if the matrix is tiny, floating point errors
    // creep up leading to zero division errors in tql2.  This is
    // specifically tested for with a tiny matrix.
    double s = 0.0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j<n; j++)
		{
            V[i][j] = A[i][j];
            s += fabs(V[i][j]);
		}
	}

    if (s == 0)
        zero_matrix_case_2D(V, d);
    else 
	{
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
                V[i][j] /= s;

        d = tred2_2D(V, d, &e[0]);
        tql2_2D(V, d, &e[0]);
		for (i = 0; i < n; i++)
            d[i] *= s;
	}
}

void MathExtraPysph::eigen_decomposition_3D(double A[3][3], double V[3][3], double *d)
{
	int n = 3;
    double e[n];
    int i, j;
    // Scale the matrix, as if the matrix is tiny, floating point errors
    // creep up leading to zero division errors in tql2.  This is
    // specifically tested for with a tiny matrix.
    double s = 0.0;
	for (i = 0; i < n; i++)
	{
		for (j = 0; j<n; j++)
		{
            V[i][j] = A[i][j];
            s += fabs(V[i][j]);
		}
	}

    if (s == 0)
        zero_matrix_case_3D(V, d);
    else 
	{
		for (i = 0; i < n; i++)
			for (j = 0; j < n; j++)
                V[i][j] /= s;

        d = tred2_3D(V, d, &e[0]);
        tql2_3D(V, d, &e[0]);
		for (i = 0; i < n; i++)
            d[i] *= s;
	}
}

/*
void MathExtraPysph::eigen_decomposition2(double A[NN][NN], double V[NN][NN], double *d, double delta)
{
	double sxx = A[0][0]; double sxy = A[0][1]; double sxz = A[0][2];
	double syx = A[1][0]; double syy = A[1][1]; double syz = A[1][2];
	double szx = A[2][0]; double szy = A[2][1]; double szz = A[2][2];
	double sum = fabs(sxx) + fabs(sxy) + fabs(sxz) + fabs(syx) + fabs(syy) + fabs(syz) + fabs(szx) + fabs(szy) + fabs(szz);
	if(sum > 0)
	{
		double nondiagSum = fabs(sxy) + fabs(syx) + fabs(sxz) + fabs(szx) + fabs(syz) + fabs(szy);
		bool diag = (nondiagSum == 0);
//		if(diag) printf("diag = %d, sum = %f\n", diag, nondiagSum);
		if(!diag)
		{
			double I1 = sxx + syy + szz;
			double I2 = sxx*syy + sxx*szz + syy*szz - sxy*sxy - syz*syz - sxz*sxz;
			double I3 = sxx*syy*szz + 2*sxy*syz*sxz - sxx*syz*syz - syy*sxz*sxz - szz*sxy*sxy;
			double R = I1*I1/3. - I2;
			double S = sqrt(R/3.);
			double Q = I1*I2/3. - I3 - 2*I1*I1*I1/27;
			double T = sqrt(R*R*R/27.);
			double Q2T = -Q/2/(T + delta);
			double alpha = acos(Q2T);
			double sa = 2*S*cos(alpha/3.) + I1/3.;
			double sb = 2*S*cos(alpha/3. + 2*M_PI/3.) + I1/3.;
			double sc = 2*S*cos(alpha/3. + 4*M_PI/3.) + I1/3.;
//			printf("I1 = %f, alpha = %f, Q = %f, T = %f\n", I1, alpha, Q, T);

			d[0] = sa; d[1] = sb; d[2] = sc;
			//bubbleSort(d, 3, 0);

			double a[3];
			double b[3];
			double c[3];
			double k[3];
			a[0] = a[1] = a[2] = 0;
			b[0] = b[1] = b[2] = 0;
			c[0] = c[1] = c[2] = 0;
			for(int i = 0; i < 3; i++)
			{
				a[i] = (syy-d[i])*(szz-d[i]) - syz*syz;
				b[i] = -(sxy*(szz-d[i]) - syz*sxz);
				c[i] = sxy*syz - (syy - d[i])*sxz;

				double sqrtabc = sqrt(a[i]*a[i] + b[i]*b[i] + c[i]*c[i]);
				k[i] = 1./sqrtabc; 

				V[0][i] = a[i]*k[i]; 
				V[1][i] = b[i]*k[i]; 
				V[2][i] = c[i]*k[i]; 
			}
		}
		else
		{
			d[0] = sxx;
			d[1] = syy;
			d[2] = szz;
			for(int i = 0; i < 3; i++)
			{
				V[0][i] = (i==0); 
				V[1][i] = (i==1); 
				V[2][i] = (i==2); 
			}

		}
	}
	else
	{
		d[0] = d[1] = d[2] = 0.0;
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
					V[i][j] = 0;
		}
	}
}
*/

/*
	Get eigenvalues and eigenvectors of matrix A.
	V is output eigenvectors and d are the eigenvalues.
	In a way suppored by GSL
*/

void MathExtraPysph::eigen_decomposition_gsl(double A[3][3], double V[3][3], double *d)
{
	
	double data[] = {
			A[0][0], A[0][1], A[0][2],\
			A[1][0], A[1][1], A[1][2],\
			A[2][0], A[2][1], A[2][2]
		};
		

	int n = 3;
	gsl_matrix_view m = gsl_matrix_view_array(data, n, n);
	gsl_vector *eval = gsl_vector_alloc(n);
	gsl_matrix *evec = gsl_matrix_alloc(n,n);

	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);

	gsl_eigen_symmv(&m.matrix, eval, evec, w);

	gsl_eigen_symmv_free(w);

	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

	for (int i = 0; i < n; i++)
	{
		d[i] = gsl_vector_get(eval, i);
		gsl_vector_view evec_i = gsl_matrix_column(evec, i);
		
		V[0][i] = gsl_vector_get(&evec_i.vector,0);
		V[1][i] = gsl_vector_get(&evec_i.vector,1);
		V[2][i] = gsl_vector_get(&evec_i.vector,2);
		
	}

	gsl_vector_free(eval);
	gsl_matrix_free(evec);
}

void MathExtraPysph::eigen_decomposition_gsl_3D(double A[3][3], double V[3][3], double *d)
{
	
	double data[] = {
			A[0][0], A[0][1], A[0][2],\
			A[1][0], A[1][1], A[1][2],\
			A[2][0], A[2][1], A[2][2]
		};
		

	int n = 3;
	gsl_matrix_view m = gsl_matrix_view_array(data, n, n);
	gsl_vector *eval = gsl_vector_alloc(n);
	gsl_matrix *evec = gsl_matrix_alloc(n,n);

	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);

	gsl_eigen_symmv(&m.matrix, eval, evec, w);

	gsl_eigen_symmv_free(w);

	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

	for (int i = 0; i < n; i++)
	{
		d[i] = gsl_vector_get(eval, i);
		gsl_vector_view evec_i = gsl_matrix_column(evec, i);
		
		V[0][i] = gsl_vector_get(&evec_i.vector,0);
		V[1][i] = gsl_vector_get(&evec_i.vector,1);
		V[2][i] = gsl_vector_get(&evec_i.vector,2);
		
	}

	gsl_vector_free(eval);
	gsl_matrix_free(evec);
}

/*
	Get eigenvalues and eigenvectors of matrix A.
	V is output eigenvectors and d are the eigenvalues.
	In a way suppored by GSL
*/

void MathExtraPysph::eigen_decomposition_gsl_2D(double A[2][2], double V[2][2], double *d)
{
	
	double data[] = {
			A[0][0], A[0][1],\
			A[1][0], A[1][1]
		};
		

	int n = 2;
	gsl_matrix_view m = gsl_matrix_view_array(data, n, n);
	gsl_vector *eval = gsl_vector_alloc(n);
	gsl_matrix *evec = gsl_matrix_alloc(n,n);

	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(n);

	gsl_eigen_symmv(&m.matrix, eval, evec, w);

	gsl_eigen_symmv_free(w);

	gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

	for (int i = 0; i < n; i++)
	{
		d[i] = gsl_vector_get(eval, i);
		gsl_vector_view evec_i = gsl_matrix_column(evec, i);
		
		V[0][i] = gsl_vector_get(&evec_i.vector,0);
		V[1][i] = gsl_vector_get(&evec_i.vector,1);
		
	}

	gsl_vector_free(eval);
	gsl_matrix_free(evec);
}

double MathExtraPysph::hypot2(double x, double y)
{
	return (sqrt(x*x + y*y));
}

void MathExtraPysph::bubbleSort(double arr[], int n, int inc) {

      bool swapped = true;
      int j = 0;
      int tmp;
	  if(inc)
	  {
		  while (swapped) {
				swapped = false;
				j++;
				for (int i = 0; i < n - j; i++) {
					  if (arr[i] > arr[i + 1]) {
							tmp = arr[i];
							arr[i] = arr[i + 1];
							arr[i + 1] = tmp;
							swapped = true;
					  }
				}
		  }
	  }
	  else {
		  while (swapped) {
				swapped = false;
				j++;
				for (int i = 0; i < n - j; i++) {
					  if (arr[i] < arr[i + 1]) {
							tmp = arr[i];
							arr[i] = arr[i + 1];
							arr[i + 1] = tmp;
							swapped = true;
					  }
				}
		  }
	  }
}

/*
  	Solve A X = b
	6 x 6 matrix
	In a way suppored by GSL
*/

void MathExtraPysph::solve6By6_gsl(double A[6][6], double my_b[6], double *my_x)
{
	
	double a_data[] = {
			A[0][0], A[0][1], A[0][2],A[0][3], A[0][4], A[0][5],\
			A[1][0], A[1][1], A[1][2],A[1][3], A[1][4], A[1][5],\
			A[2][0], A[2][1], A[2][2],A[2][3], A[2][4], A[2][5],\
			A[3][0], A[3][1], A[3][2],A[3][3], A[3][4], A[3][5],\
			A[4][0], A[4][1], A[4][2],A[4][3], A[4][4], A[4][5],\
			A[5][0], A[5][1], A[5][2],A[5][3], A[5][4], A[5][5]
		};
		
	double b_data[] = {
			my_b[0], my_b[1], my_b[2], my_b[3], my_b[4], my_b[5]
		};

	gsl_matrix_view m = gsl_matrix_view_array(a_data, 6, 6);
	gsl_vector_view b = gsl_vector_view_array(b_data, 6);

	gsl_vector *x = gsl_vector_alloc(6);

	int s;

	gsl_permutation *p = gsl_permutation_alloc(6);
	gsl_linalg_LU_decomp(&m.matrix, p, &s);
	gsl_linalg_LU_solve(&m.matrix, p, &b.vector, x);

	for (int i = 0; i < 6; i++)
	{
		my_x[i] = gsl_vector_get(x, i);
	}

	gsl_permutation_free(p);
	gsl_vector_free(x);
}

#endif
