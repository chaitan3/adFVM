#include<iostream>
#include<cstdlib>
#include<Python.h>
#include <numpy/arrayobject.h>

using namespace std;

/** Computation of the Lorenz equation right hand side. */
static inline void dlorenz(
    const double  beta, const double rho,   const double sigma,
    const double  x,    const double y,     const double z,
          double &dxdt,       double &dydt,       double &dzdt)
{
     dxdt = sigma*(y - x);
     dydt = x*(rho - z) - y;
     dzdt = x*y - beta*z;
}

/** Advance by \c dt using one step of 1st order Forward Euler. */
static void euler(
    const double dt, const double beta, const double rho, const double sigma,
    double &t, double &x, double &y, double &z)
{
    double dxdt, dydt, dzdt;
    dlorenz(beta, rho, sigma, x, y, z, dxdt, dydt, dzdt);
    x += dt*dxdt;
    y += dt*dydt;
    z += dt*dzdt;
    t += dt;
}

/**
 * Advance by \c dt using one step of 2nd order TVD Runge--Kutta.
 * This optimal scheme appears in Proposition 3.1 of Gottlieb and Shu 1998.
 */
static void tvd_rk2(
    const double dt, const double beta, const double rho, const double sigma,
    double &t, double &x, double &y, double &z)
{
    double dxdt, dydt, dzdt;

    dlorenz(beta, rho, sigma, x, y, z, dxdt, dydt, dzdt);
    double u1x = x + dt*dxdt;
    double u1y = y + dt*dydt;
    double u1z = z + dt*dzdt;

    dlorenz(beta, rho, sigma, u1x, u1y, u1z, dxdt, dydt, dzdt);
    x = (x + u1x + dt*dxdt)/2;
    y = (y + u1y + dt*dydt)/2;
    z = (z + u1z + dt*dzdt)/2;
    t += dt;
}

/**
 * Advance by \c dt using one step of 3rd order TVD Runge--Kutta.
 * The optimal scheme appears in Proposition 3.2 of Gottlieb and Shu 1998.
 */
static void tvd_rk3(
    const double dt, const double beta, const double rho, const double sigma,
    double &t, double &x, double &y, double &z)
{
    double dxdt, dydt, dzdt;

    dlorenz(beta, rho, sigma, x, y, z, dxdt, dydt, dzdt);
    double u1x = x + dt*dxdt;
    double u1y = y + dt*dydt;
    double u1z = z + dt*dzdt;

    dlorenz(beta, rho, sigma, u1x, u1y, u1z, dxdt, dydt, dzdt);
    double u2x = (3*x + u1x + dt*dxdt)/4;
    double u2y = (3*y + u1y + dt*dydt)/4;
    double u2z = (3*z + u1z + dt*dzdt)/4;

    dlorenz(beta, rho, sigma, u2x, u2y, u2z, dxdt, dydt, dzdt);
    x = (x + 2*u2x + 2*dt*dxdt)/3;
    y = (y + 2*u2y + 2*dt*dydt)/3;
    z = (z + 2*u2z + 2*dt*dzdt)/3;
    t += dt;
}

/**
 * Advance by \c dt using one step of classical 4th order Runge--Kutta.
 */
static void std_rk4(
    const double dt, const double beta, const double rho, const double sigma,
    double &t, double &x, double &y, double &z)
{
    double dxdt, dydt, dzdt;

    dlorenz(beta, rho, sigma, x, y, z, dxdt, dydt, dzdt);
    double ux = x + dt*dxdt/2;
    double uy = y + dt*dydt/2;
    double uz = z + dt*dzdt/2;

    double dx = dt*dxdt/6;
    double dy = dt*dydt/6;
    double dz = dt*dzdt/6;

    dlorenz(beta, rho, sigma, ux, uy, uz, dxdt, dydt, dzdt);
    ux = x + dt*dxdt/2;
    uy = y + dt*dydt/2;
    uz = z + dt*dzdt/2;

    dx += dt*dxdt/3;
    dy += dt*dydt/3;
    dz += dt*dzdt/3;

    dlorenz(beta, rho, sigma, ux, uy, uz, dxdt, dydt, dzdt);
    ux = x + dt*dxdt;
    uy = y + dt*dydt;
    uz = z + dt*dzdt;

    dx += dt*dxdt/3;
    dy += dt*dydt/3;
    dz += dt*dzdt/3;

    dlorenz(beta, rho, sigma, ux, uy, uz, dxdt, dydt, dzdt);
    x += (dx + dt*dxdt/6);
    y += (dy + dt*dydt/6);
    z += (dz + dt*dzdt/6);
    t += dt;
}

static PyObject* lorenz(PyObject *dummy, PyObject *args) {
    double rho, sigma, beta;
    double dt, T;
    PyArrayObject* xi;

    PyArg_ParseTuple(args, "dddddO", &rho, &sigma, &beta, &dt, &T, &xi);

    double t = 0;
    //double x = 1, y = 0, z = 1;
    //x += (double)rand() / RAND_MAX;
    //y += (double)rand() / RAND_MAX;
    //z += (double)rand() / RAND_MAX;
    double *xid = (double *) xi -> data;
    double x = xid[0];
    double y = xid[1];
    double z = xid[2];
    int N = (int)(T/dt);

    npy_intp dims[2] = {N, 3};
    PyObject* output = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double* data = (double*)PyArray_DATA(output);

    for (int i = 0; i < N; i++) {
        std_rk4(dt, beta, rho, sigma, t, x, y, z);    
        data[3*i]=x;
        data[3*i+1]=y;
        data[3*i+2]=z;
    }
    
    return output;
}

#define J 8
#define K 9
#define JK (J*K)
#define Ns (JK + K)
inline int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

/** Computation of the Lorenz equation right hand side. */
static inline void mdlorenz(
    const double  Fx, const double hx,   const double hy, const double eps,
    const double *x,    
    double *dxdt)
{
    for (int k = 0; k < K; k++) {
        dxdt[k] = -x[mod(k-1,K)]*(x[mod(k-2,K)]-x[mod(k+1,K)])-x[k] + Fx;
        for (int j = 0; j < J; j++) {
            int jk = k*J+j;
            const double *y = &x[K];
            dxdt[k] += hx*y[jk]/J;
            dxdt[K+jk] = (-y[mod(jk+1, JK)]*(y[mod(jk+2,JK)]-y[mod(jk-1,JK)]) - y[jk] + hy*x[k])/eps;
        }
    }
}

/**
 * Advance by \c dt using one step of classical 4th order Runge--Kutta.
 */
static void mstd_rk4(
    const double dt, 
    const double  Fx, const double hx,   const double hy, const double eps,
    double &t, double *x)
{
    double dxdt[Ns];
    double ux[Ns];
    double dx[Ns];
    //printf("%lf %lf %lf %lf\n", x[1], x[2], x[17], x[20]);

    mdlorenz(Fx, hx, hy, eps, x, dxdt);
    for (int i=0; i < Ns; i++) {
        ux[i] = x[i] + dt*dxdt[i]/2;
        dx[i] = dt*dxdt[i]/6;
    }
    mdlorenz(Fx, hx, hy, eps, ux, dxdt);
    for (int i=0; i < Ns; i++) {
        ux[i] = x[i] + dt*dxdt[i]/2;
        dx[i] += dt*dxdt[i]/3;
    }
    mdlorenz(Fx, hx, hy, eps, ux, dxdt);
    for (int i=0; i < Ns; i++) {
        ux[i] = x[i] + dt*dxdt[i];
        dx[i] += dt*dxdt[i]/3;
    }
    mdlorenz(Fx, hx, hy, eps, ux, dxdt);
    for (int i=0; i < Ns; i++) {
        x[i] += dx[i] + dt*dxdt[i]/6;
    }
    t += dt;
}

static PyObject* mlorenz(PyObject *dummy, PyObject *args) {
    double Fx, hx, hy, eps;
    double dt, T;
    PyArrayObject* xi;

    PyArg_ParseTuple(args, "ddddddO", &Fx, &hx, &hy, &eps, &dt, &T, &xi);


    double *xid = (double *)xi -> data;
    double t = 0;
    double x[Ns];
    for (int i = 0; i < Ns; i++) {
        //x[i] = ((double)rand()) / RAND_MAX;
        x[i] = xid[i];
    }

    int N = (int)(T/dt);

    npy_intp dims[2] = {N, Ns};
    PyObject* output = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double* data = (double*)PyArray_DATA(output);

    for (int i = 0; i < N; i++) {
        mstd_rk4(dt, Fx, hx, hy, eps, t, x);
        memcpy(&data[Ns*i], x, Ns*sizeof(double));
    }
    
    return output;
}



static PyMethodDef mymethods[] = {
    { "lorenz", lorenz, METH_VARARGS, "calc lorenz"},
    { "mlorenz", mlorenz, METH_VARARGS, "calc mlorenz"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

PyMODINIT_FUNC
initclorenz(void)
{
    (void)Py_InitModule("clorenz", mymethods);
    import_array();
}
