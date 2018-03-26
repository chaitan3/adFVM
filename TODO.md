CFD SOLVER:
    check code numerics: second order interpolator fix (stabilization?)
                         second order boundary?
PERFORMANCE:
    modify adpy to support computing multiple solutions
    better kernel generation on gpu: adjoint flops & bw much lower than primal
    jacobi solver requires too many iterations (while loop?)
    use of vector aliasing?
    use of vectorized loads, stores and arithmetic
    llvm based codegen, c++ runtime
    implement gpu async scheduler
    global renumbering/mini partitions
POLISH:
    finite diff, element solvers
TESTING:
    add conservation check, timestep unit test
    add eno and inviscid cfd
    add viscous cfd
ALGORITHMIC IMPROVEMENTS
    support general unstructured meshes
    Look into implicit for laminar
    Look into wall model BC
    debug second order boundary overshoot
    gradU/gradT on boundary restriction? 
CHARLES
    better calcCvGrad
    central blending in euler flux
MESH DECOMPOSITION
    VALIDATE
    bluewaters support
    sliding support
HDF5:
    performance not optimal
    add decomposition support
FAR OFF IMPROVEMENTS
    LES modelling
    general ggi/ami code
    better support for explicit for incompressible cases
    dual consistency
    steady state solver: preconditioning, sparse grad theano?
    multi block structured code? for speedup on gpu
