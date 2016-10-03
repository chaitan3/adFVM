import numpy as np

def localDerivative(evaluate, initPoint, bounds, algorithm='SLSQP', maxiter=100):
    from scipy.optimize import minimize  
    result = minimize(evaluate, initPoint, bounds=bounds, method=algorithm, jac=True, options={'maxiter':maxiter})
    return result

def designOfExperiment(evaluate, bounds, nPoints, method='random'):
    from pyDOE import lhs, ccdesign
    bounds = np.array(bounds)
    nParam = bounds.shape[0]
    results = []
    if method == 'random':
        experiment = lhs(nParam, samples=nPoints, criterion='centermaximin')
    else:
        assert nParam > 1
        experiment = ccdesign(nParam, face='inscribed')
        experiment = (experiment + 1)/2
    experiment = experiment * (bounds[:,1]-bounds[:,0]) + bounds[:,0]
    for point in experiment:
        results.append(evaluate(point))
    return results

if __name__ == '__main__':
    f = lambda x: [(x-1)**2, 2*(x-1)]
    print designOfExperiment(f, [(-10, 10)], 10, method='random')
    print localDerivative(f, [0.], [(-10, 10)])
