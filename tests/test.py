def check(self, res, ref, maxThres=1e-7, sumThres=1e-4):
    self.assertAlmostEqual(0, np.abs(res-ref).max(), delta=maxThres)
    self.assertAlmostEqual(0, np.abs(res-ref).sum(), delta=sumThres)

def checkSum(self, res, ref, relThres=1e-4):
    vols = self.mesh.volumes
    if len(res.shape) == 3:
        vols = vols.flatten().reshape((-1,1,1))
    diff = np.abs(res-ref)*vols
    rel = diff.sum()/(ref*vols).sum()
    self.assertAlmostEqual(0, rel, delta=relThres)


