#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# find source
afoam = FindSource('a.foam')

# create a new 'Contour'
contour1 = Contour(Input=afoam)
contour1.PointMergeMethod = 'Uniform Binning'

# set active source
SetActiveSource(afoam)

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')

# set active source
SetActiveSource(contour1)

# Properties modified on contour1
contour1.ContourBy = ['POINTS', 'M_2norm']
contour1.Isosurfaces = [1000000.0, 1291549.6650148828, 1668100.537200059, 2154434.6900318824, 2782559.402207126, 3593813.6638046256, 4641588.833612773, 5994842.503189409, 7742636.8268112615, 10000000.0]

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1229, 860]

# show data in view
contour1Display = Show(contour1, renderView1)
# trace defaults for the display properties.
contour1Display.ColorArrayName = ['CELLS', 'p']
contour1Display.LookupTable = pLUT
contour1Display.EdgeColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(afoam, renderView1)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# reset view to fit data bounds
renderView1.ResetCamera(-5.39679895155e-05, 0.0408239811659, -0.0601080879569, 0.0100035862997, 0.0, 0.00999999977648)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [0.10088236281787376, -0.07190762696023895, 0.05946474829354532]
renderView1.CameraFocalPoint = [0.020385006588185196, -0.025052250828593966, 0.004999999888241293]
renderView1.CameraViewUp = [0.4934918367798123, 0.8695494595381434, 0.018696107846543908]
renderView1.CameraParallelScale = 0.04088598045096883

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
