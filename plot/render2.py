#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

fieldName = 'U'
limits = [1.25, 232.24, 463.]
dataType = 'POINTS'
#dataType = 'CELLS'

# create a new 'OpenFOAMReader'
#afoam = OpenFOAMReader(FileName='a.foam')
afoam = OpenFOAMReader(FileName='a.foam', CaseType='Decomposed Case')
afoam.MeshRegions = ['internalMesh']
afoam.CellArrays = ['T', 'U', 'p', 'rho', 'rhoE', 'rhoU']
#afoam = XDMFReader(FileNames=['les.xmf'])
#afoam.CellArrayStatus = [fieldName]
##afoam.GridStatus = ['U']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = [904, 585]

# get animation scene
# update animation scene based on data timesteps
animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()
#animationScene1.PlayMode = 'Snap To TimeSteps'


# get color transfer function/color map for 'U'
LUT = GetColorTransferFunction(fieldName)
LUT.RGBPoints = [limits[0], 0.231373, 0.298039, 0.752941, limits[1], 0.865003, 0.865003, 0.865003, limits[-1], 0.705882, 0.0156863, 0.14902]
LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'U'
PWF = GetOpacityTransferFunction(fieldName)
PWF.Points = [limits[0], 0.0, 0.5, 0.0, limits[-1], 1.0, 0.5, 0.0]
PWF.ScalarRangeInitialized = 1

# show data in view
afoamDisplay = Show(afoam, renderView1)
# trace defaults for the display properties.
afoamDisplay.ColorArrayName = [dataType, fieldName]
afoamDisplay.LookupTable = LUT
afoamDisplay.EdgeColor = [0.0, 0.0, 0.0]
afoamDisplay.ScalarOpacityUnitDistance = 0.001241735216909729

# reset view to fit data
renderView1.ResetCamera()

# show color bar/color legend
afoamDisplay.SetScalarBarVisibility(renderView1, True)

# create a new 'Slice'
slice1 = Slice(Input=afoam)
slice1.SliceType = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.010560549795627594, -0.07964942511171103, 0.004999999888241291]

# toggle 3D widget visibility (only when running from the GUI)
Hide3DWidgets(proxy=slice1)

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView1)
# trace defaults for the display properties.
slice1Display.ColorArrayName = [dataType, fieldName]
slice1Display.LookupTable = LUT
slice1Display.EdgeColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(afoam, renderView1)

## set scalar coloring
#ColorBy(slice1Display, ('CELLS', fieldName))
#
## rescale color and/or opacity maps used to include current data range
#slice1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# current camera placement for renderView1
renderView1.CameraPosition = [0.010560549795627594, -0.07964942511171103, 0.4138181725874819]
renderView1.CameraFocalPoint = [0.010560549795627594, -0.07964942511171103, 0.004999999888241291]
renderView1.CameraParallelScale = 0.1549163171639415

index = 0
folder = 'anim'

fileName = '{0}/{1}_{2:04d}.png'.format(folder, fieldName, index)
SaveScreenshot(fileName, magnification=1, quality=100, view=renderView1)

print renderView1.ViewTime, animationScene1.EndTime
while renderView1.ViewTime != animationScene1.EndTime:
    animationScene1.GoToNext()
    print renderView1.ViewTime, animationScene1.EndTime
    index += 1
    # save screenshot

    fileName = '{0}/{1}_{2:04d}.png'.format(folder, fieldName, index)
    SaveScreenshot(fileName, magnification=1, quality=100, view=renderView1)

#### saving camera placements for all active views

# current camera placement for renderView1
#renderView1.CameraPosition = [0.010560549795627594, -0.07964942511171103, 0.4138181725874819]
#renderView1.CameraFocalPoint = [0.010560549795627594, -0.07964942511171103, 0.004999999888241291]
#renderView1.CameraParallelScale = 0.1549163171639415

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
