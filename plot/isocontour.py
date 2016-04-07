#### import the simple module from the paraview
from paraview.simple import *
import numpy as np
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

fieldName = 'rhoa'
fieldRange = [-10, -5, -1]

# create a new 'OpenFOAMReader'
afoam = OpenFOAMReader(FileName='a.foam')
afoam.MeshRegions = ['internalMesh']
afoam.CellArrays = [fieldName]

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
renderView1.ViewSize = [1220, 860]


animationScene1 = GetAnimationScene()
animationScene1.UpdateAnimationUsingDataTimeSteps()

# show data in view
afoamDisplay = Show(afoam, renderView1)
# trace defaults for the display properties.
afoamDisplay.ColorArrayName = [None, '']
afoamDisplay.EdgeColor = [0.0, 0.0, 0.0]
#afoamDisplay.ScalarOpacityUnitDistance = 0.001241735216909729

# reset view to fit data
renderView1.ResetCamera()

# set scalar coloring
ColorBy(afoamDisplay, ('CELLS', fieldName))

# rescale color and/or opacity maps used to include current data range
afoamDisplay.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
afoamDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for fieldName
fieldLUT = GetColorTransferFunction(fieldName)
fieldLUT.RGBPoints = [fieldRange[0], 0.231373, 0.298039, 0.752941, fieldRange[1], 0.865003, 0.865003, 0.865003, fieldRange[2], 0.705882, 0.0156863, 0.14902]
fieldLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for fieldName
fieldPWF = GetOpacityTransferFunction(fieldName)
fieldPWF.Points = [fieldRange[0], 0.0, 0.5, 0.0, fieldRange[2], 1.0, 0.5, 0.0]
fieldPWF.ScalarRangeInitialized = 1

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=afoam)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1)
# trace defaults for the display properties.
extractSurface1Display.ColorArrayName = ['CELLS', fieldName]
extractSurface1Display.LookupTable = fieldLUT
extractSurface1Display.EdgeColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(afoam, renderView1)

# show color bar/color legend
extractSurface1Display.SetScalarBarVisibility(renderView1, True)

# turn off scalar coloring
ColorBy(extractSurface1Display, None)

# Properties modified on extractSurface1Display
extractSurface1Display.Opacity = 0.3

# change solid color
extractSurface1Display.DiffuseColor = [0.32941176470588235, 1.0, 0.9568627450980393]

# set active source
SetActiveSource(afoam)

# create a new 'Contour'
contour1 = Contour(Input=afoam)

# Properties modified on contour1
contour1.ContourBy = ['POINTS', fieldName]
#contour1.Isosurfaces = np.linspace(fieldRange[0], fieldRange[2], 10).tolist()
#contour1.Isosurfaces = np.linspace(fieldRange[0], fieldRange[2], 10).tolist() 
contour1.Isosurfaces = np.linspace(fieldRange[0], fieldRange[2], 10).tolist()
contour1.PointMergeMethod = 'Uniform Binning'

# show data in view
contour1Display = Show(contour1, renderView1)
# trace defaults for the display properties.
contour1Display.ColorArrayName = ['CELLS', fieldName]
contour1Display.LookupTable = fieldLUT
contour1Display.EdgeColor = [0.0, 0.0, 0.0]

# hide data in view
Hide(afoam, renderView1)

# set scalar coloring
ColorBy(contour1Display, ('POINTS', fieldName))

# rescale color and/or opacity maps used to include current data range
contour1Display.RescaleTransferFunctionToDataRange(True)

# show color bar/color legend
contour1Display.SetScalarBarVisibility(renderView1, True)

# Rescale transfer function
fieldLUT.RescaleTransferFunction(fieldRange[0], fieldRange[2])

# Rescale transfer function
fieldPWF.RescaleTransferFunction(fieldRange[0], fieldRange[2])

# current camera placement for renderView1
renderView1.CameraPosition = [0.15652329443511975, -0.03445140075875224, 0.30691556175293677]
renderView1.CameraFocalPoint = [0.00595836116733855, -0.061600928039124536, 0.005673611448174642]
renderView1.CameraViewUp = [-0.03575294422593495, 0.9967662222716278, -0.07196405434329668]
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
