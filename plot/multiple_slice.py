#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()
# find source
afoam = FindSource('a.foam')

sliceOrigins = [
    [0.010560549795627594, -0.07964942511171103, 0.004999999888241291],
    [0.010560549795627594, -0.07964942511171103, 0.004999999888241291],
    [0.010560549795627594, -0.07964942511171103, 0.004999999888241291],
    [0.010560549795627594, -0.07964942511171103, 0.004999999888241291]
]
sliceNormals = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]
]

for origin, normal in zip(sliceOrigins, sliceNormals):
    # create a new 'Slice'
    slice1 = Slice(Input=afoam)
    slice1.SliceType = 'Plane'
    slice1.SliceOffsetValues = [0.0]

    # init the 'Plane' selected for 'SliceType'
    slice1.SliceType.Origin = origin

    # Properties modified on slice1.SliceType
    slice1.SliceType.Normal = normal

    # get active view
    renderView1 = GetActiveViewOrCreate('RenderView')
    # uncomment following to set a specific view size
    # renderView1.ViewSize = [1229, 860]

    # get color transfer function/color map for 'p'
    pLUT = GetColorTransferFunction('p')

    # show data in view
    slice1Display = Show(slice1, renderView1)
    # trace defaults for the display properties.
    slice1Display.ColorArrayName = ['POINTS', 'p']
    slice1Display.LookupTable = pLUT
    slice1Display.EdgeColor = [0.0, 0.0, 0.0]

    # hide data in view
    Hide(afoam, renderView1)

    # show color bar/color legend
    slice1Display.SetScalarBarVisibility(renderView1, True)

    # get opacity transfer function/opacity map for 'p'
    pPWF = GetOpacityTransferFunction('p')

    # toggle 3D widget visibility (only when running from the GUI)
    Hide3DWidgets(proxy=slice1)

# current camera placement for renderView1
renderView1.CameraPosition = [-0.12672413097260615, 0.023876386262029467, 0.15891173981798198]
renderView1.CameraFocalPoint = [0.010560549795627589, -0.07964942511171101, 0.004999999888241275]
renderView1.CameraViewUp = [0.249677885401626, 0.8918496551987318, -0.37718052185031475]
renderView1.CameraParallelScale = 0.1549163171639415

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
