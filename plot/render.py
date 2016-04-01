try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

a_foam = OpenFOAMReader( FileName='a.foam' )
field = 'U'
limits = [0, 400]

AnimationScene1 = GetAnimationScene()
a_foam.CellArrays = ['T', 'U', 'p', 'rhoa']
a_foam.LagrangianArrays = []
a_foam.MeshRegions = ['internalMesh']
a_foam.PointArrays = []

AnimationScene1.EndTime = 100.0
AnimationScene1.PlayMode = 'Snap To TimeSteps'

RenderView1 = GetRenderView()
RenderView1.ViewSize = [1280, 800]
a1_p_PVLookupTable = GetLookupTableForArray( "p", 1, RGBPoints=[89634.2421875, 0.23, 0.299, 0.754, 173761.6875, 0.706, 0.016, 0.15], VectorMode='Magnitude', NanColor=[0.25, 0.0, 0.0], ColorSpace='Diverging', ScalarRangeInitialized=1.0, AllowDuplicateScalars=1 )

a1_p_PiecewiseFunction = CreatePiecewiseFunction( Points=[0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0] )

DataRepresentation1 = Show()
DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation1.SelectionPointFieldDataArrayName = 'p'
DataRepresentation1.SelectionCellFieldDataArrayName = 'p'
DataRepresentation1.ScalarOpacityFunction = a1_p_PiecewiseFunction
DataRepresentation1.ColorArrayName = ('POINT_DATA', 'p')
DataRepresentation1.ScalarOpacityUnitDistance = 0.0072617116008152965
DataRepresentation1.LookupTable = a1_p_PVLookupTable
DataRepresentation1.ExtractedBlockIndex = 1
DataRepresentation1.ScaleFactor = 0.022112110257148744

a_foam.CaseType = 'Decomposed Case'

RenderView1.CenterOfRotation = [0.010560549795627594, -0.07964942511171103, 0.004999999888241291]

a1_p_PVLookupTable.ScalarOpacityFunction = a1_p_PiecewiseFunction

Slice1 = Slice( SliceType="Plane" )

a3_U_PVLookupTable = GetLookupTableForArray( field, 3, RGBPoints=[limits[0], 0.23, 0.299, 0.754, limits[1], 0.706, 0.016, 0.15], VectorMode='Magnitude', NanColor=[0.25, 0.0, 0.0], ColorSpace='Diverging', ScalarRangeInitialized=1.0, AllowDuplicateScalars=1 )
a3_U_PiecewiseFunction = CreatePiecewiseFunction( Points=[limits[0], 0.0, 0.5, 0.0, limits[1], 1.0, 0.5, 0.0] )

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('time', nargs='+', type=float)
user = parser.parse_args()
import time
start = time.time()

times = [float(x) for x in user.time]
startTime = times[0]
endTime = times[-1]
plotTime = startTime

AnimationScene1.AnimationTime = plotTime
AnimationScene1.EndTime = endTime
AnimationScene1.StartTime = startTime

RenderView1.CacheKey = 1.0
RenderView1.UseCache = 0

DataRepresentation1.ScalarOpacityFunction = a3_U_PiecewiseFunction
DataRepresentation1.ColorArrayName = ('CELL_DATA', 'rhoa')
DataRepresentation1.LookupTable = a3_U_PVLookupTable
DataRepresentation1.ColorAttributeType = 'CELL_DATA'

a3_U_PVLookupTable.ScalarOpacityFunction = a3_U_PiecewiseFunction

Slice1.SliceOffsetValues = [0.0]
Slice1.SliceType.Origin = [0.010560549795627594, -0.07964942511171103, 0.004999999888241291]
Slice1.SliceType = "Plane"

active_objects.source.SMProxy.InvokeEvent('UserEvent', 'ShowWidget')


DataRepresentation2 = Show()
DataRepresentation2.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation2.ColorAttributeType = 'CELL_DATA'
DataRepresentation2.SelectionPointFieldDataArrayName = 'p'
DataRepresentation2.SelectionCellFieldDataArrayName = 'p'
DataRepresentation2.ColorArrayName = ('POINT_DATA', field)
DataRepresentation2.LookupTable = a3_U_PVLookupTable
DataRepresentation2.ScaleFactor = 0.022112110257148744

DataRepresentation1.Visibility = 0

Slice1.SliceType.Normal = [0.0, 0.0, 1.0]

active_objects.source.SMProxy.InvokeEvent('UserEvent', 'HideWidget')

ScalarBarWidgetRepresentation1 = CreateScalarBar( ComponentTitle='Magnitude', Position=[0.6947232472324723, 0.3232558139534883], Title=field, Position2=[0.13, 0.49999999999999994], Enabled=1, LabelFontSize=12, LabelColor=[1.0, 1.0, 1.0], LookupTable=a3_U_PVLookupTable, TitleFontSize=12, TitleColor=[1.0, 1.0, 1.0] )
GetRenderView().Representations.append(ScalarBarWidgetRepresentation1)

RenderView1.CameraParallelScale = 0.1549163171639415
RenderView1.CameraPosition = [0.07017243669604589, -0.038286075017543095, 0.2842283126395837]
RenderView1.CameraClippingRange = [0.27643602962382896, 0.2834167374426126]
RenderView1.CameraFocalPoint = [0.07017243669604589, -0.038286075017543095, 0.004999999888241291]

for index, currTime in enumerate(times):
    RenderView1.ViewTime = currTime
    print 'saving', currTime, 'at', index
    WriteImage('anim/'+field+'mag.{0:04d}.png'.format(index))
    end = time.time()
    print end-start
