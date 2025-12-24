"""VTK class categories and method patterns for semantic chunking.

This module contains domain knowledge about VTK class relationships,
property patterns, and method chaining conventions used to improve
the quality of semantic code chunking.
"""

# Self-contained actors/props that manage their own internal mapper(s)
# You don't normally set a mapper on these classes
# Note: vtkCameraActor and vtkLightActor are NOT here - they're rendering infrastructure
SELF_CONTAINED_ACTORS = {
    # 3D props with internal mappers
    'vtkAxesActor',
    'vtkAnnotatedCubeActor',
    'vtkCubeAxesActor',
    'vtkSkybox',
    'vtkImageActor',  # Legacy; prefer vtkImageSlice + mapper for new code
    'vtkTextActor3D',
    'vtkBillboardTextActor3D',
    'vtkContextActor',       # Context2D rendering
    'vtkResliceCursorActor', # Reslice cursor for MPR
    # Axis actors (self-contained axis rendering)
    'vtkAxisActor',
    'vtkAxisActor2D',
    'vtkCubeAxesActor2D',
    'vtkPolarAxesActor',
    'vtkPolarAxesActor2D',
    'vtkGridAxesActor2D',
    'vtkGridAxesActor3D',
    'vtkGridAxesPlaneActor2D',
    # 2D / overlay actors (Actor2D family)
    'vtkTextActor',
    'vtkScalarBarActor',
    'vtkLegendBoxActor',
    'vtkLegendScaleActor',
    'vtkCornerAnnotation',
    'vtkCaptionActor2D',
    'vtkLeaderActor2D',
    # Classic chart actors (self-contained plotting)
    'vtkXYPlotActor',
    'vtkParallelCoordinatesActor',
    'vtkBarChartActor',
    'vtkPieChartActor',
    'vtkSpiderPlotActor',
    # Widgets (contain actors but aren't actors themselves)
    'vtkOrientationMarkerWidget',
}

# Displayable props that don't end with "Actor" but are added to renderer
# These are renderable objects added via AddViewProp/AddActor/AddVolume
# Note: Self-contained actors (vtkSkybox, vtkCornerAnnotation) are in SELF_CONTAINED_ACTORS
ACTOR_LIKE_PROPS = {
    # Volume rendering
    "vtkVolume",         # Volume rendering prop (uses vtkVolumeMapper)
    "vtkMultiVolume",    # Multi-volume rendering container
    # Image slices
    "vtkImageSlice",     # Modern 2D image slice prop (uses vtkImageSliceMapper)
    "vtkImageStack",     # Container for multiple vtkImageSlice objects
    # Followers (billboard-style props that face camera)
    "vtkFollower",       # Billboard actor that faces camera
    "vtkAxisFollower",   # Axis label follower
    "vtkProp3DFollower", # Generic 3D prop follower
    "vtkProp3DAxisFollower",  # Axis-specific 3D follower
    "vtkFlagpoleLabel",  # Flagpole-style label
    # Assemblies and containers
    "vtkAssembly",       # Hierarchical assembly of props
    "vtkPropAssembly",   # Assembly of props with transform
    "vtkLODProp3D",      # Level-of-detail prop
    # Axes and grids
    "vtkAxesGrid",       # Axes grid prop
    # VR/Avatar
    "vtkAvatar",         # VR avatar prop
}

# Property to parent class mappings for inference
PROPERTY_MAPPINGS = {
    'vtkProperty': 'vtkActor',              # 3D actor surface properties
    'vtkOpenGLProperty': 'vtkActor',        # OpenGL subclass of vtkProperty
    'vtkShaderProperty': 'vtkActor',        # Shader customization
    'vtkOpenGLShaderProperty': 'vtkActor',  # OpenGL subclass of vtkShaderProperty
    'vtkVolumeProperty': 'vtkVolume',       # Volume rendering properties
    'vtkImageProperty': 'vtkImageSlice',    # Image slice properties
    'vtkProperty2D': 'vtkActor2D',          # 2D actor properties
    'vtkTextProperty': 'vtkTextActor',      # Text rendering properties
}

# VTK setter methods that assign property objects to actors/props
# Used to track explicit property relationships: actor.SetProperty(prop)
PROPERTY_SETTERS = {
    # Core actor/prop properties
    'SetProperty',           # vtkActor, vtkActor2D, vtkVolume, vtkImageSlice
    'SetBackfaceProperty',   # vtkActor
    'SetShaderProperty',     # Shader properties
    'SetImageProperty',      # vtkImageSlice
    # Text properties
    'SetTextProperty',       # vtkTextActor, vtkCornerAnnotation
    'SetTitleTextProperty',  # vtkScalarBarActor, cube/axis actors
    'SetLabelTextProperty',  # vtkScalarBarActor
    'SetCaptionTextProperty', # vtkCaptionActor2D
    'SetAxisLabelTextProperty', 'SetAxisTitleTextProperty',
    'SetAxesTextProperty', 'SetDefaultTextProperty',
    'SetEdgeLabelTextProperty', 'SetVertexLabelTextProperty',
    # Widget handle/selection properties
    'SetHandleProperty', 'SetSelectedHandleProperty',
    'SetSelectedProperty', 'SetHoveringProperty', 'SetSelectingProperty',
    # Widget line/frame properties
    'SetLineProperty', 'SetSelectedLineProperty', 'SetFrameProperty',
    # Widget geometry properties
    'SetPlaneProperty',
    # Environment/texture properties
    'SetEnvironmentTextureProperty',
}

# VTK boolean property toggle suffixes
# Boolean properties use *On/*Off as setters and Get*/Is* as getters
# e.g., VisibilityOn(), VisibilityOff(), GetVisibility(), IsVisible()
BOOLEAN_ON_SUFFIXES = {'On'}   # actor.VisibilityOn()
BOOLEAN_OFF_SUFFIXES = {'Off'} # actor.VisibilityOff()

# Common boolean property names (the base name without On/Off suffix)
# Used to identify boolean property patterns in method calls
BOOLEAN_PROPERTIES = {
    # Visibility and picking (vtkProp)
    'Visibility', 'Pickable', 'Dragable', 'UseBounds', 'PickingManaged',
    # Debug and warnings (vtkObject)
    'Debug', 'GlobalWarningDisplay',
    # Pipeline (vtkAlgorithm)
    'AbortExecute', 'ReleaseDataFlag', 'GlobalReleaseDataFlag',
    # Widgets
    'Enabled', 'KeyPressActivation', 'ProcessEvents', 'ManagesCursor', 'Selectable',
    # Rendering
    'NeedToRender', 'ScalarVisibility',
    # I/O
    'WriteToOutputString', 'ReadFromInputString', 'EncodeAppendedData',
}

# VTK getter methods that return property objects (subset of CHAINABLE_GETTERS)
# Used to track inline property usage: actor.GetProperty().SetColor(...)
PROPERTY_GETTERS = {
    # Core actor/prop properties
    'GetProperty', 'GetVolumeProperty', 'GetImageProperty', 'GetProperty2D',
    'GetBackfaceProperty', 'GetShaderProperty',
    # Text properties
    'GetTextProperty', 'GetLabelTextProperty', 'GetTitleTextProperty',
    'GetCaptionTextProperty', 'GetLabelProperty',
    # Widget handle/selection properties
    'GetHandleProperty', 'GetSelectedHandleProperty',
    'GetSelectedProperty', 'GetActiveProperty',
    # Widget outline/line properties
    'GetOutlineProperty', 'GetSelectedOutlineProperty',
    'GetLineProperty', 'GetSelectedLineProperty',
    'GetBorderProperty', 'GetEdgesProperty',
    # Widget geometry properties
    'GetPlaneProperty', 'GetSelectedPlaneProperty',
    'GetNormalProperty', 'GetSelectedNormalProperty',
    'GetFaceProperty', 'GetSelectedFaceProperty',
    'GetAxisProperty', 'GetSelectedAxisProperty',
    'GetSphereProperty', 'GetSelectedSphereProperty',
    'GetSliderProperty', 'GetTubeProperty',
}

# VTK getter methods that return sub-objects commonly used in method chaining
# e.g., actor.GetProperty().SetColor(...), renderer.GetActiveCamera().SetPosition(...)
CHAINABLE_GETTERS = {
    # Scene props - properties and mappers
    *PROPERTY_GETTERS,
    'GetMapper',
    # Mappers - lookup tables
    'GetLookupTable',
    # Cameras, renderers, windows, interactors
    'GetActiveCamera', 'GetRenderWindow', 'GetInteractorStyle',
    'GetRenderer', 'GetCurrentRenderer', 'GetDefaultRenderer', 'GetInteractor',
    # Dataset attributes
    'GetPointData', 'GetCellData', 'GetFieldData',
    'GetScalars', 'GetVectors', 'GetTensors', 'GetArray',
    # Geometry and transforms
    'GetPoints', 'GetTransform', 'GetMatrix',
    # Pipeline information
    'GetInformation',
    # Composite datasets
    'GetBlock',
    # Specialized actors - axis properties
    'GetXAxesLinesProperty', 'GetYAxesLinesProperty', 'GetZAxesLinesProperty',
    'GetXAxisCaptionActor2D', 'GetYAxisCaptionActor2D', 'GetZAxisCaptionActor2D',
    'GetTextActor',
}

# Image mappers that conceptually "go with" vtkImageActor/vtkImageSlice
IMAGE_MAPPERS = {
    # 3D image slice mappers (for vtkImageActor/vtkImageSlice)
    'vtkImageMapper3D',           # Base class for 3D image mappers
    'vtkImageSliceMapper',        # Fast orthogonal/oblique slice mapper
    'vtkImageResliceMapper',      # Resampling slice mapper (thick/oblique MPR)
    'vtkOpenGLImageSliceMapper',  # OpenGL rendering backend for 3D
    # 2D screen-space image mapper (for vtkActor2D overlay/UI)
    'vtkImageMapper',             # 2D mapper for screen-space images
    'vtkOpenGLImageMapper',       # OpenGL rendering backend for 2D
}

