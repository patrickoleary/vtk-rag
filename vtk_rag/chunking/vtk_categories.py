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

# VTK output methods to check for extracting output datatypes
# Priority order: typed methods first, then generic
OUTPUT_DATATYPE_METHODS = [
    "GetOutput",                 # Common typed convenience method
    "GetOutputDataObject",       # Generic vtkDataObject* method
    "GetPolyDataOutput",         # Typed getter for vtkPolyData
    "GetUnstructuredGridOutput", # Typed getter for vtkUnstructuredGrid
    "GetStructuredGridOutput",   # Typed getter for vtkStructuredGrid
    "GetRectilinearGridOutput",  # Typed getter for vtkRectilinearGrid
    "GetStructuredPointsOutput", # Typed getter for vtkStructuredPoints
    "GetImageDataOutput",        # Typed getter for vtkImageData
    "GetHyperTreeGridOutput",    # Typed getter for vtkHyperTreeGrid
    "GetGraphOutput",            # Typed getter for vtkGraph
    "GetMoleculeOutput",         # Typed getter for vtkMolecule
    "GetTableOutput",            # Typed getter for vtkTable
]

# VTK input methods to check for extracting input datatypes
# Priority order: SetInputConnection (pipeline) > SetInputData (static data)
INPUT_DATATYPE_METHODS = [
    "SetInputConnection",  # Preferred: pipeline connection (takes vtkAlgorithmOutput)
    "SetInputData",        # Static data (gives us actual datatype)
    "SetInputDataObject",  # Generic multi-port version
    "AddInputConnection",  # Multi-input filters (e.g., vtkAppendPolyData)
    "AddInputData",        # Multi-input static data
    "AddInputDataObject",  # Multi-input generic version
]

# Boilerplate methods to skip when creating standalone method chunks
# These are Python dunder methods and VTK introspection/infrastructure methods
BOILERPLATE_METHODS = {
    # Python dunder methods (common across most VTK classes)
    '__init__', '__new__', '__repr__', '__str__', '__delattr__',
    '__getattribute__', '__setattr__', '__buffer__', '__release_buffer__',
    '__call__', '__rrshift__', '__rshift__', '__hash__',
    # Python dunder methods (collection types)
    '__delitem__', '__getitem__', '__len__', '__setitem__', '__iter__', '__next__',
    # Python dunder methods (comparable types)
    '__eq__', '__ge__', '__gt__', '__le__', '__lt__', '__ne__',
    # Python dunder methods (numeric types)
    '__abs__', '__add__', '__and__', '__bool__', '__ceil__', '__divmod__',
    '__float__', '__floor__', '__floordiv__', '__format__', '__getnewargs__',
    '__index__', '__int__', '__invert__', '__lshift__', '__mod__', '__mul__',
    '__neg__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__',
    '__rdivmod__', '__rfloordiv__', '__rlshift__', '__rmod__', '__rmul__',
    '__ror__', '__round__', '__rpow__', '__rsub__', '__rtruediv__', '__rxor__',
    '__sizeof__', '__sub__', '__truediv__', '__trunc__', '__xor__',
    # VTK introspection/type-checking methods (vtkObjectBase)
    'GetNumberOfGenerationsFromBase', 'GetNumberOfGenerationsFromBaseType',
    'IsA', 'IsTypeOf', 'NewInstance', 'SafeDownCast',
    # VTK memory/reference management methods (vtkObjectBase)
    'FastDelete', 'GetAddressAsString', 'GetClassName', 'GetIsInMemkind',
    'GetReferenceCount', 'GetUsingMemkind', 'InitializeObjectBase', 'Register',
    'SetMemkindDirectory', 'SetReferenceCount', 'UnRegister', 'UsesGarbageCollector',
}
