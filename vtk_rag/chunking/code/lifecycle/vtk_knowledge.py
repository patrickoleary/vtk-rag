"""VTK class categories and method patterns for lifecycle tracking.

Used by LifecycleVisitor, LifecycleBuilder, and LifecycleGrouper to:
- Identify actor-like props that need mapper/property tracking
- Track property setter/getter relationships
- Recognize chainable method patterns
"""

# -------------------------------------------------------------------------
# Actor-Like Props
#
# Displayable props that don't end with "Actor" but are added to renderer.
# Used to identify classes that need mapper/property relationship tracking.
#
# Example: vtkVolume uses vtkVolumeMapper, vtkImageSlice uses vtkImageSliceMapper
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Property Mappings
#
# Maps property classes to their parent actor/prop classes.
# Used by LifecycleBuilder to infer property class for chained usage.
#
# Example: actor.GetProperty().SetColor() → infer vtkProperty for vtkActor
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Property Setters
#
# Methods that assign property objects to actors/props.
# Used by LifecycleVisitor to track property→parent relationships.
#
# Example: actor.SetProperty(prop) → link prop to actor
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Property Getters
#
# Methods that return property objects (subset of CHAINABLE_GETTERS).
# Used by LifecycleVisitor to track chained property usage.
#
# Example: actor.GetProperty().SetColor() → mark actor has chained properties
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# Chainable Getters
#
# Methods that return sub-objects commonly used in method chaining.
# Used by LifecycleVisitor to attribute chained calls to parent variable.
#
# Example: actor.GetProperty().SetColor() → add statement to actor's lifecycle
# Example: renderer.GetActiveCamera().SetPosition() → add to renderer's lifecycle
# -------------------------------------------------------------------------
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
