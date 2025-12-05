"""VTK query patterns for natural language query generation.

This module contains domain knowledge about VTK methods and classes
mapped to natural language queries for RAG retrieval.
"""

from dataclasses import dataclass


@dataclass
class QueryCategory:
    """Query category for natural language query generation."""
    name: str
    methods: list[str]
    classes: list[str]
    queries: list[str]
    method_to_query: dict[str, str]  # Method name -> query template


# Pattern templates for composite chunk types
PATTERN_QUERIES: dict[str, list[str]] = {
    "Rendering Infrastructure": [
        "How do I set up VTK rendering?",
        "VTK render window example",
        "How to create a VTK visualization window?",
        "Basic VTK rendering setup",
        "VTK renderer interactor example",
    ],
    "Visualization Pipeline": [
        "How do I create a VTK pipeline?",
        "VTK mapper actor example",
        "How to connect source to actor in VTK?",
        "VTK visualization pipeline",
        "How to map data to graphics in VTK?",
    ],
    "Actors": [
        "How do I create actors in VTK?",
        "VTK actor example",
        "How to add geometry to a scene?",
    ],
    "Mappers": [
        "How do I create mappers in VTK?",
        "VTK mapper example",
        "How to map data to graphics?",
    ],
    "Renderers": [
        "How do I create a renderer in VTK?",
        "VTK renderer example",
    ],
    "Render windows & interactors": [
        "How do I create a render window in VTK?",
        "VTK interactor example",
        "How to handle user interaction in VTK?",
    ],
}


# Query categories to extract from method calls
QUERY_CATEGORIES: dict[str, QueryCategory] = {
    "camera": QueryCategory(
        name="camera",
        methods=[
            "SetPosition", "SetFocalPoint", "SetViewUp", "SetViewAngle",
            "Elevation", "Azimuth", "Dolly", "Zoom", "SetClippingRange",
            "SetParallelProjection", "SetParallelScale", "Roll", "Yaw", "Pitch"
        ],
        classes=["vtkCamera"],
        queries=[
            "How do I position the camera in VTK?",
            "VTK camera setup",
            "How to set camera view angle in VTK?",
            "How to control camera in VTK?",
        ],
        method_to_query={
            "SetPosition": "set camera position",
            "SetFocalPoint": "set camera focal point",
            "SetViewUp": "set camera up vector",
            "SetViewAngle": "set camera field of view",
            "Elevation": "elevate camera",
            "Azimuth": "rotate camera azimuth",
            "Dolly": "dolly camera",
            "Zoom": "zoom camera",
            "SetClippingRange": "set camera clipping range",
        }
    ),
    "lighting": QueryCategory(
        name="lighting",
        methods=[
            "SetPosition", "SetFocalPoint", "SetColor", "PositionalOn",
            "AddLight", "SetLightTypeToHeadlight", "AutomaticLightCreationOff",
            "SetIntensity", "SetConeAngle", "SetAmbientColor", "SetDiffuseColor",
            "SetSpecularColor", "TwoSidedLightingOn", "TwoSidedLightingOff"
        ],
        classes=["vtkLight", "vtkLightKit", "vtkLightActor"],
        queries=[
            "How do I add lights in VTK?",
            "VTK lighting setup",
            "How to position lights in VTK?",
            "VTK light example",
        ],
        method_to_query={
            "AddLight": "add light to scene",
            "SetIntensity": "set light intensity",
            "SetColor": "set light color",
            "SetConeAngle": "set spotlight cone angle",
            "PositionalOn": "make light positional",
        }
    ),
    "background": QueryCategory(
        name="background",
        methods=[
            "SetBackground", "SetBackground2", "GradientBackgroundOn",
            "SetGradientBackground", "TexturedBackgroundOn"
        ],
        classes=[],
        queries=[
            "How do I set background color in VTK?",
            "VTK gradient background",
            "How to change renderer background?",
        ],
        method_to_query={
            "SetBackground": "set background color",
            "SetBackground2": "set gradient background color",
            "GradientBackgroundOn": "enable gradient background",
        }
    ),
    "window": QueryCategory(
        name="window",
        methods=[
            "SetSize", "SetWindowName", "SetPosition", "SetMultiSamples",
            "FullScreenOn", "BordersOn", "BordersOff", "SetAlphaBitPlanes"
        ],
        classes=[],
        queries=[
            "How do I set window size in VTK?",
            "VTK render window configuration",
            "How to set window title in VTK?",
        ],
        method_to_query={
            "SetSize": "set window size",
            "SetWindowName": "set window title",
            "SetMultiSamples": "set antialiasing samples",
            "FullScreenOn": "enable fullscreen",
        }
    ),
    "interaction": QueryCategory(
        name="interaction",
        methods=[
            "SetInteractorStyle", "AddObserver", "CreateRepeatingTimer",
            "CreateOneShotTimer", "SetDesiredUpdateRate", "SetStillUpdateRate"
        ],
        classes=["vtkInteractorStyle", "vtkInteractorStyleTrackballCamera",
                 "vtkInteractorStyleTrackballActor", "vtkInteractorStyleImage"],
        queries=[
            "How do I handle interaction in VTK?",
            "VTK interactor style example",
            "How to add callbacks in VTK?",
            "VTK mouse interaction",
        ],
        method_to_query={
            "SetInteractorStyle": "set interaction style",
            "AddObserver": "add event callback",
        }
    ),
    "property": QueryCategory(
        name="property",
        methods=[
            "SetColor", "SetOpacity", "SetAmbient", "SetDiffuse", "SetSpecular",
            "SetSpecularPower", "SetRepresentationToWireframe", "SetRepresentationToSurface",
            "SetLineWidth", "SetPointSize", "EdgeVisibilityOn", "SetEdgeColor",
            "BackfaceCullingOn", "FrontfaceCullingOn", "SetInterpolationToFlat",
            "SetInterpolationToGouraud", "SetInterpolationToPhong"
        ],
        classes=["vtkProperty", "vtkProperty2D"],
        queries=[
            "How do I set actor color in VTK?",
            "VTK property example",
            "How to make object transparent in VTK?",
            "How to show wireframe in VTK?",
            "VTK material properties",
        ],
        method_to_query={
            "SetColor": "set color",
            "SetOpacity": "set transparency",
            "SetRepresentationToWireframe": "show as wireframe",
            "SetRepresentationToSurface": "show as surface",
            "SetLineWidth": "set line width",
            "SetPointSize": "set point size",
            "EdgeVisibilityOn": "show edges",
        }
    ),
    "transform": QueryCategory(
        name="transform",
        methods=[
            "RotateX", "RotateY", "RotateZ", "RotateWXYZ",
            "SetPosition", "AddPosition", "SetScale", "SetOrigin",
            "SetOrientation", "SetUserTransform"
        ],
        classes=["vtkTransform", "vtkTransformFilter"],
        queries=[
            "How do I rotate an actor in VTK?",
            "VTK transform example",
            "How to position an actor in VTK?",
            "How to scale objects in VTK?",
        ],
        method_to_query={
            "RotateX": "rotate around X axis",
            "RotateY": "rotate around Y axis",
            "RotateZ": "rotate around Z axis",
            "SetPosition": "set position",
            "SetScale": "set scale",
            "SetOrientation": "set orientation",
        }
    ),
    "texture": QueryCategory(
        name="texture",
        methods=[
            "SetTexture", "SetInputConnection", "InterpolateOn", "RepeatOn",
            "SetBlendingMode", "MipmapOn"
        ],
        classes=["vtkTexture", "vtkTextureMapToPlane", "vtkTextureMapToCylinder",
                 "vtkTextureMapToSphere"],
        queries=[
            "How do I apply texture in VTK?",
            "VTK texture mapping example",
            "How to texture an object in VTK?",
        ],
        method_to_query={
            "SetTexture": "apply texture",
            "InterpolateOn": "enable texture interpolation",
            "RepeatOn": "enable texture repeat",
        }
    ),
}
