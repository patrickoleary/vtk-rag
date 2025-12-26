"""Group VTK lifecycles into semantic chunks based on their roles and relationships.

This module organizes lifecycles for chunking by grouping related objects together.
Properties (mappers, actors) are grouped with their dependencies; infrastructure
(windows, interactors) are grouped together with proper ordering.

Called by:
    CodeChunker in chunker.py

Code Map:
    group_lifecycles()
        ├── _group_properties()     # mapper/actor relationship grouping
        └── _group_infrastructure() # window/interactor ordering
"""

from __future__ import annotations

import ast

from .models import VTKLifecycle
from .vtk_knowledge import ACTOR_LIKE_PROPS


def group_lifecycles(lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
    """Group lifecycles into semantic chunks based on role.

    Called by: CodeChunker._extract_function_chunks(), _extract_module_chunks(),
               _extract_class_chunks()

    Output order: Infrastructure → Properties → Singles
    """
    # 1. Classify lifecycles by role
    properties_lifecycles: list[VTKLifecycle] = []
    infrastructure_lifecycles: list[VTKLifecycle] = []
    single_lifecycle_groups: list[list[VTKLifecycle]] = []

    for lc in lifecycles:
        role = lc["type"]
        if role == "properties":
            properties_lifecycles.append(lc)  # mappers, actors, property objects
        elif role == "infrastructure":
            infrastructure_lifecycles.append(lc)  # windows, interactors, styles
        else:
            single_lifecycle_groups.append([lc])  # input, filter, output, renderer, scene, utility, color

    # 2. Group properties with proper ordering (mapper props → mapper → actor props → actor)
    properties_groups = _group_properties(properties_lifecycles)

    # 3. Group infrastructure with proper ordering (window → interactor → styles → Start)
    infrastructure_group = _group_infrastructure(infrastructure_lifecycles)

    # 4. Return all groups: infrastructure first, then properties, then singles
    return infrastructure_group + properties_groups + single_lifecycle_groups

def _group_properties(properties_lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
    """Group mappers and actors with their related property objects.

    Output order per group: Mapper props → Mapper → Actor props → Actor
    """
    if not properties_lifecycles:
        return []

    groups: list[list[VTKLifecycle]] = []
    processed: set[str] = set()

    # 1. Build lookup for quick access by variable name
    var_to_lc = {lc["variable"]: lc for lc in properties_lifecycles if lc["variable"]}

    # 2. Process actors first (top of VTK pipeline)
    for lc in properties_lifecycles:
        if lc["variable"] in processed:
            continue  # already grouped, next lifecycle

        # Only process actors here; mappers handled in section 3
        is_actor = (
            ("Actor" in lc["class"] and not lc["class"].endswith("Property"))
            or lc["class"] in ACTOR_LIKE_PROPS
        )
        if not is_actor:
            continue  # skip mappers/properties, handle later

        # Found an actor - build its group with mapper and properties
        group: list[VTKLifecycle] = []

        # 2a. Add actor's mapper and mapper's properties first
        mapper_var = lc.get("mapper")
        if mapper_var and mapper_var in var_to_lc:
            mapper_lc = var_to_lc[mapper_var]
            # Mapper's properties first
            for prop in mapper_lc.get("properties", []):
                prop_var = prop["variable"]
                # Skip inline properties (no separate lifecycle)
                if prop_var != "inline" and prop_var in var_to_lc:
                    group.append(var_to_lc[prop_var])  # add mapper properties to properties group
                    processed.add(prop_var)  # mark as grouped
            # Then mapper itself
            group.append(mapper_lc)  # add mapper to properties group
            processed.add(mapper_var)  # mark as grouped

        # 2b. Add actor's properties
        for prop in lc.get("properties", []):
            prop_var = prop["variable"]
            # Skip inline, missing, or already-grouped properties
            if prop_var != "inline" and prop_var in var_to_lc and prop_var not in processed:
                group.append(var_to_lc[prop_var])  # add actor's properties to properties group
                processed.add(prop_var)  # mark as grouped

        # 2c. Add actor
        group.append(lc)  # add actor to properties group
        processed.add(lc["variable"])  # mark as grouped

        # Add completed group to groups
        groups.append(group)

    # 3. Process orphan mappers (mappers not claimed by an actor in section 2)
    for lc in properties_lifecycles:
        if lc["variable"] in processed:
            continue  # already grouped with an actor
        if "Mapper" not in lc["class"]:
            continue  # not a mapper, handle in section 4

        # Found an orphan mapper - build its group with properties
        group: list[VTKLifecycle] = []

        # 3a. Mapper's properties first
        for prop in lc.get("properties", []):
            prop_var = prop["variable"]
            # Skip inline, missing, or already-grouped properties
            if prop_var != "inline" and prop_var in var_to_lc and prop_var not in processed:
                group.append(var_to_lc[prop_var])  # add mapper properties to properties group
                processed.add(prop_var)  # mark as grouped

        # 3b. Then mapper
        group.append(lc)  # add mapper to properties group
        processed.add(lc["variable"])  # mark as grouped

        groups.append(group)  # add completed group to groups

    # 4. Process remaining (LUTs, transfer functions, textures, etc.)
    for lc in properties_lifecycles:
        if lc["variable"] not in processed:
            groups.append([lc])  # add remaining lifecycle to groups
            processed.add(lc["variable"]) # mark as grouped

    return groups

def _group_infrastructure(infrastructure_lifecycles: list[VTKLifecycle]) -> list[list[VTKLifecycle]]:
    """Group infrastructure lifecycles into a single ordered group.

    Output order: Window → Interactor (setup) → Styles → Interactor.Start()

    The interactor is split so Start() comes last (it blocks until window closes).
    """
    if not infrastructure_lifecycles:
        return []

    # 1. Classify lifecycles by type
    window_lifecycles: list[VTKLifecycle] = []
    interactor_lifecycle: VTKLifecycle | None = None
    style_lifecycles: list[VTKLifecycle] = []

    for lc in infrastructure_lifecycles:
        vtk_class = lc["class"]
        if "RenderWindow" in vtk_class and "Interactor" not in vtk_class:
            window_lifecycles.append(lc)
        elif "Interactor" in vtk_class and "Style" not in vtk_class:
            interactor_lifecycle = lc
        else:
            style_lifecycles.append(lc)

    # 2. Build result in order: Window → Interactor → Styles → Start()
    result: list[VTKLifecycle] = []

    # 2a. Add windows first
    result.extend(window_lifecycles)

    # 2b. Add interactor (split: setup statements now, Start() at end)
    if interactor_lifecycle:
        # Separate Start() from other statements
        start_stmt = None
        other_stmts = []
        for stmt in interactor_lifecycle.get("statements", []):
            is_start = (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "Start"
            )
            if is_start:
                start_stmt = stmt
            else:
                other_stmts.append(stmt)

        # Add interactor setup (without Start)
        interactor_setup = dict(interactor_lifecycle)
        interactor_setup["statements"] = other_stmts
        interactor_setup["methods"] = [m for m in interactor_lifecycle.get("methods", []) if m != "Start"]
        result.append(interactor_setup)

        # 2c. Add styles after interactor setup
        result.extend(style_lifecycles)

        # 2d. Add Start() as final lifecycle (blocks until window closes)
        if start_stmt:
            result.append({
                "variable": interactor_lifecycle["variable"],
                "class": interactor_lifecycle["class"],
                "type": interactor_lifecycle["type"],
                "statements": [start_stmt],
                "properties": [],
                "mapper": None,
                "actor": None,
                "methods": ["Start"],
                "method_calls": [],
            })
    else:
        # No interactor, just add styles
        result.extend(style_lifecycles)

    return [result] if result else []
