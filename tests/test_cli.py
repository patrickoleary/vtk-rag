"""Tests for the CLI module."""

import sys
from unittest.mock import patch

import pytest


class TestCLI:
    """Tests for the CLI module."""

    def test_import(self):
        """Test that CLI can be imported."""
        from vtk_rag import cli
        assert cli is not None

    def test_main_no_args(self):
        """Test main with no arguments shows help."""
        from vtk_rag.cli import main

        with patch.object(sys, 'argv', ['vtk-rag']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_cmd_chunk_function_exists(self):
        """Test that cmd_chunk function exists."""
        from vtk_rag.cli import cmd_chunk
        assert callable(cmd_chunk)

    def test_cmd_index_function_exists(self):
        """Test that cmd_index function exists."""
        from vtk_rag.cli import cmd_index
        assert callable(cmd_index)

    def test_cmd_build_function_exists(self):
        """Test that cmd_build function exists."""
        from vtk_rag.cli import cmd_build
        assert callable(cmd_build)

    def test_cmd_clean_function_exists(self):
        """Test that cmd_clean function exists."""
        from vtk_rag.cli import cmd_clean
        assert callable(cmd_clean)

    def test_cmd_search_function_exists(self):
        """Test that cmd_search function exists."""
        from vtk_rag.cli import cmd_search
        assert callable(cmd_search)


class TestBuild:
    """Tests for the build module."""

    def test_import(self):
        """Test that build module can be imported."""
        from vtk_rag import build
        assert build is not None

    def test_run_clean_function_exists(self):
        """Test that run_clean function exists."""
        from vtk_rag.build import run_clean
        assert callable(run_clean)

