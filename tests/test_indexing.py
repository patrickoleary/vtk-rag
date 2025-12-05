"""Tests for the indexing module."""



class TestIndexer:
    """Tests for the Indexer class."""

    def test_import(self):
        """Test that Indexer can be imported."""
        from vtk_rag.indexing import Indexer
        assert Indexer is not None

    def test_collection_configs_import(self):
        """Test that collection configs can be imported."""
        from vtk_rag.indexing import (
            CODE_COLLECTION_CONFIG,
            DOC_COLLECTION_CONFIG,
            CollectionConfig,
            FieldConfig,
        )
        assert CollectionConfig is not None
        assert FieldConfig is not None
        assert CODE_COLLECTION_CONFIG is not None
        assert DOC_COLLECTION_CONFIG is not None

    def test_code_collection_config(self):
        """Test code collection configuration."""
        from vtk_rag.indexing import CODE_COLLECTION_CONFIG

        assert CODE_COLLECTION_CONFIG.name == "vtk_code"
        assert len(CODE_COLLECTION_CONFIG.indexed_fields) > 0

    def test_doc_collection_config(self):
        """Test doc collection configuration."""
        from vtk_rag.indexing import DOC_COLLECTION_CONFIG

        assert DOC_COLLECTION_CONFIG.name == "vtk_docs"
        assert len(DOC_COLLECTION_CONFIG.indexed_fields) > 0


class TestCollectionConfig:
    """Tests for CollectionConfig dataclass."""

    def test_create_config(self):
        """Test creating a collection config."""
        from qdrant_client.models import PayloadSchemaType

        from vtk_rag.indexing import CollectionConfig, FieldConfig

        config = CollectionConfig(
            name="test_collection",
            description="Test collection",
            indexed_fields=[
                FieldConfig(name="test_field", index_type=PayloadSchemaType.KEYWORD),
            ],
        )

        assert config.name == "test_collection"
        assert config.description == "Test collection"
        assert len(config.indexed_fields) == 1
        assert config.indexed_fields[0].name == "test_field"
