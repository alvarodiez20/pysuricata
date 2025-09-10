"""Tests for the service layer architecture.

This module tests the new service-oriented architecture for processing operations,
ensuring that the service layer provides proper abstraction, error handling,
and dependency injection capabilities.
"""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import polars as pl
import pytest

from pysuricata.compute.orchestration.services import (
    ChunkingService,
    EngineService,
    MetricsService,
    ProcessingResult,
    ProcessingService,
    ResourceManager,
)
from pysuricata.config import EngineConfig


class TestProcessingResult:
    """Test the ProcessingResult class."""

    def test_success_result(self):
        """Test successful processing result."""
        result = ProcessingResult(success=True, data="test_data", duration=1.5)

        assert result.success is True
        assert result.data == "test_data"
        assert result.error is None
        assert result.duration == 1.5
        assert result.metrics == {}

    def test_failure_result(self):
        """Test failed processing result."""
        result = ProcessingResult(
            success=False, error="Test error", metrics={"attempts": 3}
        )

        assert result.success is False
        assert result.data is None
        assert result.error == "Test error"
        assert result.metrics == {"attempts": 3}
        assert result.duration is None


class TestChunkingService:
    """Test the ChunkingService class."""

    def test_init(self):
        """Test service initialization."""
        config = EngineConfig(chunk_size=1000)
        service = ChunkingService(config)

        assert service.config == config
        assert service.chunk_size_cache == {}

    def test_chunks_from_source_success(self):
        """Test successful chunking."""
        service = ChunkingService()
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

        result = service.chunks_from_source(
            df, chunk_size=2, force_chunk_in_memory=True
        )

        assert result.success is True
        assert result.data is not None
        assert result.error is None
        assert result.duration >= 0  # Duration can be 0 for very fast operations
        assert result.metrics["chunk_size"] == 2
        assert result.metrics["force_chunk_in_memory"] is True

    def test_chunks_from_source_failure(self):
        """Test chunking failure handling."""
        service = ChunkingService()

        # Mock chunks_from_source to raise an exception
        with patch(
            "pysuricata.compute.orchestration.services.ChunkingService._chunk_dataframe_simple"
        ) as mock_chunks:
            mock_chunks.side_effect = Exception("Chunking failed")

            result = service.chunks_from_source("invalid_data", 100, False)

            assert result.success is False
            assert "Chunking failed" in result.error
            assert result.data is None

    def test_adaptive_chunk_size(self):
        """Test adaptive chunk size determination."""
        service = ChunkingService()
        df = pd.DataFrame({"a": [1, 2, 3]})

        # First call should cache the result
        size1 = service.adaptive_chunk_size(df)
        size2 = service.adaptive_chunk_size(df)

        assert size1 == size2
        assert type(df) in service.chunk_size_cache


class TestEngineService:
    """Test the EngineService class."""

    def test_init(self):
        """Test service initialization."""
        config = EngineConfig()
        service = EngineService(config)

        assert service.config == config

    def test_select_adapter_success(self):
        """Test successful adapter selection."""
        service = EngineService()
        df = pd.DataFrame({"a": [1, 2, 3]})

        result = service.select_adapter(df)

        assert result.success is True
        assert result.data is not None
        assert result.error is None
        assert result.duration >= 0  # Duration can be 0 for very fast operations
        assert "adapter_type" in result.metrics
        assert "adapter_tag" in result.metrics

    def test_select_adapter_failure(self):
        """Test adapter selection failure."""
        service = EngineService()

        # Test with unsupported data type
        result = service.select_adapter("unsupported_data")

        assert result.success is False
        assert "Unsupported input type" in result.error
        assert result.data is None

    @pytest.mark.skip(reason="Implementation detail test - needs refactoring")
    def test_consume_stream_success(self):
        """Test successful stream consumption."""
        service = EngineService()

        # Mock all the required parameters
        mock_adapter = Mock()
        mock_chunks = iter([pd.DataFrame({"a": [1, 2, 3]})])
        mock_accs = {}
        mock_kinds = Mock()
        mock_logger = Mock()
        mock_corr_est = None
        mock_row_kmv = Mock()

        # Mock consume_stream to return expected result
        with patch(
            "pysuricata.compute.orchestration.engine.consume_stream"
        ) as mock_consume:
            mock_consume.return_value = (100, 1000, 5, 1)

            result = service.consume_stream(
                adapter=mock_adapter,
                chunks=mock_chunks,
                accs=mock_accs,
                kinds=mock_kinds,
                logger=mock_logger,
                corr_est=mock_corr_est,
                row_kmv=mock_row_kmv,
                approx_mem_bytes=500,
                total_missing_cells=2,
                first_columns=[],
                sample_section_html="",
                cfg=EngineConfig(),
                ckpt_mgr=None,
                start_time=0.0,
                report_title=None,
                n_rows=50,
                chunk_idx=0,
            )

            assert result.success is True
            assert result.data == (100, 1000, 5, 1)
            assert result.duration >= 0  # Duration can be 0 for very fast operations
            assert "chunks_processed" in result.metrics


class TestMetricsService:
    """Test the MetricsService class."""

    def test_init(self):
        """Test service initialization."""
        config = EngineConfig()
        service = MetricsService(config)

        assert service.config == config

    @pytest.mark.skip(reason="Implementation detail test - needs refactoring")
    def test_build_manifest_inputs_success(self):
        """Test successful manifest building."""
        service = MetricsService()

        # Mock the required parameters
        mock_kinds = Mock()
        mock_accs = {"col1": Mock()}
        mock_row_kmv = Mock()
        mock_first_columns = ["col1"]

        # Mock build_manifest_inputs to return expected result
        with patch(
            "pysuricata.compute.analysis.metrics.build_manifest_inputs"
        ) as mock_build:
            mock_build.return_value = (
                {"col1": ("numeric", Mock())},
                ["col1"],
                100,
                1,
                [],
            )

            result = service.build_manifest_inputs(
                kinds=mock_kinds,
                accs=mock_accs,
                row_kmv=mock_row_kmv,
                first_columns=mock_first_columns,
            )

            assert result.success is True
            assert result.data is not None
            assert result.duration >= 0  # Duration can be 0 for very fast operations
            assert "n_rows" in result.metrics
            assert "n_cols" in result.metrics

    @pytest.mark.skip(reason="Implementation detail test - needs refactoring")
    def test_apply_correlation_chips_success(self):
        """Test successful correlation chips application."""
        service = MetricsService()

        mock_accs = {"col1": Mock()}
        mock_kinds = Mock()
        mock_top_map = {"col1": [("col2", 0.8)]}

        with patch(
            "pysuricata.compute.analysis.metrics.apply_corr_chips"
        ) as mock_apply:
            mock_apply.return_value = None

            result = service.apply_correlation_chips(
                accs=mock_accs,
                kinds=mock_kinds,
                top_map=mock_top_map,
            )

            assert result.success is True
            assert result.data is None
            assert result.duration >= 0  # Duration can be 0 for very fast operations
            assert result.metrics["correlations_applied"] == 1

    @pytest.mark.skip(reason="Implementation detail test - needs refactoring")
    def test_build_summary_success(self):
        """Test successful summary building."""
        service = MetricsService()

        mock_kinds_map = {"col1": ("numeric", Mock())}
        mock_col_order = ["col1"]
        mock_row_kmv = Mock()
        mock_miss_list = []

        with patch("pysuricata.compute.manifest.build_summary") as mock_build:
            mock_build.return_value = {"dataset": {"rows": 100}}

            result = service.build_summary(
                kinds_map=mock_kinds_map,
                col_order=mock_col_order,
                row_kmv=mock_row_kmv,
                total_missing_cells=5,
                n_rows=100,
                n_cols=1,
                miss_list=mock_miss_list,
            )

            assert result.success is True
            assert result.data is not None
            assert result.duration >= 0  # Duration can be 0 for very fast operations
            assert "summary_size" in result.metrics


class TestResourceManager:
    """Test the ResourceManager class."""

    def test_init(self):
        """Test resource manager initialization."""
        config = EngineConfig()
        manager = ResourceManager(config)

        assert manager.config == config
        assert manager.resources == {}

    def test_processing_context(self):
        """Test processing context manager."""
        manager = ResourceManager()

        with manager.processing_context() as resources:
            assert "start_time" in resources
            assert "memory_monitor" in resources
            assert "progress_tracker" in resources
            assert resources["memory_monitor"]["enabled"] is True
            assert resources["progress_tracker"]["enabled"] is True

        # Resources should be cleaned up after context
        assert manager.resources == {}


class TestProcessingService:
    """Test the ProcessingService class."""

    def test_init_with_dependency_injection(self):
        """Test initialization with dependency injection."""
        config = EngineConfig()
        chunking_service = ChunkingService(config)
        engine_service = EngineService(config)
        metrics_service = MetricsService(config)
        resource_manager = ResourceManager(config)

        service = ProcessingService(
            config=config,
            chunking_service=chunking_service,
            engine_service=engine_service,
            metrics_service=metrics_service,
            resource_manager=resource_manager,
        )

        assert service.config == config
        assert service.chunking_service == chunking_service
        assert service.engine_service == engine_service
        assert service.metrics_service == metrics_service
        assert service.resource_manager == resource_manager

    def test_init_with_defaults(self):
        """Test initialization with default services."""
        config = EngineConfig()
        service = ProcessingService(config)

        assert service.config == config
        assert isinstance(service.chunking_service, ChunkingService)
        assert isinstance(service.engine_service, EngineService)
        assert isinstance(service.metrics_service, MetricsService)
        assert isinstance(service.resource_manager, ResourceManager)

    def test_process_with_retry_success(self):
        """Test successful retry processing."""
        service = ProcessingService()

        def successful_operation():
            return ProcessingResult(success=True, data="success")

        result = service.process_with_retry(successful_operation, max_retries=3)

        assert result.success is True
        assert result.data == "success"

    def test_process_with_retry_failure(self):
        """Test failed retry processing."""
        service = ProcessingService()

        def failing_operation():
            raise Exception("Operation failed")

        result = service.process_with_retry(failing_operation, max_retries=2)

        assert result.success is False
        assert "Failed after 2 attempts" in result.error

    def test_process_with_retry_partial_success(self):
        """Test retry with partial success."""
        service = ProcessingService()

        call_count = 0

        def partially_failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Temporary failure")
            return ProcessingResult(success=True, data="success")

        result = service.process_with_retry(partially_failing_operation, max_retries=3)

        assert result.success is True
        assert result.data == "success"
        assert call_count == 3


class TestServiceIntegration:
    """Integration tests for the service layer."""

    def test_report_orchestrator_with_services(self):
        """Test ReportOrchestrator with injected services."""
        from pysuricata.report import ReportOrchestrator

        # Create mock services
        mock_chunking_service = Mock(spec=ChunkingService)
        mock_engine_service = Mock(spec=EngineService)
        mock_metrics_service = Mock(spec=MetricsService)
        mock_resource_manager = Mock(spec=ResourceManager)

        # Configure mock services to return successful results
        mock_chunking_service.chunks_from_source.return_value = ProcessingResult(
            success=True, data=iter([pd.DataFrame({"a": [1, 2, 3]})])
        )

        mock_engine_service.select_adapter.return_value = ProcessingResult(
            success=True, data=Mock()
        )

        mock_engine_service.consume_stream.return_value = ProcessingResult(
            success=True, data=(100, 1000, 5, 1)
        )

        mock_metrics_service.build_manifest_inputs.return_value = ProcessingResult(
            success=True, data=({}, [], 100, 1, [])
        )

        mock_metrics_service.build_summary.return_value = ProcessingResult(
            success=True, data={"dataset": {"rows": 100}}
        )

        # Create orchestrator with injected services
        orchestrator = ReportOrchestrator(
            chunking_service=mock_chunking_service,
            engine_service=mock_engine_service,
            metrics_service=mock_metrics_service,
            resource_manager=mock_resource_manager,
        )

        # Test that services are properly injected
        assert orchestrator.chunking_service == mock_chunking_service
        assert orchestrator.engine_service == mock_engine_service
        assert orchestrator.metrics_service == mock_metrics_service
        assert orchestrator.resource_manager == mock_resource_manager

    def test_service_error_handling(self):
        """Test that services properly handle and propagate errors."""
        service = ProcessingService()

        # Test with operation that returns failed ProcessingResult
        def failing_operation():
            return ProcessingResult(success=False, error="Service error")

        result = service.process_with_retry(failing_operation, max_retries=1)

        assert result.success is False
        assert "Service error" in result.error
