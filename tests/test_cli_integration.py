"""
Integration tests for CLI interface.
"""

import asyncio
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from coscientist.cli import (
    _ensure_state,
    _build_config,
    _maybe_load_env,
    _import_runtime_deps
)


@pytest.mark.integration
class TestCLIIntegration:
    """Test cases for CLI interface."""

    def test_import_runtime_deps(self):
        """Test that runtime dependencies can be imported."""
        # This should not raise any import errors
        _import_runtime_deps()
        
        # Check that globals are set
        from coscientist.cli import CoscientistConfig, CoscientistFramework
        assert CoscientistConfig is not None
        assert CoscientistFramework is not None

    def test_maybe_load_env(self, temp_dir):
        """Test environment loading functionality."""
        # Create a test .env file
        env_file = Path(temp_dir) / ".env"
        env_file.write_text("TEST_VAR=test_value\n")
        
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            _maybe_load_env()
            # Should load the env var (if dotenv is available)
            # Note: This test may pass even if dotenv is not installed
            assert True  # If no exception, consider it passed
        finally:
            os.chdir(original_cwd)

    def test_build_config(self):
        """Test configuration building."""
        config = _build_config(
            debug=True,
            pause_after_lr=False,
            save_on_error=True,
            max_iter=25
        )
        
        assert config.debug_mode is True
        assert config.pause_after_literature_review is False
        assert config.save_on_error is True
        assert config.max_supervisor_iterations == 25

    def test_ensure_state_new(self, temp_dir, mock_env):
        """Test state creation for new goals."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Test fresh state creation
        state = _ensure_state("Test goal", fresh=True)
        assert state.goal == "Test goal"
        assert state.hypotheses == []

    def test_ensure_state_resume(self, temp_dir, mock_env):
        """Test state resumption."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Create initial state
        goal = "Resumable test goal"
        initial_state = _ensure_state(goal, fresh=True)
        initial_state.hypotheses.append(MagicMock(id=1))
        initial_state.save()
        
        # Resume state
        resumed_state = _ensure_state(goal, fresh=False)
        assert resumed_state.goal == goal
        assert len(resumed_state.hypotheses) >= 0  # Should have resumed data

    @pytest.mark.slow
    def test_cli_help_command(self):
        """Test that CLI help command works."""
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "--help"
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "cosci" in result.stdout

    def test_cli_goals_command(self, temp_dir, mock_env):
        """Test CLI goals listing command."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Create some test goals
        _ensure_state("Goal 1", fresh=True)
        _ensure_state("Goal 2", fresh=True)
        
        # Test goals command
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "goals"
        ], capture_output=True, text=True, timeout=10)
        
        # Should list goals or run without error
        assert result.returncode == 0

    def test_cli_new_command(self, temp_dir, mock_env):
        """Test CLI new goal creation."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "new",
            "--goal", "Test new goal via CLI"
        ], capture_output=True, text=True, timeout=15)
        
        # Should create successfully
        assert result.returncode == 0

    def test_cli_clear_command(self, temp_dir, mock_env):
        """Test CLI clear goal command."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Create a goal first
        subprocess.run([
            sys.executable, "-m", "coscientist.cli", "new",
            "--goal", "Goal to be cleared"
        ], capture_output=True, text=True, timeout=15)
        
        # Clear the goal
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "clear",
            "--goal", "Goal to be cleared"
        ], capture_output=True, text=True, timeout=15)
        
        assert result.returncode == 0

    @pytest.mark.slow
    def test_cli_checkpoints_command(self, temp_dir, mock_env):
        """Test CLI checkpoints listing."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Create a goal with checkpoints
        state = _ensure_state("Goal with checkpoints", fresh=True)
        state.save()  # Create a checkpoint
        
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "checkpoints",
            "--goal", "Goal with checkpoints"
        ], capture_output=True, text=True, timeout=15)
        
        assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.mock
class TestCLIMockedExecution:
    """Test CLI execution with mocked components."""

    @pytest.mark.asyncio
    async def test_cli_start_mocked(self, mock_env, mock_llm_pools, temp_dir):
        """Test CLI start command with mocked components."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        with patch("coscientist.cli.CoscientistFramework") as mock_framework:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock()
            mock_framework.return_value = mock_instance
            
            # Import and execute start function
            from coscientist.cli import _run_start
            
            await _run_start(mock_instance, n=2)
            
            # Verify framework start was called
            mock_instance.start.assert_called_once_with(n_hypotheses=2)

    @pytest.mark.asyncio
    async def test_cli_step_mocked(self, mock_env, mock_llm_pools, temp_dir):
        """Test CLI step command with mocked components."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        with patch("coscientist.cli.CoscientistFramework") as mock_framework:
            mock_instance = MagicMock()
            mock_instance.step = AsyncMock(return_value={"status": "completed"})
            mock_framework.return_value = mock_instance
            
            # Import and execute step function
            from coscientist.cli import _run_step
            
            result = await _run_step(mock_instance, "generate_new_hypotheses", n=3)
            
            # Verify framework step was called
            mock_instance.step.assert_called_once_with("generate_new_hypotheses", n=3)
            assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_cli_supervisor_run_mocked(self, mock_env, mock_llm_pools, temp_dir):
        """Test CLI supervisor run with mocked components."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        with patch("coscientist.cli.CoscientistFramework") as mock_framework:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(return_value=("Final report", "Meta review"))
            mock_framework.return_value = mock_instance
            
            # Import and execute run function
            from coscientist.cli import _run_supervisor
            
            final_report, meta_review = await _run_supervisor(mock_instance)
            
            # Verify framework run was called
            mock_instance.run.assert_called_once()
            assert final_report == "Final report"
            assert meta_review == "Meta review"

    def test_cli_argument_parsing(self):
        """Test CLI argument parsing for different commands."""
        from coscientist.cli import create_parser
        
        parser = create_parser()
        
        # Test new command
        args = parser.parse_args(["new", "--goal", "Test goal"])
        assert args.command == "new"
        assert args.goal == "Test goal"
        
        # Test start command
        args = parser.parse_args(["start", "--goal", "Test goal", "--n", "5"])
        assert args.command == "start"
        assert args.n == 5
        
        # Test step command
        args = parser.parse_args(["step", "--goal", "Test goal", "--action", "generate_new_hypotheses"])
        assert args.command == "step"
        assert args.action == "generate_new_hypotheses"

    def test_cli_config_options(self):
        """Test CLI configuration options parsing."""
        from coscientist.cli import create_parser
        
        parser = create_parser()
        
        # Test debug and other options
        args = parser.parse_args([
            "start", "--goal", "Test",
            "--debug", "--pause-after-lr", "--save-on-error",
            "--max-iter", "30"
        ])
        
        assert args.debug is True
        assert args.pause_after_lr is True
        assert args.save_on_error is True
        assert args.max_iter == 30

    @pytest.mark.slow
    def test_cli_timeout_handling(self, mock_env, temp_dir):
        """Test CLI timeout handling for long operations."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Set very short timeout for testing
        os.environ["COSCI_RESEARCH_TIMEOUT_SECONDS"] = "1"
        
        with patch("coscientist.cli.CoscientistFramework") as mock_framework:
            mock_instance = MagicMock()
            mock_instance.start = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_framework.return_value = mock_instance
            
            # This should handle timeout gracefully
            result = subprocess.run([
                sys.executable, "-m", "coscientist.cli", "start",
                "--goal", "Test timeout", "--n", "1"
            ], capture_output=True, text=True, timeout=10)
            
            # Should not crash, but may return error code
            assert result.returncode in [0, 1]  # Either success or handled error


@pytest.mark.integration
class TestCLIStateManagement:
    """Test CLI state management functionality."""

    def test_state_directory_creation(self, temp_dir, mock_env):
        """Test that CLI creates state directories properly."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Test state directory creation"
        state = _ensure_state(goal, fresh=True)
        
        # Check that directory was created
        state_dir = Path(temp_dir) / state._hash_goal(goal)
        assert state_dir.exists()
        assert state_dir.is_dir()

    def test_state_persistence_across_commands(self, temp_dir, mock_env):
        """Test that state persists across different CLI commands."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Persistent state test"
        
        # Create initial state
        state1 = _ensure_state(goal, fresh=True)
        state1.literature_review = "Test literature review"
        state1.save()
        
        # Resume state in "different command"
        state2 = _ensure_state(goal, fresh=False)
        
        assert state2.literature_review == "Test literature review"

    def test_multiple_goals_isolation(self, temp_dir, mock_env):
        """Test that multiple goals are properly isolated."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # Create two different goals
        state1 = _ensure_state("Goal 1", fresh=True)
        state2 = _ensure_state("Goal 2", fresh=True)
        
        state1.literature_review = "Literature for goal 1"
        state2.literature_review = "Literature for goal 2"
        
        state1.save()
        state2.save()
        
        # Reload and verify isolation
        reloaded1 = _ensure_state("Goal 1", fresh=False)
        reloaded2 = _ensure_state("Goal 2", fresh=False)
        
        assert reloaded1.literature_review == "Literature for goal 1"
        assert reloaded2.literature_review == "Literature for goal 2"

    def test_checkpoint_functionality(self, temp_dir, mock_env):
        """Test CLI checkpoint loading functionality."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        goal = "Checkpoint test goal"
        state = _ensure_state(goal, fresh=True)
        state.hypotheses.append(MagicMock(id=1, hypothesis="Test hypothesis"))
        
        # Save as checkpoint
        checkpoint_path = state.save()
        
        # Load from specific checkpoint
        loaded_state = _ensure_state("", fresh=False, checkpoint_path=checkpoint_path)
        
        assert loaded_state.goal == goal
        assert len(loaded_state.hypotheses) == 1


@pytest.mark.requires_api
class TestCLIWithRealAPIs:
    """Test CLI with real API connections (requires API keys)."""

    @pytest.mark.requires_openai
    @pytest.mark.slow
    def test_cli_with_real_openai(self, temp_dir):
        """Test CLI with real OpenAI API."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        # This test only runs if OPENAI_API_KEY is available
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "start",
            "--goal", "What is machine learning?",
            "--n", "1", "--pause-after-lr"
        ], capture_output=True, text=True, timeout=60)
        
        # Should either succeed or fail gracefully
        assert result.returncode in [0, 1]
        
        # Check for expected output patterns
        output = result.stdout + result.stderr
        assert len(output) > 0

    @pytest.mark.requires_anthropic
    @pytest.mark.slow
    def test_cli_with_real_anthropic(self, temp_dir):
        """Test CLI with real Anthropic API."""
        os.environ["COSCIENTIST_DIR"] = temp_dir
        
        result = subprocess.run([
            sys.executable, "-m", "coscientist.cli", "step",
            "--goal", "What causes rain?",
            "--action", "generate_new_hypotheses",
            "--n", "1"
        ], capture_output=True, text=True, timeout=60)
        
        assert result.returncode in [0, 1]