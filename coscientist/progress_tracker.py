"""
Progress tracking system for the Coscientist multi-agent framework.

This module provides real-time progress visibility during research operations,
addressing the "black box" problem where users have no insight into ongoing processes.
"""

import json
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict, Union
from dataclasses import dataclass, asdict


class ProgressPhase(Enum):
    """Enumeration of main research phases."""
    
    INITIALIZING = "initializing"
    LITERATURE_REVIEW = "literature_review"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    REFLECTION = "reflection"
    TOURNAMENT = "tournament"
    EVOLUTION = "evolution"
    META_REVIEW = "meta_review"
    FINAL_REPORT = "final_report"
    COMPLETED = "completed"
    ERROR = "error"


class ProgressStatus(Enum):
    """Status of a progress event."""
    
    STARTED = "started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class ProgressEvent:
    """A single progress event in the research process."""
    
    timestamp: float
    phase: ProgressPhase
    status: ProgressStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    progress_percentage: Optional[float] = None
    estimated_duration: Optional[float] = None
    error_info: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['phase'] = self.phase.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressEvent':
        """Create from dictionary."""
        data['phase'] = ProgressPhase(data['phase'])
        data['status'] = ProgressStatus(data['status'])
        return cls(**data)


class PhaseEstimator:
    """Estimates completion time based on historical data."""
    
    DEFAULT_DURATIONS = {
        ProgressPhase.LITERATURE_REVIEW: 300,  # 5 minutes
        ProgressPhase.HYPOTHESIS_GENERATION: 180,  # 3 minutes per hypothesis
        ProgressPhase.REFLECTION: 120,  # 2 minutes per hypothesis
        ProgressPhase.TOURNAMENT: 240,  # 4 minutes
        ProgressPhase.EVOLUTION: 200,  # 3.3 minutes
        ProgressPhase.META_REVIEW: 150,  # 2.5 minutes
        ProgressPhase.FINAL_REPORT: 120,  # 2 minutes
    }
    
    def __init__(self):
        self.phase_starts: Dict[ProgressPhase, float] = {}
        self.historical_durations: Dict[ProgressPhase, List[float]] = {}
    
    def start_phase(self, phase: ProgressPhase) -> None:
        """Record the start time of a phase."""
        self.phase_starts[phase] = time.time()
    
    def complete_phase(self, phase: ProgressPhase) -> float:
        """Record completion and return duration."""
        if phase not in self.phase_starts:
            return 0
        
        duration = time.time() - self.phase_starts[phase]
        
        if phase not in self.historical_durations:
            self.historical_durations[phase] = []
        self.historical_durations[phase].append(duration)
        
        return duration
    
    def estimate_remaining_time(self, current_phase: ProgressPhase, 
                              progress_percentage: Optional[float] = None) -> float:
        """Estimate total remaining time."""
        remaining_time = 0
        
        # Current phase remaining time
        if progress_percentage is not None and current_phase in self.phase_starts:
            current_duration = time.time() - self.phase_starts[current_phase]
            if progress_percentage > 0:
                estimated_total = current_duration / (progress_percentage / 100)
                remaining_time += max(0, estimated_total - current_duration)
        else:
            # Use default duration for current phase
            remaining_time += self.get_average_duration(current_phase)
        
        # Add estimated time for remaining phases
        phase_order = list(ProgressPhase)
        current_index = phase_order.index(current_phase)
        
        for phase in phase_order[current_index + 1:]:
            if phase in [ProgressPhase.COMPLETED, ProgressPhase.ERROR]:
                break
            remaining_time += self.get_average_duration(phase)
        
        return remaining_time
    
    def get_average_duration(self, phase: ProgressPhase) -> float:
        """Get average duration for a phase."""
        if phase in self.historical_durations and self.historical_durations[phase]:
            return sum(self.historical_durations[phase]) / len(self.historical_durations[phase])
        return self.DEFAULT_DURATIONS.get(phase, 120)  # Default 2 minutes


class ProgressTracker:
    """Main progress tracking system."""
    
    def __init__(self, goal: str, output_dir: Optional[str] = None):
        self.goal = goal
        self.output_dir = output_dir or self._get_output_dir(goal)
        self.events: List[ProgressEvent] = []
        self.current_phase = ProgressPhase.INITIALIZING
        self.estimator = PhaseEstimator()
        self.start_time = time.time()
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize with first event
        self.emit_event(
            phase=ProgressPhase.INITIALIZING,
            status=ProgressStatus.STARTED,
            message="Initializing Coscientist research framework"
        )
    
    def _get_output_dir(self, goal: str) -> str:
        """Get the output directory for progress files."""
        from coscientist.global_state import CoscientistState
        goal_hash = CoscientistState._hash_goal(goal)
        return os.path.join(
            os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
            goal_hash
        )
    
    def emit_event(self, 
                   phase: ProgressPhase,
                   status: ProgressStatus,
                   message: str,
                   details: Optional[Dict[str, Any]] = None,
                   progress_percentage: Optional[float] = None,
                   estimated_duration: Optional[float] = None,
                   error_info: Optional[str] = None) -> None:
        """Emit a new progress event."""
        
        # Update phase tracking
        if phase != self.current_phase and status == ProgressStatus.STARTED:
            if self.current_phase != ProgressPhase.INITIALIZING:
                self.estimator.complete_phase(self.current_phase)
            self.current_phase = phase
            self.estimator.start_phase(phase)
        
        # Calculate estimated remaining time
        if estimated_duration is None:
            estimated_duration = self.estimator.estimate_remaining_time(
                phase, progress_percentage
            )
        
        event = ProgressEvent(
            timestamp=time.time(),
            phase=phase,
            status=status,
            message=message,
            details=details,
            progress_percentage=progress_percentage,
            estimated_duration=estimated_duration,
            error_info=error_info
        )
        
        self.events.append(event)
        self._persist_progress()
    
    def _persist_progress(self) -> None:
        """Save progress to disk for real-time monitoring."""
        progress_file = os.path.join(self.output_dir, "progress.json")
        
        progress_data = {
            "goal": self.goal,
            "start_time": self.start_time,
            "current_phase": self.current_phase.value,
            "events": [event.to_dict() for event in self.events],
            "summary": self.get_progress_summary()
        }
        
        # Atomic write to prevent corruption
        temp_file = progress_file + ".tmp"
        with open(temp_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        os.rename(temp_file, progress_file)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress."""
        if not self.events:
            return {}
        
        latest_event = self.events[-1]
        total_elapsed = time.time() - self.start_time
        
        # Calculate overall progress percentage
        phase_order = [
            ProgressPhase.LITERATURE_REVIEW,
            ProgressPhase.HYPOTHESIS_GENERATION,
            ProgressPhase.REFLECTION,
            ProgressPhase.TOURNAMENT,
            ProgressPhase.EVOLUTION,
            ProgressPhase.META_REVIEW,
            ProgressPhase.FINAL_REPORT
        ]
        
        if latest_event.phase == ProgressPhase.COMPLETED:
            overall_progress = 100.0
        elif latest_event.phase in phase_order:
            phase_index = phase_order.index(latest_event.phase)
            base_progress = (phase_index / len(phase_order)) * 100
            
            if latest_event.progress_percentage:
                phase_contribution = (latest_event.progress_percentage / len(phase_order))
                overall_progress = base_progress + phase_contribution
            else:
                overall_progress = base_progress
        else:
            overall_progress = 0.0
        
        return {
            "overall_progress": min(100.0, overall_progress),
            "current_phase": latest_event.phase.value,
            "current_status": latest_event.status.value,
            "current_message": latest_event.message,
            "total_elapsed_time": total_elapsed,
            "estimated_remaining_time": latest_event.estimated_duration or 0,
            "estimated_total_time": total_elapsed + (latest_event.estimated_duration or 0),
            "last_update": latest_event.timestamp,
            "error_occurred": any(e.status == ProgressStatus.ERROR for e in self.events)
        }
    
    def start_phase(self, phase: ProgressPhase, message: str) -> None:
        """Start a new research phase."""
        self.emit_event(
            phase=phase,
            status=ProgressStatus.STARTED,
            message=message
        )
    
    def update_phase_progress(self, message: str, progress_percentage: Optional[float] = None,
                            details: Optional[Dict[str, Any]] = None) -> None:
        """Update progress within the current phase."""
        self.emit_event(
            phase=self.current_phase,
            status=ProgressStatus.IN_PROGRESS,
            message=message,
            details=details,
            progress_percentage=progress_percentage
        )
    
    def complete_phase(self, message: str) -> None:
        """Mark the current phase as completed."""
        self.emit_event(
            phase=self.current_phase,
            status=ProgressStatus.COMPLETED,
            message=message
        )
    
    def report_error(self, error_message: str, error_info: Optional[str] = None) -> None:
        """Report an error in the current phase."""
        self.emit_event(
            phase=self.current_phase,
            status=ProgressStatus.ERROR,
            message=f"Error in {self.current_phase.value}: {error_message}",
            error_info=error_info
        )
        
        # Also emit a general error phase event
        self.emit_event(
            phase=ProgressPhase.ERROR,
            status=ProgressStatus.COMPLETED,
            message=f"Research failed: {error_message}",
            error_info=error_info
        )
    
    def complete_research(self) -> None:
        """Mark the entire research process as completed."""
        self.emit_event(
            phase=ProgressPhase.COMPLETED,
            status=ProgressStatus.COMPLETED,
            message="Research process completed successfully"
        )
    
    @classmethod
    def load_progress(cls, goal: str) -> Optional['ProgressTracker']:
        """Load existing progress tracker from disk."""
        output_dir = cls._get_output_dir_static(goal)
        progress_file = os.path.join(output_dir, "progress.json")
        
        if not os.path.exists(progress_file):
            return None
        
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
            
            tracker = cls(goal=goal, output_dir=output_dir)
            tracker.start_time = data.get('start_time', time.time())
            tracker.current_phase = ProgressPhase(data.get('current_phase', 'initializing'))
            
            # Load events
            tracker.events = []
            for event_data in data.get('events', []):
                tracker.events.append(ProgressEvent.from_dict(event_data))
            
            return tracker
            
        except Exception:
            # If loading fails, return None to create fresh tracker
            return None
    
    @staticmethod
    def _get_output_dir_static(goal: str) -> str:
        """Static version of output directory calculation."""
        import hashlib
        goal_hash = hashlib.sha256(goal.strip().lower().encode("utf-8")).hexdigest()[:12]
        return os.path.join(
            os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
            goal_hash
        )
    
    @classmethod
    def get_current_progress(cls, goal: str) -> Optional[Dict[str, Any]]:
        """Get current progress summary for a goal without loading full tracker."""
        output_dir = cls._get_output_dir_static(goal)
        progress_file = os.path.join(output_dir, "progress.json")
        
        if not os.path.exists(progress_file):
            return None
        
        try:
            with open(progress_file, 'r') as f:
                data = json.load(f)
            return data.get('summary', {})
        except Exception:
            return None