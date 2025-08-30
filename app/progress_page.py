"""
Live progress monitoring page for the Coscientist web interface.

Provides real-time visibility into the research process, addressing user concerns
about the "black box" nature of the system during long-running operations.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from coscientist.progress_tracker import ProgressTracker, ProgressPhase, ProgressStatus


def display_progress_page():
    """Display the live progress monitoring page."""
    st.header("📊 Live Progress Monitor")
    
    # Check if there's an ongoing research process
    st.markdown("""
    Monitor the real-time progress of ongoing Coscientist research sessions.
    Get detailed insights into each phase of the research process and track completion status.
    """)
    
    # Goal selection for progress monitoring
    st.subheader("🎯 Select Research Goal")
    
    # Get available research goals from the goal-based directory structure
    from common import get_available_states
    available_states = get_available_states()
    
    if not available_states:
        st.warning("No active research goals found.")
        st.markdown("""
        ## How to Use the Progress Monitor
        
        1. **Start a research session** using the Configuration Agent page
        2. **Return to this page** to monitor real-time progress
        3. **View detailed progress** including current phase, completion percentage, and time estimates
        4. **Track performance metrics** such as API usage and processing times
        
        ### Features Available:
        - **Real-time progress bars** showing overall and phase-specific completion
        - **Live activity feed** with recent actions and milestones
        - **Performance metrics** including time per phase and resource usage
        - **Error monitoring** with detailed error logs and recovery suggestions
        - **Historical tracking** showing progress over time
        """)
        return
    
    # Select goal to monitor
    selected_goal = st.selectbox(
        "Choose research goal to monitor:",
        options=available_states,
        help="Select the research goal you want to monitor in real-time"
    )
    
    if selected_goal:
        # Get current progress data
        progress_data = ProgressTracker.get_current_progress(selected_goal)
        
        if progress_data is None:
            st.info(f"No active progress data found for: {selected_goal}")
            st.markdown("This research goal may not have started yet or may have completed.")
        else:
            display_progress_dashboard(selected_goal, progress_data)


def display_progress_dashboard(goal: str, progress_data: Dict[str, Any]):
    """Display the main progress dashboard."""
    
    # Auto-refresh functionality
    st.sidebar.subheader("🔄 Auto-Refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10)
    
    if auto_refresh:
        # Auto-refresh using st.rerun()
        time.sleep(refresh_interval)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Progress overview
    st.subheader("📈 Progress Overview")
    
    overall_progress = progress_data.get("overall_progress", 0)
    current_phase = progress_data.get("current_phase", "unknown")
    current_status = progress_data.get("current_status", "unknown")
    current_message = progress_data.get("current_message", "No status available")
    
    # Main progress metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Progress", f"{overall_progress:.1f}%")
    
    with col2:
        st.metric("Current Phase", current_phase.replace("_", " ").title())
    
    with col3:
        total_elapsed = progress_data.get("total_elapsed_time", 0)
        elapsed_str = format_duration(total_elapsed)
        st.metric("Elapsed Time", elapsed_str)
    
    with col4:
        remaining_time = progress_data.get("estimated_remaining_time", 0)
        remaining_str = format_duration(remaining_time) if remaining_time > 0 else "Calculating..."
        st.metric("Est. Remaining", remaining_str)
    
    # Progress bar
    st.progress(overall_progress / 100)
    
    # Current status
    status_color = get_status_color(current_status)
    st.markdown(f"**Current Status:** :{status_color}[{current_status.upper()}] {current_message}")
    
    # Check for errors
    if progress_data.get("error_occurred", False):
        st.error("⚠️ Errors were detected during the research process. Check the activity log below for details.")
    
    # Phase-specific progress
    display_phase_progress(current_phase, progress_data)
    
    # Activity feed and detailed information
    col1, col2 = st.columns([2, 1])
    
    with col1:
        display_activity_feed(goal)
    
    with col2:
        display_performance_metrics(progress_data)
        display_system_info(goal)


def display_phase_progress(current_phase: str, progress_data: Dict[str, Any]):
    """Display phase-specific progress information."""
    st.subheader("🔍 Phase Details")
    
    # Phase descriptions
    phase_descriptions = {
        "initializing": "Setting up the research framework and preparing agents",
        "literature_review": "Conducting comprehensive literature analysis and subtopic research",
        "hypothesis_generation": "Generating research hypotheses using multi-agent collaboration",
        "reflection": "Performing deep verification and review of generated hypotheses",
        "tournament": "Running ELO tournament to rank hypotheses through head-to-head comparisons",
        "evolution": "Evolving hypotheses based on tournament feedback and insights",
        "meta_review": "Analyzing and synthesizing insights from top-performing hypotheses",
        "final_report": "Compiling comprehensive final research report",
        "completed": "Research process completed successfully",
        "error": "Error encountered during research process"
    }
    
    description = phase_descriptions.get(current_phase, "Unknown phase")
    st.info(f"**{current_phase.replace('_', ' ').title()}:** {description}")
    
    # Phase-specific progress bar if available
    current_progress = progress_data.get("phase_progress_percentage")
    if current_progress is not None:
        st.progress(current_progress / 100)
        st.caption(f"Phase Progress: {current_progress:.1f}%")


def display_activity_feed(goal: str):
    """Display recent activity and events."""
    st.subheader("📋 Recent Activity")
    
    # Load the full progress tracker to get event details
    tracker = ProgressTracker.load_progress(goal)
    if tracker is None or not tracker.events:
        st.info("No activity data available yet.")
        return
    
    # Display recent events (last 10)
    recent_events = tracker.events[-10:]
    
    for event in reversed(recent_events):  # Show most recent first
        timestamp = datetime.fromtimestamp(event.timestamp)
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Color code by status
        status_icon = get_status_icon(event.status)
        phase_emoji = get_phase_emoji(event.phase)
        
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.caption(time_str)
            with col2:
                st.markdown(f"{status_icon} {phase_emoji} {event.message}")
                
                # Show details if available
                if event.details:
                    with st.expander("Details"):
                        st.json(event.details)
                
                # Show error info if available
                if event.error_info:
                    st.error(f"Error: {event.error_info}")


def display_performance_metrics(progress_data: Dict[str, Any]):
    """Display performance metrics and statistics."""
    st.subheader("⚡ Performance")
    
    total_elapsed = progress_data.get("total_elapsed_time", 0)
    if total_elapsed > 0:
        st.metric("Total Runtime", format_duration(total_elapsed))
    
    # Estimated completion time
    estimated_total = progress_data.get("estimated_total_time", 0)
    if estimated_total > 0:
        completion_time = datetime.now() + timedelta(seconds=progress_data.get("estimated_remaining_time", 0))
        st.metric("Est. Completion", completion_time.strftime("%H:%M"))
    
    # Last update time
    last_update = progress_data.get("last_update", 0)
    if last_update > 0:
        last_update_str = datetime.fromtimestamp(last_update).strftime("%H:%M:%S")
        st.caption(f"Last updated: {last_update_str}")


def display_system_info(goal: str):
    """Display system information and status."""
    st.subheader("🔧 System Info")
    
    # Goal information
    st.info(f"**Goal:** {goal[:100]}{'...' if len(goal) > 100 else ''}")
    
    # Progress file information
    progress_file_exists = ProgressTracker.get_current_progress(goal) is not None
    status = "✅ Active" if progress_file_exists else "❌ Inactive"
    st.caption(f"Progress tracking: {status}")


def get_status_color(status: str) -> str:
    """Get color for status display."""
    color_map = {
        "started": "blue",
        "in_progress": "orange",
        "completed": "green",
        "error": "red",
        "skipped": "gray"
    }
    return color_map.get(status, "gray")


def get_status_icon(status: ProgressStatus) -> str:
    """Get icon for status display."""
    icon_map = {
        ProgressStatus.STARTED: "🚀",
        ProgressStatus.IN_PROGRESS: "⚙️",
        ProgressStatus.COMPLETED: "✅",
        ProgressStatus.ERROR: "❌",
        ProgressStatus.SKIPPED: "⏭️"
    }
    return icon_map.get(status, "🔄")


def get_phase_emoji(phase: ProgressPhase) -> str:
    """Get emoji for phase display."""
    emoji_map = {
        ProgressPhase.INITIALIZING: "🔧",
        ProgressPhase.LITERATURE_REVIEW: "📚",
        ProgressPhase.HYPOTHESIS_GENERATION: "💡",
        ProgressPhase.REFLECTION: "🤔",
        ProgressPhase.TOURNAMENT: "🏆",
        ProgressPhase.EVOLUTION: "🧬",
        ProgressPhase.META_REVIEW: "📊",
        ProgressPhase.FINAL_REPORT: "📄",
        ProgressPhase.COMPLETED: "🎉",
        ProgressPhase.ERROR: "⚠️"
    }
    return emoji_map.get(phase, "🔄")


def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_progress_timeline(tracker: ProgressTracker) -> go.Figure:
    """Create a timeline visualization of the research progress."""
    if not tracker.events:
        return go.Figure()
    
    # Prepare data for timeline
    phases = []
    start_times = []
    end_times = []
    durations = []
    
    current_phase = None
    phase_start = None
    
    for event in tracker.events:
        if event.status == ProgressStatus.STARTED:
            if current_phase is not None and phase_start is not None:
                # End previous phase
                phases.append(current_phase.value.replace('_', ' ').title())
                start_times.append(phase_start)
                end_times.append(event.timestamp)
                durations.append(event.timestamp - phase_start)
            
            # Start new phase
            current_phase = event.phase
            phase_start = event.timestamp
    
    # Handle the last phase
    if current_phase is not None and phase_start is not None:
        phases.append(current_phase.value.replace('_', ' ').title())
        start_times.append(phase_start)
        end_times.append(tracker.events[-1].timestamp)
        durations.append(tracker.events[-1].timestamp - phase_start)
    
    # Create Gantt chart
    fig = go.Figure()
    
    for i, (phase, start, end, duration) in enumerate(zip(phases, start_times, end_times, durations)):
        fig.add_trace(go.Scatter(
            x=[datetime.fromtimestamp(start), datetime.fromtimestamp(end)],
            y=[phase, phase],
            mode='lines',
            line=dict(width=20),
            name=phase,
            hovertemplate=f"{phase}<br>Duration: {format_duration(duration)}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Research Progress Timeline",
        xaxis_title="Time",
        yaxis_title="Phase",
        height=400
    )
    
    return fig