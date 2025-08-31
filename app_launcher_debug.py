#!/usr/bin/env python3
"""
Streamlit app for confirming research question and launching CoScientist.
Run with: streamlit run app_launcher.py
"""

import streamlit as st
import asyncio
import logging
from pathlib import Path
import sys
import os
import time
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any

# Add coscientist to path
sys.path.insert(0, str(Path(__file__).parent))

from coscientist.framework import CoscientistConfig, CoscientistFramework, _SMARTER_LLM_POOL
from coscientist.global_state import CoscientistState, CoscientistStateManager
from coscientist.progress_tracker import ProgressTracker, ProgressPhase, ProgressStatus


class StreamlitProgressTracker:
    """Custom progress tracker that updates Streamlit session state."""
    
    def __init__(self):
        self.current_phase = "Waiting"
        self.current_subtask = ""
        self.progress_percentage = 0
        self.phases = {
            "Literature Review": 20,
            "Hypothesis Generation": 40,
            "Reflection & Review": 60,
            "Tournament Ranking": 75,
            "Evolution": 85,
            "Meta Review": 95,
            "Final Report": 100
        }
    
    def update_phase(self, phase_name: str, subtask: str = ""):
        """Update the current research phase."""
        self.current_phase = phase_name
        self.current_subtask = subtask
        self.progress_percentage = self.phases.get(phase_name, 0)
        
        # Update Streamlit session state
        if hasattr(st.session_state, 'research_phase'):
            st.session_state.research_phase = phase_name
            st.session_state.current_subtask = subtask
            st.session_state.progress_percentage = self.progress_percentage
            st.session_state.research_status = f"Running {phase_name}..."
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary."""
        return {
            "phase": self.current_phase,
            "subtask": self.current_subtask,
            "progress": self.progress_percentage,
            "status": f"Running {self.current_phase}..." if self.current_phase != "Waiting" else "Not started"
        }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Your specific research goal
DEFAULT_RQ = """
アンケートで得た収入カテゴリとレセプトデータを連結し、主要脆弱性骨折（大腿骨近位部、臨床椎体、上腕骨近位部、橈骨遠位部）時点で骨粗鬆症未診断かつ骨折前に骨粗鬆症治療歴のない50歳以上を対象に、骨折後6カ月以内の(1)骨粗鬆症の新規診断付与、(2)二次予防の抗骨粗鬆症薬（経口・注射を含む）の開始、ならびに骨折後12カ月時点での治療継続の有無が収入カテゴリによって異なるかを検証する。
"""

# Page config
st.set_page_config(
    page_title="CoScientist Research Launcher",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if 'research_started' not in st.session_state:
    st.session_state.research_started = False
if 'research_goal' not in st.session_state:
    st.session_state.research_goal = DEFAULT_RQ
if 'checkpoint_path' not in st.session_state:
    st.session_state.checkpoint_path = None
if 'research_status' not in st.session_state:
    st.session_state.research_status = "Not started"
if 'research_phase' not in st.session_state:
    st.session_state.research_phase = "Waiting"
if 'progress_percentage' not in st.session_state:
    st.session_state.progress_percentage = 0
if 'current_subtask' not in st.session_state:
    st.session_state.current_subtask = ""
if 'research_thread' not in st.session_state:
    st.session_state.research_thread = None
if 'research_complete' not in st.session_state:
    st.session_state.research_complete = False
if 'research_results' not in st.session_state:
    st.session_state.research_results = None
if 'research_error' not in st.session_state:
    st.session_state.research_error = None


def check_existing_research(goal: str):
    """Check if research already exists for this goal."""
    try:
        checkpoints = CoscientistState.list_checkpoints(goal=goal)
        return len(checkpoints) > 0, checkpoints
    except:
        return False, []


def check_research_progress():
    """Check the progress of running research and update UI accordingly."""
    if not st.session_state.get('research_thread'):
        return
    
    future = st.session_state.research_thread
    
    if future.done():
        # Research completed or failed
        try:
            result = future.result()
            if result and not st.session_state.get('research_error'):
                st.session_state.research_complete = True
                st.session_state.research_results = result
        except Exception as e:
            st.session_state.research_error = f"Research failed: {str(e)}"
        finally:
            st.session_state.research_thread = None
    
    # Update progress display
    if hasattr(st.session_state, 'progress_tracker'):
        status = st.session_state.progress_tracker.get_status()
        st.session_state.research_status = status['status']
        st.session_state.research_phase = status['phase']
        st.session_state.current_subtask = status['subtask']
        st.session_state.progress_percentage = status['progress']


def run_coscientist_background(goal: str, resume: bool, progress_tracker: StreamlitProgressTracker):
    """Run CoScientist in background thread with progress tracking."""
    
    try:
        progress_tracker.update_phase("Initializing", "Setting up research state")
        
        if resume:
            # Load existing state
            state = CoscientistState.load_latest(goal=goal)
            if not state:
                st.session_state.research_error = "Failed to load existing state"
                return None, None
        else:
            # Clear any existing research for this goal
            CoscientistState.clear_goal_directory(goal)
            
            # Create new state
            state = CoscientistState(goal=goal)
        
        # Create state manager
        state_manager = CoscientistStateManager(state)
        
        # Configure with debug mode for better visibility
        config = CoscientistConfig(
            literature_review_agent_llm=_SMARTER_LLM_POOL["claude-opus-4-1-20250805"],
            debug_mode=True,
            save_on_error=True,
            pause_after_literature_review=False  # Don't pause in Streamlit
        )
        
        # Create framework
        framework = CoscientistFramework(config, state_manager)
        
        # Create custom progress tracker for framework
        class FrameworkProgressCallback:
            def __init__(self, tracker):
                self.tracker = tracker
            
            def on_phase_start(self, phase_name, description=""):
                self.tracker.update_phase(phase_name, description)
        
        progress_callback = FrameworkProgressCallback(progress_tracker)
        
        # Update progress at different phases
        progress_tracker.update_phase("Literature Review", "Analyzing research goal and conducting literature search")
        
        # Run the framework - this will be the main research execution
        final_report, meta_review = asyncio.run(framework.run())
        
        # Mark as complete
        progress_tracker.update_phase("Final Report", "Research completed successfully")
        st.session_state.research_complete = True
        st.session_state.research_results = (final_report, meta_review)
        
        return final_report, meta_review
        
    except Exception as e:
        error_msg = f"Error during research: {str(e)}"
        st.session_state.research_error = error_msg
        progress_tracker.update_phase("Error", error_msg)
        
        # Try to save error checkpoint
        try:
            if 'state' in locals():
                checkpoint = state.save()
                logging.info(f"Error checkpoint saved: {checkpoint}")
        except Exception as save_error:
            logging.error(f"Failed to save error checkpoint: {save_error}")
        
        return None, None


def start_coscientist_nonblocking(goal: str, resume: bool = False):
    """Start CoScientist research in a non-blocking way."""
    
    # Create progress tracker
    progress_tracker = StreamlitProgressTracker()
    
    # Initialize session state for this research run
    st.session_state.research_started = True
    st.session_state.research_complete = False
    st.session_state.research_results = None
    st.session_state.research_error = None
    st.session_state.progress_tracker = progress_tracker
    
    # Start research in background thread
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run_coscientist_background, goal, resume, progress_tracker)
    st.session_state.research_thread = future
    
    return future


def main():
    # Auto-refresh every 2 seconds if research is running
    if (st.session_state.get('research_started') and 
        not st.session_state.get('research_complete') and 
        not st.session_state.get('research_error')):
        time.sleep(2)
        st.rerun()
    
    # Check research progress
    check_research_progress()
    st.title("🔬 CoScientist Research Launcher")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check environment variables
        st.subheader("API Keys Status")
        api_keys = {
            "OpenAI": "OPENAI_API_KEY" in os.environ,
            "Anthropic": "ANTHROPIC_API_KEY" in os.environ,
            "Google": "GOOGLE_API_KEY" in os.environ,
            "Tavily": "TAVILY_API_KEY" in os.environ,
        }
        
        for key, exists in api_keys.items():
            if exists:
                st.success(f"✅ {key} API key configured")
            else:
                st.error(f"❌ {key} API key missing")
        
        if not all(api_keys.values()):
            st.warning("Please configure all API keys in .env file")
        
        st.markdown("---")
        
        # Research options
        st.subheader("Research Options")
        max_subtopics = st.slider("Max Literature Subtopics", 2, 10, 5)
        n_hypotheses = st.slider("Initial Hypotheses", 4, 16, 8)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("1️⃣ Research Question (RQ) Confirmation")
        
        # Text area for research goal
        research_goal = st.text_area(
            "Enter your research question:",
            value=st.session_state.research_goal,
            height=150,
            help="This will be the goal for the CoScientist multi-agent system"
        )
        
        # Update session state
        if research_goal != st.session_state.research_goal:
            st.session_state.research_goal = research_goal
            st.session_state.research_started = False
        
        # Check for existing research
        has_existing, checkpoints = check_existing_research(research_goal)
        
        if has_existing:
            st.warning(f"⚠️ Found {len(checkpoints)} existing checkpoint(s) for this research goal")
            col_new, col_resume = st.columns(2)
            
            with col_new:
                if st.button("🆕 Start Fresh Research", type="secondary", use_container_width=True):
                    start_coscientist_nonblocking(research_goal, resume=False)
            
            with col_resume:
                if st.button("📂 Resume from Checkpoint", type="primary", use_container_width=True):
                    start_coscientist_nonblocking(research_goal, resume=True)
        else:
            if st.button("✅ Confirm RQ and Start CoScientist", type="primary", use_container_width=True):
                start_coscientist_nonblocking(research_goal, resume=False)
    
    with col2:
        st.header("📊 Status")
        
        # Status display
        status_container = st.container()
        with status_container:
            if st.session_state.research_started:
                st.info(f"Status: {st.session_state.research_status}")
            else:
                st.info("Status: Waiting for RQ confirmation")
            
            # Display current goal hash
            if research_goal:
                goal_hash = CoscientistState._hash_goal(research_goal)
                st.text(f"Goal ID: {goal_hash}")
    
    # Divider
    st.markdown("---")
    
    # Research execution area
    if st.session_state.research_started:
        st.header("2️⃣ CoScientist Research Execution")
        
        # Create placeholder for dynamic updates
        research_container = st.container()
        
            if final_report:
                st.success("✅ Research completed successfully!")
                
                # Display results in tabs
                tab1, tab2, tab3 = st.tabs(["📄 Final Report", "🔍 Meta Review", "💾 Checkpoints"])
                
                with tab1:
                    st.subheader("Final Research Report")
                    if isinstance(final_report, dict):
                        st.markdown(final_report.get('content', str(final_report)))
                    else:
                        st.markdown(str(final_report))
                
                with tab2:
                    st.subheader("Meta Review")
                    if meta_review:
                        if isinstance(meta_review, dict):
                            st.markdown(meta_review.get('content', str(meta_review)))
                        else:
                            st.markdown(str(meta_review))
                
                with tab3:
                    st.subheader("Saved Checkpoints")
                    checkpoints = CoscientistState.list_checkpoints(goal=st.session_state.research_goal)
                    for cp in checkpoints[:10]:  # Show last 10
                        st.text(f"📁 {Path(cp).name}")
            
            elif st.session_state.research_status.startswith("Error"):
                st.error("Research failed. Check logs for details.")
                
                # Show option to retry
                if st.button("🔄 Retry Research"):
                    st.session_state.research_started = True
                    st.rerun()
    
    else:
        st.info("👆 Please confirm your research question above to start CoScientist")
        
        # Show example workflow
        with st.expander("📖 How it works"):
            st.markdown("""
            ### CoScientist Research Workflow
            
            1. **Literature Review**: Decomposes your RQ into subtopics and conducts comprehensive research
            2. **Hypothesis Generation**: Multiple agents generate diverse hypotheses
            3. **Reflection & Review**: Deep verification and analysis of hypotheses
            4. **Tournament Ranking**: ELO-based competition to rank hypotheses
            5. **Evolution**: Refinement based on feedback
            6. **Meta Review**: Synthesis of insights
            7. **Final Report**: Comprehensive research summary
            
            The system automatically saves checkpoints throughout the process, allowing you to resume if interrupted.
            """)


if __name__ == "__main__":
    main()