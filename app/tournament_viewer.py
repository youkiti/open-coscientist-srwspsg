import os
import multiprocessing

import streamlit as st

# Import from modular pages
from common import (
    get_available_states,
    load_coscientist_state,
    load_coscientist_state_by_goal,
    load_last_confirmed_goal,
)
from background import (
    coscientist_process_target,
    check_coscientist_status,
    cleanup_coscientist_run,
)
from coscientist.global_state import CoscientistState
from configuration_page import display_configuration_page
from final_report_page import display_final_report_page
from literature_review_page import display_literature_review_page
from meta_reviews_page import display_meta_reviews_page
from progress_page import display_progress_page
from proximity_page import display_proximity_graph_page
from resume_page import display_resume_page
from supervisor_page import display_supervisor_page
from tournament_page import display_tournament_page

st.set_page_config(page_title="Coscientist Viewer", page_icon="🧪", layout="wide")

# Sidebar navigation
st.sidebar.title("🧪 Coscientist Viewer")
page = st.sidebar.selectbox(
    "Select Page",
    [
        "Configuration Agent",
        "Live Progress Monitor",
        "Literature Review",
        "Tournament Rankings",
        "Proximity Graph",
        "Meta-Reviews",
        "Supervisor Decisions",
        "Final Report",
        "Resume from Checkpoint",
    ],
)


def main():
    st.title("🧪 Coscientist Viewer")

    # --- Quick Test Run (uses last confirmed CQ/refined goal) ---
    if "test_run_running" not in st.session_state:
        st.session_state.test_run_running = False
    if "test_run_process" not in st.session_state:
        st.session_state.test_run_process = None
    if "test_run_goal" not in st.session_state:
        st.session_state.test_run_goal = None
    if "test_run_error" not in st.session_state:
        st.session_state.test_run_error = None

    with st.container():
        st.markdown("### 🧪 Start Test Run")
        st.caption("Runs agents assuming the last confirmed CQ (refined goal).")

        # Determine default goal for test run
        default_goal = load_last_confirmed_goal()
        if not default_goal:
            # Fallback to current selection if available
            default_goal = st.session_state.get("current_file")

        col_a, col_b = st.columns([1, 3])
        with col_a:
            start_clicked = st.button("▶️ Start Test Run", type="primary")
        with col_b:
            st.write(
                f"Default goal: {default_goal if default_goal else 'Not found. Select a goal in the sidebar.'}"
            )

        if start_clicked and not st.session_state.test_run_running:
            if not default_goal:
                st.warning(
                    "No confirmed goal found. Complete Configuration or select a goal in the sidebar."
                )
            else:
                try:
                    process = multiprocessing.Process(
                        target=coscientist_process_target, args=(default_goal,)
                    )
                    process.start()
                    st.session_state.test_run_process = process
                    st.session_state.test_run_running = True
                    st.session_state.test_run_goal = default_goal
                    st.success(f"Test run started (PID: {process.pid})")
                    st.rerun()
                except Exception as e:
                    st.session_state.test_run_error = str(e)
                    st.error(f"Failed to start test run: {e}")

        # Status/Control panel for active test run
        if st.session_state.test_run_running and st.session_state.test_run_goal:
            goal = st.session_state.test_run_goal
            process = st.session_state.test_run_process
            st.info(
                f"Running for goal: {goal} — PID: {process.pid if process else 'N/A'} — Alive: {process.is_alive() if process else 'N/A'}"
            )

            # Show last few log lines if available
            goal_hash = CoscientistState._hash_goal(goal)
            output_dir = os.path.join(
                os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
                goal_hash,
            )
            log_file = os.path.join(output_dir, "process.log")
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            st.text("Recent log entries:")
                            for line in lines[-5:]:
                                st.text(line.strip())
                except Exception as e:
                    st.warning(f"Could not read log file: {e}")

            # Check status controls
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔄 Refresh"):
                    st.rerun()
            with col2:
                if st.button("⏹ Stop Test Run"):
                    try:
                        if process and process.is_alive():
                            process.terminate()
                        cleanup_coscientist_run(goal)
                    except Exception:
                        pass
                    st.session_state.test_run_running = False
                    st.session_state.test_run_process = None
                    st.session_state.test_run_goal = None
                    st.rerun()
            with col3:
                # Show background status
                status = check_coscientist_status(goal)
                if status == "done":
                    st.success("Test run completed.")
                    st.session_state.test_run_running = False
                    st.session_state.test_run_process = None
                    st.session_state.test_run_goal = None
                elif status.startswith("error:"):
                    st.error(f"Test run failed: {status.replace('error: ', '')}")
                    st.session_state.test_run_running = False
                    st.session_state.test_run_process = None
                    st.session_state.test_run_goal = None
                else:
                    st.info("Status: running")

    # Initialize session state for file selection
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "file_list_updated" not in st.session_state:
        st.session_state.file_list_updated = False

    # Variables for file handling
    selected_file = None
    temp_path = None
    state = None

    # Sidebar for file selection (only for pages that need state files)
    if page in [
        "Literature Review",
        "Tournament Rankings",
        "Proximity Graph",
        "Meta-Reviews",
        "Supervisor Decisions",
        "Final Report",
    ]:
        with st.sidebar:
            st.header("📁 Select Research Goal")

            # Update button
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Available Goals:**")
            with col2:
                if st.button("Update", help="Refresh file list and load latest"):
                    st.session_state.file_list_updated = True
                    st.rerun()

            # Get available state files
            available_states = get_available_states()

            # Auto-select most recent file if update was clicked or no file is selected
            if (
                st.session_state.file_list_updated
                or st.session_state.current_file is None
            ):
                if available_states:
                    st.session_state.current_file = available_states[
                        0
                    ]  # Most recent file
                    if st.session_state.file_list_updated:
                        st.success(
                            f"📁 Updated! Latest goal: {available_states[0][:50]}{'...' if len(available_states[0]) > 50 else ''}"
                        )
                st.session_state.file_list_updated = False

            if available_states:
                # Find index of current file in the list (in case files changed)
                current_index = 0
                if st.session_state.current_file in available_states:
                    current_index = available_states.index(
                        st.session_state.current_file
                    )

                selected_file = st.selectbox(
                    "Choose a research goal:",
                    options=available_states,
                    format_func=lambda x: x,  # Display the goal text directly
                    index=current_index,
                    key="file_selector",
                )

                # Update session state when selection changes
                if selected_file != st.session_state.current_file:
                    st.session_state.current_file = selected_file
            else:
                st.warning("No Coscientist research goals found.")
                selected_file = None
                st.session_state.current_file = None

            # File upload option
            st.markdown("**Or upload a file:**")
            uploaded_file = st.file_uploader("Upload .pkl file", type="pkl")

            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                selected_file = temp_path

        # Load state if goal is selected
        if selected_file:
            # If it's a temp file (uploaded), use the original function
            if selected_file.startswith("temp_"):
                state = load_coscientist_state(selected_file)
            else:
                # It's a goal text, use the new function
                state = load_coscientist_state_by_goal(selected_file)

    # Display appropriate page based on navigation
    if page == "Configuration Agent":
        display_configuration_page()
    elif page == "Live Progress Monitor":
        display_progress_page()
    elif page == "Literature Review":
        if state is None:
            st.info(
                "👈 Please select a research goal or upload a Coscientist state file from the sidebar to get started."
            )
            st.markdown("""
            ## Literature Review Page
            
            View the comprehensive literature review that forms the foundation of the research:
            
            1. **Research Subtopics** - see how the main research goal was systematically decomposed
            2. **Subtopic Reports** - select any subtopic to view its detailed literature analysis
            3. **Knowledge Foundation** - understand the scientific background informing hypothesis generation
            
            **What you'll see:**
            - **Dropdown Navigation**: Select from numbered subtopics to explore different research areas
            - **Detailed Reports**: Comprehensive literature analysis for each subtopic
            - **Research Context**: Scientific foundation that guides hypothesis generation
            - **Summary Statistics**: Overview of subtopics covered and reports available
            
            The literature review is one of the first steps in the research process, providing
            the scientific foundation for generating well-informed, evidence-based research hypotheses.
            """)
        else:
            display_literature_review_page(state)
    elif page == "Tournament Rankings":
        if state is None:
            st.info(
                "👈 Please select a research goal or upload a Coscientist state file from the sidebar to get started."
            )
            st.markdown("""
            ## Tournament Rankings Page
            
            View and explore hypotheses ranked by ELO rating:
            
            1. **Browse tournament rankings** - see all hypotheses ranked by ELO rating
            2. **Select a hypothesis** for detailed view to see:
               - Full hypothesis text and predictions
               - Causal reasoning and verification results
               - Assumptions and supporting research
               - Complete match history with debate transcripts
            
            **What you'll see:**
            - **ELO Ratings**: Higher scores indicate stronger performance in head-to-head comparisons
            - **Win-Loss Records**: Track record against other hypotheses
            - **Match History**: Full debate transcripts showing why one hypothesis beat another
            - **Hypothesis Lineage**: See which hypotheses evolved from others
            """)
        else:
            display_tournament_page(state)
    elif page == "Proximity Graph":
        if state is None:
            st.info(
                "👈 Please select a research goal or upload a Coscientist state file from the sidebar to get started."
            )
            st.markdown("""
            ## Proximity Graph Page
            
            Explore the semantic relationships between hypotheses using advanced network visualization:
            
            1. **Interactive Cytoscape.js graph** with hypotheses as nodes and similarities as edges
            2. **Community detection** to find groups of semantically similar hypotheses
            3. **Click nodes** to select them and see full hypothesis text
            4. **Drag and rearrange** nodes to explore relationships
            5. **Adjust parameters** to control community detection sensitivity
            
            **What you'll see:**
            - **Node colors**: Different colors represent different semantic communities
            - **Interactive layout**: Force-directed positioning based on similarity
            - **Edges**: Connections show cosine similarity between hypothesis embeddings
            - **Statistics**: Number of hypotheses, connections, and average similarity
            - **Selection feedback**: Click nodes to see their full hypothesis text below the graph
            
            **Advanced Features:**
            - Multi-node selection with Ctrl/Cmd + click
            - Smooth animations and transitions
            - Professional network graph layout algorithms
            - Real-time interaction feedback
            """)
        else:
            display_proximity_graph_page(state)
    elif page == "Meta-Reviews":
        if state is None:
            st.info(
                "👈 Please select a research goal or upload a Coscientist state file from the sidebar to get started."
            )
            st.markdown("""
            ## Meta-Reviews Page
            
            View the strategic analysis and review process of the research:
            
            1. **Reviews Timeline** - see all meta-reviews generated during the research process
            2. **Strategic Analysis** - click on any meta-review to see the full analysis
            3. **Research Guidance** - understand how each review guided future research directions
            
            **What you'll see:**
            - **Numbered Reviews**: Latest meta-reviews appear first with sequential numbering
            - **Strategic Analysis**: Full text of the meta-review analysis and insights
            - **Research Context**: Tournament state and hypothesis counts at review time
            - **Quality Assessment**: Evaluation of hypothesis performance and research progress
            
            Meta-reviews are generated periodically to analyze the current state of research,
            identify patterns and gaps, and guide the supervisor agent's strategic decisions.
            """)
        else:
            display_meta_reviews_page(state)
    elif page == "Supervisor Decisions":
        if state is None:
            st.info(
                "👈 Please select a research goal or upload a Coscientist state file from the sidebar to get started."
            )
            st.markdown("""
            ## Supervisor Decisions Page
            
            View the decision-making process of the supervisor agent:
            
            1. **Actions Timeline** - see all actions taken by the supervisor in chronological order
            2. **Decision Reasoning** - click on any action to see the detailed reasoning behind it
            3. **System Context** - understand the research state that influenced each decision
            
            **What you'll see:**
            - **Numbered Actions**: Latest actions appear first with sequential numbering
            - **Decision Reasoning**: Full text of the supervisor's strategic thinking
            - **System Metrics**: Research state, hypothesis counts, tournament progress at decision time
            - **Recent Context**: What other actions were taken recently that influenced the decision
            
            The supervisor agent analyzes the research progress and decides what actions to take next,
            such as generating new hypotheses, evolving existing ones, running tournaments, or finishing the research.
            """)
        else:
            display_supervisor_page(state)
    elif page == "Final Report":
        if state is None:
            st.info(
                "👈 Please select a research goal or upload a Coscientist state file from the sidebar to get started."
            )
            st.markdown("""
            ## Final Report Page
            
            View the comprehensive final research report generated upon completion:
            
            1. **Complete Analysis** - comprehensive summary of all research findings
            2. **Top Hypotheses** - detailed review of the highest-ranked hypotheses  
            3. **Research Conclusions** - final insights and recommendations
            4. **Process Summary** - overview of the research methodology and evaluation
            
            **What you'll see:**
            - **Final Report**: Complete research summary and conclusions
            - **Research Statistics**: Overview of hypotheses generated, tournaments run, and key metrics
            - **Process Completion**: Confirmation that the research process finished successfully
            
            The final report is generated only when the supervisor agent determines that the research
            has achieved sufficient depth and quality, and further investigation would yield diminishing returns.
            """)
        else:
            display_final_report_page(state)
    elif page == "Resume from Checkpoint":
        display_resume_page()

    # Clean up temp file if it was uploaded
    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)


if __name__ == "__main__":
    main()
