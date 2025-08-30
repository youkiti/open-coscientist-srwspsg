import asyncio
import os
import traceback
from datetime import datetime
from pathlib import Path

# Ensure environment variables are loaded in child process
from dotenv import load_dotenv
load_dotenv()

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager


def _get_done_file_path(goal: str) -> str:
    """Gets the path for the 'done' file for a given goal."""
    goal_hash = CoscientistState._hash_goal(goal)
    # This assumes _OUTPUT_DIR is consistent.
    output_dir = os.path.join(
        os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
        goal_hash,
    )
    return os.path.join(output_dir, "done.txt")


def coscientist_process_target(goal: str):
    """The target function for the multiprocessing.Process."""
    goal_hash = CoscientistState._hash_goal(goal)
    output_dir = os.path.join(
        os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
        goal_hash,
    )
    
    # Create output directory early
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed log file
    log_file = os.path.join(output_dir, "process.log")
    
    try:
        with open(log_file, "w") as log:
            log.write(f"[{datetime.now()}] Starting Coscientist process for goal: {goal}\n")
            log.write(f"[{datetime.now()}] Output directory: {output_dir}\n")
            log.write(f"[{datetime.now()}] Goal hash: {goal_hash}\n")
            
            # Check environment variables
            api_keys = {
                'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', 'NOT_SET')[:10] + '...',
                'ANTHROPIC_API_KEY': os.environ.get('ANTHROPIC_API_KEY', 'NOT_SET')[:10] + '...',
                'TAVILY_API_KEY': os.environ.get('TAVILY_API_KEY', 'NOT_SET')[:10] + '...',
            }
            log.write(f"[{datetime.now()}] Environment variables: {api_keys}\n")
            
            # Initialize state and config with retry logic for FileExistsError
            log.write(f"[{datetime.now()}] Initializing CoscientistState with retry logic...\n")
            
            max_retries = 3
            initial_state = None
            
            for attempt in range(max_retries):
                try:
                    log.write(f"[{datetime.now()}] Attempt {attempt + 1}: Creating CoscientistState...\n")
                    
                    # Try to load existing state first
                    try:
                        initial_state = CoscientistState.load_latest(goal=goal)
                        if initial_state:
                            log.write(f"[{datetime.now()}] Loaded existing state for goal\n")
                            break
                    except Exception as load_error:
                        log.write(f"[{datetime.now()}] No existing state found: {load_error}\n")
                    
                    # If no existing state, create new one
                    # First clear any existing directory to prevent FileExistsError
                    log.write(f"[{datetime.now()}] Clearing goal directory before creating new state...\n")
                    try:
                        clear_result = CoscientistState.clear_goal_directory(goal)
                        log.write(f"[{datetime.now()}] Directory cleanup result: {clear_result}\n")
                    except Exception as clear_error:
                        log.write(f"[{datetime.now()}] Directory cleanup failed (may not exist): {clear_error}\n")
                    
                    # Small delay to ensure filesystem operations complete
                    import time
                    time.sleep(0.5)
                    
                    initial_state = CoscientistState(goal=goal)
                    log.write(f"[{datetime.now()}] Successfully created new CoscientistState\n")
                    break
                    
                except FileExistsError as fee:
                    log.write(f"[{datetime.now()}] FileExistsError on attempt {attempt + 1}: {fee}\n")
                    if attempt < max_retries - 1:
                        # Force cleanup and retry
                        try:
                            log.write(f"[{datetime.now()}] Force clearing directory for retry...\n")
                            CoscientistState.clear_goal_directory(goal)
                            time.sleep(1.0)  # Longer delay for retry
                        except Exception as cleanup_error:
                            log.write(f"[{datetime.now()}] Force cleanup failed: {cleanup_error}\n")
                    else:
                        log.write(f"[{datetime.now()}] All retry attempts exhausted\n")
                        raise
                except Exception as e:
                    log.write(f"[{datetime.now()}] Unexpected error on attempt {attempt + 1}: {e}\n")
                    if attempt == max_retries - 1:
                        raise
            
            if initial_state is None:
                raise RuntimeError("Failed to create or load CoscientistState after all retry attempts")
            
            log.write(f"[{datetime.now()}] Creating CoscientistConfig...\n")
            config = CoscientistConfig()
            
            log.write(f"[{datetime.now()}] Creating CoscientistStateManager...\n")
            state_manager = CoscientistStateManager(initial_state)
            
            log.write(f"[{datetime.now()}] Creating CoscientistFramework...\n")
            cosci = CoscientistFramework(config, state_manager)
            
            log.write(f"[{datetime.now()}] Starting research framework...\n")
            # Run the framework
            asyncio.run(cosci.run())
            
            log.write(f"[{datetime.now()}] Research framework completed successfully\n")

    except Exception as e:
        error_msg = f"Error: {str(e)}\nTraceback:\n{traceback.format_exc()}"
        
        # Write detailed error log
        try:
            with open(log_file, "a") as log:
                log.write(f"[{datetime.now()}] CRITICAL ERROR:\n{error_msg}\n")
        except:
            pass  # Fallback if log file can't be written
            
        # Write error.log for compatibility
        try:
            with open(os.path.join(output_dir, "error.log"), "w") as f:
                f.write(error_msg)
        except Exception as log_error:
            # Last resort - try to write to a different location
            try:
                with open(os.path.expanduser("~/coscientist_error.log"), "w") as f:
                    f.write(f"Original error: {error_msg}\nLog write error: {log_error}")
            except:
                pass  # Give up on logging
                
    finally:
        # Create a "done" file to signal completion
        try:
            done_file = _get_done_file_path(goal)
            os.makedirs(os.path.dirname(done_file), exist_ok=True)
            with open(done_file, "w") as f:
                f.write(f"completed at {datetime.now()}")
            
            # Also log completion
            try:
                with open(log_file, "a") as log:
                    log.write(f"[{datetime.now()}] Process completed, done file created\n")
            except:
                pass
        except Exception as e:
            # Try to create emergency done file
            try:
                emergency_done = os.path.expanduser(f"~/coscientist_done_{goal_hash}.txt")
                with open(emergency_done, "w") as f:
                    f.write(f"Emergency completion marker: {datetime.now()}\nError creating done file: {e}")
            except:
                pass  # Ultimate fallback


def check_coscientist_status(goal: str) -> str:
    """Checks the status of a Coscientist run."""
    goal_hash = CoscientistState._hash_goal(goal)
    output_dir = os.path.join(
        os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
        goal_hash,
    )

    done_file = os.path.join(output_dir, "done.txt")
    error_file = os.path.join(output_dir, "error.log")

    if os.path.exists(done_file):
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                error_message = f.read()
            return f"error: {error_message}"
        return "done"
    return "running"


def get_coscientist_results(goal: str) -> tuple[str, str]:
    """Gets the results from a completed Coscientist run."""
    state = CoscientistState.load_latest(goal=goal)
    if state and state.final_report and state.meta_reviews:
        # These are TypedDicts, access by key.
        final_report_text = state.final_report.get(
            "result", "Final report not generated."
        )
        meta_review_text = state.meta_reviews[-1].get(
            "result", "Meta review not generated."
        )
        return final_report_text, meta_review_text
    return "Results not found.", "Results not found."


def cleanup_coscientist_run(goal: str):
    """Cleans up files after a run."""
    goal_hash = CoscientistState._hash_goal(goal)
    output_dir = os.path.join(
        os.environ.get("COSCIENTIST_DIR", os.path.expanduser("~/.coscientist")),
        goal_hash,
    )
    done_file = os.path.join(output_dir, "done.txt")
    error_file = os.path.join(output_dir, "error.log")
    if os.path.exists(done_file):
        os.remove(done_file)
    if os.path.exists(error_file):
        os.remove(error_file)
