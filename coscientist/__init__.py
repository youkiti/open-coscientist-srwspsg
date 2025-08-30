"""Open CoScientist Agents - Multi-agent system for AI co-scientist research."""

__version__ = "0.0.1"

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logging():
    """Configure comprehensive logging for the Coscientist framework."""
    
    # Create log directory if it doesn't exist
    log_dir = Path(__file__).parent.parent / "log"
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define log format with phase tracking
    log_format = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    detailed_format = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Remove any existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Create main log file handler
    main_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"main_{timestamp}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    main_handler.setLevel(logging.INFO)
    main_handler.setFormatter(log_format)
    root_logger.addHandler(main_handler)
    
    # Create debug log file handler with more detail
    debug_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"debug_{timestamp}.log",
        maxBytes=20*1024*1024,  # 20MB
        backupCount=3
    )
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(detailed_format)
    root_logger.addHandler(debug_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(log_format)
    root_logger.addHandler(console_handler)
    
    # Create specialized loggers for different components
    loggers_config = {
        'coscientist.framework': logging.INFO,
        'coscientist.literature_review': logging.DEBUG,
        'coscientist.generation': logging.INFO,
        'coscientist.reflection': logging.INFO,
        'coscientist.tournament': logging.INFO,
        'coscientist.evolution': logging.INFO,
        'coscientist.supervisor': logging.INFO,
        'coscientist.gpt_researcher': logging.DEBUG,
        'coscientist.progress': logging.INFO,
    }
    
    for logger_name, level in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        
        # Create component-specific log file
        component_name = logger_name.split('.')[-1]
        component_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{component_name}_{timestamp}.log",
            maxBytes=10*1024*1024,
            backupCount=3
        )
        component_handler.setLevel(level)
        component_handler.setFormatter(log_format)
        logger.addHandler(component_handler)
        logger.propagate = True  # Also send to root logger
    
    # Log initialization
    logging.info(f"Logging initialized - Session: {timestamp}")
    logging.info(f"Log directory: {log_dir}")
    logging.debug(f"Python environment: {os.environ.get('VIRTUAL_ENV', 'No virtual env')}")
    
    return timestamp

# Initialize logging when module is imported
SESSION_ID = setup_logging()

from coscientist.framework import CoscientistConfig, CoscientistFramework
from coscientist.global_state import CoscientistState, CoscientistStateManager
