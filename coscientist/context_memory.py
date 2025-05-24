"""
Context memory
-------------
- A memory of an Agent's prior work
- Enables storage and retrieval of past states
- Supports persistence across sessions
- Tracks research progress and agent performance

"""

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from coscientist.custom_types import HypothesisWithID, ResearchPlanConfig
from coscientist.ranking_agent import EloTournament


class ContextMemory:
    """
    Persistent memory system for storing and retrieving research states.
    """

    def __init__(self, db_path: str = "coscientist_memory.db"):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Research sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    research_config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active'
                )
            """)

            # Hypotheses table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id INTEGER PRIMARY KEY,
                    session_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    review TEXT,
                    elo_rating REAL DEFAULT 1200.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    agent_type TEXT,
                    FOREIGN KEY (session_id) REFERENCES research_sessions (id)
                )
            """)

            # Agent states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    agent_type TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    iteration INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES research_sessions (id)
                )
            """)

            # Tasks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    session_id INTEGER NOT NULL,
                    agent_type TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    parameters TEXT,
                    priority INTEGER,
                    status TEXT DEFAULT 'pending',
                    result TEXT,
                    error TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES research_sessions (id)
                )
            """)

            # Tournament matches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tournament_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    hypothesis_1_id INTEGER,
                    hypothesis_2_id INTEGER,
                    winner INTEGER,
                    debate_transcript TEXT,
                    match_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES research_sessions (id),
                    FOREIGN KEY (hypothesis_1_id) REFERENCES hypotheses (id),
                    FOREIGN KEY (hypothesis_2_id) REFERENCES hypotheses (id)
                )
            """)

            # Performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    agent_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    iteration INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES research_sessions (id)
                )
            """)

            conn.commit()

    def create_session(self, goal: str, research_config: ResearchPlanConfig) -> int:
        """
        Create a new research session.

        Parameters
        ----------
        goal: str
            Research goal
        research_config: ResearchPlanConfig
            Research configuration

        Returns
        -------
        int
            Session ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO research_sessions (goal, research_config)
                VALUES (?, ?)
            """,
                (
                    goal,
                    json.dumps(
                        asdict(research_config)
                        if hasattr(research_config, "__dataclass_fields__")
                        else dict(research_config)
                    ),
                ),
            )

            return cursor.lastrowid

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a research session.

        Parameters
        ----------
        session_id: int
            Session ID

        Returns
        -------
        Optional[Dict[str, Any]]
            Session data or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, goal, research_config, created_at, updated_at, status
                FROM research_sessions WHERE id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "goal": row[1],
                    "research_config": json.loads(row[2]),
                    "created_at": row[3],
                    "updated_at": row[4],
                    "status": row[5],
                }
            return None

    def store_hypothesis(
        self,
        session_id: int,
        hypothesis: HypothesisWithID,
        agent_type: str = "unknown",
        elo_rating: float = 1200.0,
    ):
        """
        Store a hypothesis in memory.

        Parameters
        ----------
        session_id: int
            Session ID
        hypothesis: HypothesisWithID
            Hypothesis to store
        agent_type: str
            Type of agent that generated the hypothesis
        elo_rating: float
            Current ELO rating
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO hypotheses 
                (id, session_id, content, review, elo_rating, agent_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    hypothesis.id,
                    session_id,
                    hypothesis.content,
                    hypothesis.review,
                    elo_rating,
                    agent_type,
                ),
            )
            conn.commit()

    def get_hypotheses(self, session_id: int) -> List[HypothesisWithID]:
        """
        Retrieve all hypotheses for a session.

        Parameters
        ----------
        session_id: int
            Session ID

        Returns
        -------
        List[HypothesisWithID]
            List of hypotheses
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, content, review FROM hypotheses 
                WHERE session_id = ? ORDER BY created_at
            """,
                (session_id,),
            )

            hypotheses = []
            for row in cursor.fetchall():
                hypotheses.append(
                    HypothesisWithID(id=row[0], content=row[1], review=row[2] or "")
                )

            return hypotheses

    def store_agent_state(
        self,
        session_id: int,
        agent_type: str,
        state_data: Dict[str, Any],
        iteration: int,
    ):
        """
        Store agent state.

        Parameters
        ----------
        session_id: int
            Session ID
        agent_type: str
            Type of agent
        state_data: Dict[str, Any]
            Agent state data
        iteration: int
            Current iteration
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO agent_states (session_id, agent_type, state_data, iteration)
                VALUES (?, ?, ?, ?)
            """,
                (session_id, agent_type, json.dumps(state_data), iteration),
            )
            conn.commit()

    def get_agent_state(
        self, session_id: int, agent_type: str, iteration: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve agent state.

        Parameters
        ----------
        session_id: int
            Session ID
        agent_type: str
            Type of agent
        iteration: Optional[int]
            Specific iteration (latest if None)

        Returns
        -------
        Optional[Dict[str, Any]]
            Agent state or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if iteration is not None:
                cursor.execute(
                    """
                    SELECT state_data FROM agent_states 
                    WHERE session_id = ? AND agent_type = ? AND iteration = ?
                """,
                    (session_id, agent_type, iteration),
                )
            else:
                cursor.execute(
                    """
                    SELECT state_data FROM agent_states 
                    WHERE session_id = ? AND agent_type = ?
                    ORDER BY created_at DESC LIMIT 1
                """,
                    (session_id, agent_type),
                )

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None

    def store_task(self, session_id: int, task_data: Dict[str, Any]):
        """
        Store task information.

        Parameters
        ----------
        session_id: int
            Session ID
        task_data: Dict[str, Any]
            Task data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO tasks 
                (id, session_id, agent_type, task_type, parameters, priority, 
                 status, result, error, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    task_data.get("id"),
                    session_id,
                    task_data.get("agent_type"),
                    task_data.get("task_type"),
                    json.dumps(task_data.get("parameters", {})),
                    task_data.get("priority"),
                    task_data.get("status"),
                    json.dumps(task_data.get("result")),
                    task_data.get("error"),
                    task_data.get("started_at"),
                    task_data.get("completed_at"),
                ),
            )
            conn.commit()

    def get_tasks(
        self, session_id: int, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tasks for a session.

        Parameters
        ----------
        session_id: int
            Session ID
        status: Optional[str]
            Filter by status

        Returns
        -------
        List[Dict[str, Any]]
            List of tasks
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if status:
                cursor.execute(
                    """
                    SELECT * FROM tasks WHERE session_id = ? AND status = ?
                    ORDER BY created_at
                """,
                    (session_id, status),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM tasks WHERE session_id = ?
                    ORDER BY created_at
                """,
                    (session_id,),
                )

            tasks = []
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                task = dict(zip(columns, row))
                task["parameters"] = (
                    json.loads(task["parameters"]) if task["parameters"] else {}
                )
                task["result"] = json.loads(task["result"]) if task["result"] else None
                tasks.append(task)

            return tasks

    def store_tournament_match(
        self,
        session_id: int,
        hypothesis_1_id: int,
        hypothesis_2_id: int,
        winner: int,
        debate_transcript: str,
        match_type: str = "tournament",
    ):
        """
        Store tournament match result.

        Parameters
        ----------
        session_id: int
            Session ID
        hypothesis_1_id: int
            First hypothesis ID
        hypothesis_2_id: int
            Second hypothesis ID
        winner: int
            Winner (1 or 2)
        debate_transcript: str
            Debate transcript
        match_type: str
            Type of match
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO tournament_matches 
                (session_id, hypothesis_1_id, hypothesis_2_id, winner, 
                 debate_transcript, match_type)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    hypothesis_1_id,
                    hypothesis_2_id,
                    winner,
                    debate_transcript,
                    match_type,
                ),
            )
            conn.commit()

    def get_tournament_matches(self, session_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve tournament matches for a session.

        Parameters
        ----------
        session_id: int
            Session ID

        Returns
        -------
        List[Dict[str, Any]]
            List of matches
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM tournament_matches WHERE session_id = ?
                ORDER BY created_at
            """,
                (session_id,),
            )

            matches = []
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                matches.append(dict(zip(columns, row)))

            return matches

    def store_performance_metric(
        self,
        session_id: int,
        agent_type: str,
        metric_name: str,
        metric_value: float,
        iteration: int,
    ):
        """
        Store performance metric.

        Parameters
        ----------
        session_id: int
            Session ID
        agent_type: str
            Type of agent
        metric_name: str
            Name of metric
        metric_value: float
            Metric value
        iteration: int
            Current iteration
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO performance_metrics 
                (session_id, agent_type, metric_name, metric_value, iteration)
                VALUES (?, ?, ?, ?, ?)
            """,
                (session_id, agent_type, metric_name, metric_value, iteration),
            )
            conn.commit()

    def get_performance_metrics(
        self, session_id: int, agent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve performance metrics.

        Parameters
        ----------
        session_id: int
            Session ID
        agent_type: Optional[str]
            Filter by agent type

        Returns
        -------
        List[Dict[str, Any]]
            List of metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if agent_type:
                cursor.execute(
                    """
                    SELECT * FROM performance_metrics 
                    WHERE session_id = ? AND agent_type = ?
                    ORDER BY iteration, created_at
                """,
                    (session_id, agent_type),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM performance_metrics WHERE session_id = ?
                    ORDER BY iteration, created_at
                """,
                    (session_id,),
                )

            metrics = []
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                metrics.append(dict(zip(columns, row)))

            return metrics

    def get_session_summary(self, session_id: int) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a research session.

        Parameters
        ----------
        session_id: int
            Session ID

        Returns
        -------
        Dict[str, Any]
            Session summary
        """
        session = self.get_session(session_id)
        if not session:
            return {}

        hypotheses = self.get_hypotheses(session_id)
        tasks = self.get_tasks(session_id)
        matches = self.get_tournament_matches(session_id)
        metrics = self.get_performance_metrics(session_id)

        # Calculate statistics
        task_stats = {}
        for task in tasks:
            status = task["status"]
            task_stats[status] = task_stats.get(status, 0) + 1

        return {
            "session": session,
            "statistics": {
                "total_hypotheses": len(hypotheses),
                "total_tasks": len(tasks),
                "total_matches": len(matches),
                "task_breakdown": task_stats,
            },
            "hypotheses": hypotheses,
            "recent_tasks": tasks[-10:],  # Last 10 tasks
            "performance_metrics": metrics,
        }

    def cleanup_old_sessions(self, days_old: int = 30):
        """
        Clean up sessions older than specified days.

        Parameters
        ----------
        days_old: int
            Days threshold for cleanup
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM research_sessions 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old)
            )
            conn.commit()


# Global context memory instance
_context_memory = None


def get_context_memory(db_path: str = "coscientist_memory.db") -> ContextMemory:
    """Get the global context memory instance."""
    global _context_memory
    if _context_memory is None:
        _context_memory = ContextMemory(db_path)
    return _context_memory
