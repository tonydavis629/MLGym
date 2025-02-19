"""
Copyright (c) Meta Platforms, Inc. and affiliates.

MLGym Trajectory Visualizer

This module provides a Streamlit-based web application for visualizing MLGym trajectories.
It allows users to inspect step-by-step progression of agents through various ML tasks,
including their thought processes, actions taken, and execution results.

Usage:
    streamlit run trajectory_visualizer.py [--trajectory_dir PATH]
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import streamlit as st


def configure_page_style() -> None:
    """Configure the Streamlit page layout and apply custom CSS styling."""
    st.set_page_config(
        page_title="MLGym Demo",
        page_icon="👩‍🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
    <style>
        /* Base colors */
        :root {
            --slate-50: #f8fafc;
            --slate-100: #f1f5f9;
            --slate-200: #e2e8f0;
            --slate-300: #cbd5e1;
            --slate-400: #94a3b8;
            --slate-500: #64748b;
            --slate-600: #475569;
            --slate-700: #334155;
            --slate-800: #1e293b;
            --slate-900: #0f172a;
            --slate-950: #020617;
            --blue-500: #3b82f6;
            --green-500: #22c55e;
            --purple-500: #a855f7;
        }

        .stApp {
            background: linear-gradient(135deg, var(--slate-900) 0%, var(--slate-800) 100%);
        }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            color: var(--slate-50) !important;
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 1.5rem;
        }

        /* Step indicator */
        .step-indicator {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--slate-50);
            margin: 2rem 0 1rem 0;
        }

        .step-caption {
            font-size: 1.25rem;
            color: var(--slate-300);
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            font-weight: normal;
        }

        /* Content Boxes */
        .content-box {
            background: var(--slate-800);
            border: 1px solid var(--slate-700);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            width: 100%;
        }

        /* Add margin after progress bar */
        .stProgress {
            margin-bottom: 1rem;
        }

        .thought-box { border-top: 4px solid var(--blue-500); }
        .action-box { border-top: 4px solid var(--green-500); }
        .result-box { border-top: 4px solid var(--purple-500); }

        /* Headers */
        .box-header {
            font-size: 1.5rem !important;
            font-weight: 700;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--slate-700);
        }

        .thought-header { color: var(--blue-500) !important; }
        .action-header { color: var(--green-500) !important; }
        .result-header { color: var(--purple-500) !important; }

        /* Content styling */
        .box-content {
            padding: 0 1rem;
            font-size: 1.125rem;
            line-height: 1.75;
            color: var(--slate-200);
        }

        /* Sidebar styling */
        .sidebar-section {
            background: linear-gradient(to right, var(--slate-800) 0%, var(--slate-900) 100%);
            border-left: 6px solid var(--green-500);
            padding: 1.5rem;
            border-radius: 0 1rem 1rem 0;
            margin: 2rem 0;
        }

        /* Task card */
        .task-card {
            background: var(--slate-800);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--slate-700);
        }

        /* Content Summary Styling */
        .summary-box {
            background: var(--slate-800);
            border: 1px solid var(--slate-700);
            border-radius: 1rem;
            padding: 1rem;
            margin-bottom: 0.1em;
            width: 100%;
            font-size: 1.3rem;
        }
        
    </style>
    """,
        unsafe_allow_html=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the trajectory visualizer.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="MLGym Trajectory Visualizer")
    parser.add_argument(
        "--trajectory_dir",
        type=str,
        default=os.path.join(os.getcwd(), "trajectories"),
        help="Directory containing trajectory files",
    )
    
    return parser.parse_known_args()[0]


def append_exit(content: Dict[str, Any]) -> Dict[str, Any]:
    """Append exit status and submission information to the content history.

    Args:
        content: Dictionary containing trajectory content and metadata

    Returns:
        Dict[str, Any]: Updated content dictionary with exit information

    Raises:
        ValueError: If submission is referenced but not found in content
    """
    last_entry = content["history"][-1]
    if last_entry["role"] == "system":
        return content

    exit_status = content.get("info", {}).get("exit_status")
    if not exit_status:
        return content

    if exit_status.startswith("submitted"):
        if "submission" in content["info"]:
            submission = content["info"]["submission"]
            content["history"].append(
                {
                    "role": "model_patch",
                    "content": submission,
                }
            )
        else:
            raise ValueError("No submission in history or info")
    return content


def format_metric_value(value: Optional[Union[int, float]]) -> str:
    """Format metric values for display with appropriate formatting.

    Args:
        value: Numeric value to format

    Returns:
        str: Formatted string representation of the value
    """
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}"


def append_results(
    traj_path: Path,
    instance_id: str,
    content: Dict[str, Any],
    results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Append evaluation results and statistics to the content history.

    Args:
        traj_path: Path to the trajectory file
        instance_id: Identifier for the current instance
        content: Content dictionary to update
        results: Results dictionary containing scores and metrics

    Returns:
        Dict[str, Any]: Updated content dictionary with results
    """
    stats: List[str] = []
    model_stats = {}
    exit_status = None

    # Load trajectory data and extract statistics
    if traj_path.exists():
        data = json.loads(traj_path.read_text())
        info = data.get("info", {})
        exit_status = info.get("exit_status")
        model_stats = info.get("model_stats", {})

    # Format model statistics
    instance_cost = format_metric_value(model_stats.get("total_cost"))
    tokens_sent = format_metric_value(model_stats.get("tokens_sent"))
    tokens_received = format_metric_value(model_stats.get("tokens_received"))
    api_calls = format_metric_value(model_stats.get("api_calls"))

    # Build statistics report
    stats.extend([
        "*" * 39,
        "Run Stats",
        "*" * 39,
        f"Instance Cost: ${instance_cost}",
        f"Tokens Sent: {tokens_sent}",
        f"Tokens Received: {tokens_received}",
        f"API Calls: {api_calls}",
        f"Exit Status: {exit_status}",
    ])

    # Process and format results
    status = process_results(results)
    
    # Create and insert evaluation report
    eval_report = {
        "role": "Evaluation Report",
        "content": "\n".join([*stats, *status]),
    }
    content["history"].insert(0, eval_report)
    content["history"].append(eval_report)

    return content


def process_results(results: Optional[Dict[str, Any]]) -> List[str]:
    """Process and format evaluation results for display.

    Args:
        results: Dictionary containing evaluation results

    Returns:
        List[str]: Formatted status messages for display
    """
    if not results:
        return ["No scores found"]

    agent_results = results.get("agent")
    baseline_results = results.get("baseline")

    if not agent_results and not baseline_results:
        return ["Baseline and Agent scores not found"]

    status = []
    
    if baseline_results and agent_results:
        status.extend([
            "*" * 39,
            "Agent vs Baseline Scores",
            "*" * 39,
        ])
        
        formatted_scores = defaultdict(dict)
        for score_type, score in baseline_results.items():
            formatted_scores[score_type]["Baseline"] = score
            
        for i, agent_score in enumerate(agent_results):
            for score_type, score in agent_score.items():
                formatted_scores[score_type][f"Attempt {i+1}"] = score
                
        for score_type, scores in formatted_scores.items():
            status.append(f"Metric: {score_type}")
            status.extend(f"  {model}: {score:.3f}" for model, score in scores.items())
            
    elif baseline_results:
        status.append("**** Baseline Scores ****")
        status.extend(f"  {score_type}: {score}" for score_type, score in baseline_results.items())
        
    elif agent_results:
        status.append("**** Agent Scores ****")
        status.extend(f"  {score_type}: {score}" for score_type, score in agent_results.items())

    return status


def load_results(results_path: Path) -> Optional[Dict[str, Any]]:
    """Load results from a JSON file.

    Args:
        results_path: Path to the results file

    Returns:
        Optional[Dict[str, Any]]: Loaded results or None if file not found
    """
    if not results_path.exists():
        return None
        
    with open(results_path) as infile:
        return json.load(infile)


def load_content(file_name: str) -> Dict[str, Any]:
    """Load and process trajectory content from a file.

    Args:
        file_name: Path to the trajectory file

    Returns:
        Dict[str, Any]: Processed content with results and exit information
    """
    with open(file_name) as infile:
        content = json.load(infile)
        
    results_file = Path(file_name).parent / "results.json"
    results = load_results(results_file)

    content = append_exit(content)
    return append_results(
        Path(file_name),
        Path(file_name).stem,
        content,
        results,
    )


def find_trajectory_files(root_dir: str) -> List[Dict[str, str]]:
    """Recursively find all trajectory files in the given directory.

    Args:
        root_dir: Root directory to search for trajectory files

    Returns:
        List[Dict[str, str]]: List of dictionaries containing file information
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        st.error(f"Directory not found: {root_dir}")
        return []

    return [
        {
            "filename": file.name,
            "filepath": str(file.absolute()),
        }
        for file in root_path.rglob("*.traj")
    ]


def load_trajectory(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Load trajectory data from a JSON file.

    Args:
        file_path: Path to the trajectory file

    Returns:
        Optional[List[Dict[str, Any]]]: Loaded trajectory data or None if error occurs
    """
    try:
        with open(file_path, "r") as file:
            return json.load(file)["trajectory"]
    except FileNotFoundError:
        st.error(f"Trajectory file not found: {file_path}")
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in trajectory file: {file_path}")
    except KeyError:
        st.error(f"Missing 'trajectory' key in file: {file_path}")
    return None


def display_content_summary(content: str) -> None:
    """Display a collapsible summary of content.

    Args:
        content: Content to display in the summary
    """
    if not content:
        return

    st.markdown(
        f"""
        <div class="summary-box">
            <details class="content-summary">
                <summary class="summary-header">
                    📝 Evaluation Report
                </summary>
                <div class="summary-content">
                    <pre><code>{content}</code></pre>
                </div>
            </details>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_step(step_data: Dict[str, Any], step_num: int, total_steps: int) -> None:
    """Display a single step of the trajectory.

    Args:
        step_data: Data for the current step
        step_num: Current step number
        total_steps: Total number of steps
    """
    # Step indicator
    st.markdown(
        f"""
        <div class="step-indicator">Step {step_num + 1} / {total_steps}</div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.view_mode == "step":
        # Navigation controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "⬅️ Previous Step",
                key="prev_step",
                disabled=step_num == 0,
                use_container_width=True,
            ):
                st.session_state.current_step -= 1
                st.rerun()

        with col2:
            if st.button(
                "Next Step ➡️",
                key="next_step",
                disabled=step_num == total_steps - 1,
                use_container_width=True,
            ):
                st.session_state.current_step += 1
                st.rerun()

        # Progress tracking
        st.progress((step_num + 1) / total_steps)
        st.markdown(
            f"""
            <div class="step-caption">{step_data.get('caption', '')}</div>
            """,
            unsafe_allow_html=True,
        )

    # Display step components
    display_step_components(step_data)


def display_step_components(step_data: Dict[str, Any]) -> None:
    """Display the individual components of a step.

    Args:
        step_data: Data containing thought process, action, and observation
    """
    # Thought Process
    st.markdown(
        '<div class="content-box thought-box">'
        '<div class="box-header thought-header">💭 Thought Process</div>'
        f'<div class="box-content">{step_data["thought"].replace("DISCUSSION", "")}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Action
    st.markdown(
        '<div class="content-box action-box">'
        '<div class="box-header action-header">🤖 Action Taken</div>'
        '<div class="box-content">'
        f'<pre><code class="language-python">{step_data["action"]}</code></pre>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Result
    st.markdown(
        '<div class="content-box result-box">'
        '<div class="box-header result-header">💻 Execution Result</div>'
        '<div class="box-content">'
        f'<pre><code class="language-bash">{step_data["observation"]}</code></pre>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize the session state variables for the Streamlit application."""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'full'


def display_welcome_message() -> None:
    """Display the welcome message when no trajectory is selected."""
    st.markdown(
        """
        <div style='text-align: center; padding: 4rem 2rem;'>
            <h1>👋 Welcome to the MLGym Demo</h1>
            <p style='font-size: 1.2rem; color: #e0e0e0; margin: 2rem 0;'>
                Select a task from the sidebar to view the MLGym Agent's trajectory.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def setup_sidebar(args: argparse.Namespace) -> None:
    """Set up the sidebar with trajectory selection options.

    Args:
        args: Command line arguments containing trajectory directory
    """
    with st.sidebar:
        st.markdown("# 👩‍🔬 MLGym Agent")
        st.markdown("### Select Trajectory")
        st.markdown(f"**Current Directory:** {args.trajectory_dir}")
        
        all_trajectories = find_trajectory_files(args.trajectory_dir)
        
        if not all_trajectories:
            st.warning("No trajectory files found in the specified directory.")
            return
        
        for trajectory in all_trajectories:
            display_path = str(Path(trajectory['filepath']).relative_to(args.trajectory_dir))
            if st.button(display_path, key=trajectory["filepath"]):
                st.session_state.current_trajectory = trajectory["filepath"]
                st.session_state.view_mode = 'step'
                st.session_state.current_step = 0


def display_trajectory_content() -> None:
    """Display the selected trajectory content and visualization."""
    if "current_trajectory" not in st.session_state:
        display_welcome_message()
        return

    st.title("👩‍🔬 MLGym Agent")
    
    data = load_trajectory(st.session_state.current_trajectory)
    if not data:
        return

    content = load_content(st.session_state.current_trajectory)
    display_content_summary(content["history"][0]["content"])
    
    if st.session_state.view_mode == 'full':
        for i, step_data in enumerate(data):
            display_step(step_data, i, len(data))
    else:
        current_step = st.session_state.current_step
        if current_step < len(data):
            display_step(data[current_step], current_step, len(data))


def main() -> None:
    """Main entry point for the MLGym Trajectory Visualizer application."""
    args = parse_args()
    configure_page_style()
    initialize_session_state()
    setup_sidebar(args)
    display_trajectory_content()


if __name__ == "__main__":
    main()
