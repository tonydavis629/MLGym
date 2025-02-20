"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import json
import re
import time

import streamlit as st
from vars import SYSTEM_PROMPT, TASK_DESCRIPTIONS, TOOLS

# Configure the page with wide layout and custom theme
st.set_page_config(
    page_title="MLGym Demo",
    page_icon="👩‍🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with distinct colors for each section
st.markdown("""
<style>
    /* Meta Design System Colors */
    :root {
        /* Base colors */
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

        /* Primary blue */
        --blue-50: #eff6ff;
        --blue-100: #dbeafe;
        --blue-200: #bfdbfe;
        --blue-300: #93c5fd;
        --blue-400: #60a5fa;
        --blue-500: #3b82f6;
        --blue-600: #2563eb;
        --blue-700: #1d4ed8;
        --blue-800: #1e40af;
        --blue-900: #1e3a8a;
        --blue-950: #172554;

        /* Accent colors */
        --green-500: #22c55e;
        --green-600: #16a34a;
        --purple-500: #a855f7;
        --purple-600: #9333ea;
        --orange-500: #f97316;
        --orange-600: #ea580c;
    }

    /* Main app styling */
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

    /* Existing step indicator styles */
    .step-indicator {
        background: linear-gradient(135deg, #93c5fd 0%, #60a5fa 100%);
        padding: 2.5rem 3rem;
        border-radius: 1rem;
        margin-bottom: 2.5rem;
        color: var(--slate-900);
        font-weight: 700;
        font-size: 1.875rem;
        border: 1px solid rgba(147, 197, 253, 0.4);
        text-align: center;
        transform: translateY(0);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(8px);
    }

    /* Blue keyword highlighting */
    .step-indicator .keyword {
        color: #1E40AF; /* Deep blue */
        font-weight: 800;
        transition: color 0.2s ease;
    }

    .step-indicator .keyword:hover {
        color: #2563EB; /* Slightly brighter blue on hover */
    }


    /* Common Box Styles */
    .box-common {
        background: var(--slate-800);
        backdrop-filter: blur(12px);
        border: 1px solid var(--slate-700);
        border-radius: 1rem;
        padding: 2rem;
        height: 100%;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .box-common:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }

    /* Thought Box */
    .thought-box {
        composes: box-common;
        border-top: 4px solid var(--blue-500);
    }

    .thought-header {
        color: var(--blue-400) !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--blue-900);
    }

    /* Action Box */
    .action-box {
        composes: box-common;
        border-top: 4px solid var(--green-500);
    }

    .action-header {
        color: var(--green-500) !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--green-600);
    }

    /* Result Box */
    .result-box {
        composes: box-common;
        border-top: 4px solid var(--purple-500);
    }

    .result-header {
        color: var(--purple-500) !important;
        font-size: 1.5rem !important;
        font-weight: 700;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid var(--purple-600);
    }

    /* Scrollable Content */
    .scrollable-content {
        max-height: 600px;
        overflow-y: auto;
        padding: 1.5rem;
        background: var(--slate-900);
        border-radius: 0.75rem;
        font-size: 1.125rem;
        line-height: 1.75;
        color: var(--slate-200);
    }

    .scrollable-content::-webkit-scrollbar {
        width: 8px;
    }

    .scrollable-content::-webkit-scrollbar-track {
        background: var(--slate-800);
        border-radius: 4px;
    }

    .scrollable-content::-webkit-scrollbar-thumb {
        background: var(--slate-600);
        border-radius: 4px;
    }

    .scrollable-content::-webkit-scrollbar-thumb:hover {
        background: var(--slate-500);
    }

    /* Streaming Text Animation */
    .streaming-text {
        animation: fadeIn 0.3s ease-in;
        font-size: 1.125rem;
        line-height: 1.75;
        color: var(--slate-200);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Sidebar Styling */
    .sidebar-system-prompt {
        background: linear-gradient(to right, var(--slate-800) 0%, var(--slate-900) 100%);
        border-left: 6px solid var(--green-500);
        padding: 1.5rem;
        border-radius: 0 1rem 1rem 0;
        margin: 2rem 0;
        font-size: 1rem;
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.1);
    }

    .sidebar-system-prompt h3 {
        color: var(--green-500) !important;
        font-size: 1.375rem;
        margin-bottom: 1.25rem;
        border-bottom: 2px solid var(--green-600);
        padding-bottom: 0.75rem;
    }

    .sidebar-system-prompt p {
        color: var(--slate-400);
        line-height: 1.75;
        margin-bottom: 1rem;
    }

    /* Task Cards */
    .task-card {
        background: var(--slate-800);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--slate-700);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .task-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }

    /* Progress Bar */
    .step-progress {
        background: var(--slate-800);
        border-radius: 1rem;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 1px solid var(--slate-700);
    }

    .step-progress h4 {
        color: var(--slate-200);
        margin-bottom: 1rem;
    }

    /* Welcome Screen */
    .welcome-screen {
        text-align: center;
        padding: 5rem 2rem;
        background: var(--slate-800);
        border-radius: 1rem;
        margin: 2rem 0;
        border: 1px solid var(--slate-700);
    }

    .welcome-screen h1 {
        font-size: 3rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--blue-400) 0%, var(--blue-500) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .welcome-screen p {
        font-size: 1.25rem;
        color: var(--slate-400);
        margin-bottom: 1.5rem;
        line-height: 1.75;
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: var(--slate-800) !important;
        color: var(--slate-200) !important;
        border: 1px solid var(--slate-700) !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        transition: all 0.2s ease;
    }

    .streamlit-expanderHeader:hover {
        background-color: var(--slate-700) !important;
    }

    .streamlit-expanderContent {
        background-color: var(--slate-900) !important;
        border: 1px solid var(--slate-700) !important;
        border-radius: 0.5rem !important;
        padding: 1.5rem !important;
        margin-top: 0.5rem !important;
    }

    .welcome-box, .completion-box {
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.1) 0%, rgba(226, 232, 240, 0.15) 100%);
        border-radius: 1.5rem;
        padding: 3rem;
        margin: 2rem 0;
        border: 1px solid rgba(203, 213, 225, 0.2);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    .welcome-content, .completion-content {
        max-width: 900px;
        margin: 0 auto;
        text-align: center;
    }

    .welcome-icon, .completion-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
    }

    .welcome-box h1, .completion-box h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #93c5fd 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        line-height: 1.2;
        font-weight: 700;
    }

    .welcome-box p, .completion-box p {
        font-size: 1.25rem;
        color: #e2e8f0;
        line-height: 1.7;
        margin-bottom: 1.5rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin-top: 2.5rem;
    }

    .feature-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.75rem;
        transition: all 0.2s ease-in-out;
        backdrop-filter: blur(5px);
    }

    .feature-item:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 255, 255, 0.2);
        box-shadow: 
            0 8px 20px rgba(0, 0, 0, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }

    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .feature-item span:not(.feature-icon) {
        color: #f1f5f9;
        font-size: 1.1rem;
        font-weight: 500;
    }

    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }

    @keyframes pulseIcon {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }

    /* Action Keywords - blue highlight */
    .streaming-text span.action-keyword {
        background: linear-gradient(120deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.1) 100%);
        color: #60a5fa;
        padding: 0.1em 0.3em;
        border-radius: 0.3em;
        font-weight: 500;
        margin: 0 0.1em;
    }

    /* Analysis Keywords - purple highlight */
    .streaming-text span.analysis-keyword {
        background: linear-gradient(120deg, rgba(168, 85, 247, 0.2) 0%, rgba(168, 85, 247, 0.1) 100%);
        color: #c084fc;
        padding: 0.1em 0.3em;
        border-radius: 0.3em;
        font-weight: 500;
        margin: 0 0.1em;
    }

    /* Performance Keywords - green highlight */
    .streaming-text span.performance-keyword {
        background: linear-gradient(120deg, rgba(34, 197, 94, 0.2) 0%, rgba(34, 197, 94, 0.1) 100%);
        color: #4ade80;
        padding: 0.1em 0.3em;
        border-radius: 0.3em;
        font-weight: 500;
        margin: 0 0.1em;
    }

    /* Error Keywords - red highlight */
    .streaming-text span.error-keyword {
        background: linear-gradient(120deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        color: #f87171;
        padding: 0.1em 0.3em;
        border-radius: 0.3em;
        font-weight: 500;
        margin: 0 0.1em;
    }

    /* Optimization Keywords - orange highlight */
    .streaming-text span.optimization-keyword {
        background: linear-gradient(120deg, rgba(249, 115, 22, 0.2) 0%, rgba(249, 115, 22, 0.1) 100%);
        color: #fb923c;
        padding: 0.1em 0.3em;
        border-radius: 0.3em;
        font-weight: 500;
        margin: 0 0.1em;
    }

    /* Hover effect for all keywords */
    .streaming-text span[class*="-keyword"]:hover {
        filter: brightness(1.2);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }

</style>
""", unsafe_allow_html=True)

# Rest of the functions remain the same
def get_preview_lines(text, num_lines=5):
    lines = text.split('\n')
    preview = '\n'.join(lines[:num_lines])
    return preview, len(lines) > num_lines


def stream_text(text, placeholder, is_code=False, language=None, delay=0.01, stream=True):
    if not stream:
        if is_code:
            placeholder.code(text, language=language)
        else:
            placeholder.markdown(f'<div class="streaming-text">{text}</div>', unsafe_allow_html=True)
        return
    
    # For streaming text, use the same placeholder and update it
    displayed_text = ""
    for char in text:
        displayed_text += char
        if is_code:
            placeholder.code(displayed_text, language=language)
        else:
            placeholder.markdown(f'<div class="streaming-text">{displayed_text}</div>', unsafe_allow_html=True)
        time.sleep(delay)


def highlight_step_keywords(text):
    # List of keywords to highlight
    keywords = [
        # Action words
        r'\b(inspects|reads|understand|runs|reruns|implements|trains|evaluates|overachieving AI Scientist| comes up)\b',
        
        # Analysis words
        r'\b(analyzes|infers)\b',
        
        # Performance metrics
        r'(training accuracy improved significantly from \d+\.\d+% to \d+\.\d+%|test accuracy has improved from \d+\.\d+% to \d+\.\d+%|test accuracy is \d+\.\d+%|test accuracy of \d+\.\d+%)',
        
        # Error terms
        r'\b(execution fails|missing library|timeout)\b',
        
        # Optimization terms
        r'\b(increasing model complexity|adding data augmentation|using a learning rate scheduler|increasing the number of training epochs|make the code faster|reducing the number of epochs|using a smaller|random rotation and crop|convolutional layer|batch normalization|increases the number of training epochs|learning rate scheduler|weight decay|label smoothing|ResNet18 architecture|increased number of epochs|increased batch size|reduced learning rate|architecture|optimizer|learning rate warmup|cosine annealing with restarts|double checks| train a CNN| using a smaller architecture)\b'
    ]
    
    highlighted_text = text
    for pattern in keywords:
        highlighted_text = re.sub(
            pattern,
            lambda m: f'<span class="keyword">{m.group()}</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    return highlighted_text


def display_transition_page(step_container):
    """Display a transition page indicating steps are being skipped"""
    step_container.markdown("""
        <div class="step-indicator" style="background: linear-gradient(135deg, var(--slate-800) 0%, var(--slate-700) 100%);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">...</div>
            <div style="font-size: 1.5rem; color: var(--slate-400);">Skipping intermediate steps</div>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(2)


def display_steps(data, index):
    # Display welcome message
    welcome_box = st.empty()
    welcome_box.markdown("""
        <div class="welcome-box">
            <div class="welcome-content">
                <div class="welcome-icon">🧬</div>
                <h1>Welcome to the MLGym Demo</h1>
                <p>The MLGym Agent is tasked with maximising performance on a classical image classification task.</p>
                <p>Watch as it iteratively improves performance through:</p>
                <div class="feature-grid">
                    <div class="feature-item">
                        <span class="feature-icon">🔬</span>
                        <span>Idea Generation</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">⚡</span>
                        <span>Implementation</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📊</span>
                        <span>Experimentation</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">🔄</span>
                        <span>Iteration</span>
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(5)  # Show welcome message for 5 seconds
    welcome_box.empty()

    # Create persistent containers for each step
    step_container = st.empty()
    cols = st.columns(3)
    
    # Create containers for column contents
    thought_containers = {
        'main': cols[0].empty(),
        'content': cols[0].empty()
    }

    action_containers = {
        'main': cols[1].empty(),
        'content': cols[1].empty()
    }
    
    result_containers = {
        'main': cols[2].empty(),
        'content': cols[2].empty()
    }
    
    timestamp_container = st.empty()


    for i in range(index, len(data)):
        step_data = data[i]
        
        # Clear all containers at the start of each step
        for containers in [thought_containers, action_containers, result_containers]:
            containers['main'].empty()
            containers['content'].empty()

        # Update step indicator with highlighted keywords
        if "caption" in step_data:
            highlighted_caption = highlight_step_keywords(step_data["caption"])        
            step_container.markdown(f"""
                <div class="step-indicator">
                    🔍 Step {i + 1} / {len(data)}: {highlighted_caption}
                </div>
            """, unsafe_allow_html=True)
            time.sleep(2)
        else:
            step_container.markdown(f"""
                <div class="step-indicator">
                    🔍 Step {i + 1} / {len(data)}
                </div>
            """, unsafe_allow_html=True)
            time.sleep(2)

        # Display Thought Process
        thought_containers['main'].markdown('<div class="thought-header">💭 Thought Process</div>', unsafe_allow_html=True)
        thought_text = step_data["thought"].replace("DISCUSSION", "")
        preview, has_more = get_preview_lines(thought_text)
        
        if has_more:
            with thought_containers['content'].container():
                stream_text(preview, st.empty())
                with st.expander("💭 Full Thought Process", expanded=False):
                    st.markdown(thought_text)
        else:
            stream_text(thought_text, thought_containers['content'])
        time.sleep(1)
        
        # Display Action
        action_containers['main'].markdown('<div class="action-header">🤖 Action Taken</div>', unsafe_allow_html=True)
        action_text = step_data["action"]
        preview, has_more = get_preview_lines(action_text)
        
        if has_more:
            with action_containers['content'].container():
                stream_text(preview, st.empty(), is_code=True, language="python")
                with st.expander("🤖 Full Action Taken", expanded=False):
                    st.code(action_text, language="python")
        else:
            stream_text(action_text, action_containers['content'], is_code=True, language="python")
        time.sleep(min(step_data["execution_time"], 2))
        
        # Display Result
        result_containers['main'].markdown('<div class="result-header">💻 Execution Result</div>', unsafe_allow_html=True)
        result_text = step_data["observation"]
        preview, has_more = get_preview_lines(result_text)
        
        if has_more:
            with result_containers['content'].container():
                stream_text(preview, st.empty(), is_code=True, language="bash", stream=False)
                with st.expander("💻 Full Execution Result", expanded=False):
                    st.code(result_text, language="bash")
        else:
            stream_text(result_text, result_containers['content'], is_code=True, language="bash", stream=False)
                
        time.sleep(2)
        # Clear all containers at the end of the step
        for containers in [thought_containers, action_containers, result_containers]:
            containers['main'].empty()
            containers['content'].empty()


    # Display completion message after all steps
    step_container.empty()
    completion_box = st.empty()
    completion_box.markdown("""
        <div class="completion-box">
            <div class="completion-content">
                <div class="welcome-icon">🚀</div>
                <h1>Future of AI Research</h1>
                <p>This demo illustrates the potential of AI Research Assistants. We envision a future where AI Research Assistants:</p>
                <div class="feature-grid">
                    <div class="feature-item">
                        <span class="feature-icon">🤖</span>
                        <span>Enhance the process of generating novel ideas and algorithms </span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">⚙️</span>
                        <span>Assist with implementation, experimentation, and optimization</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📈</span>
                        <span>Self-improve over extended periods</span>
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">🔄</span>
                        <span>Accelerate research cycles and innovation</span>
                    </div>
                </div>
                <p style="margin-top: 2rem;">Together, human researchers and AI assistants can accelerate scientific progress and push the boundaries of what's possible.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


## Initialize session states
if "current_trajectory" not in st.session_state:
    st.session_state.current_trajectory = None
if "index" not in st.session_state:
    st.session_state.index = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

def load_trajectory(file_path):
    with open(file_path, "r") as file:
        return json.load(file)["trajectory"]

# Sidebar with trajectory selection and system prompt
with st.sidebar:
    st.markdown("# 👩‍🔬 MLGym Agent")
    
    # Add system prompt to sidebar
    st.markdown("""
    <div class="sidebar-system-prompt">
        <h3>💻 System Prompt</h3>
        <p>You are an autonomous machine learning researcher working directly in the command line with a special interface.</p>
        <p>Starting with baseline code, your goal is to achieve maximum accuracy on the test set within 50 steps.</p>
    </div>
    """, unsafe_allow_html=True)

    # Full system prompt in an expander
    with st.expander("📖 View Full System Prompt"):
        st.markdown(f"""
        <div class=system-prompt-full>
            {SYSTEM_PROMPT}
        </div>
        """, unsafe_allow_html=True)

    # Full system prompt in an expander
    with st.expander("🛠️ Tools"):
        st.markdown(f"""
        <div class=system-prompt-full>
            {TOOLS}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Select Task")
    
    trajectory_dir = "trajectories/mlgym_bench_v0"
    trajectories = [
        {
            "name": "Image Classification (CIFAR-10)",
            "icon": "🖼️",
            "description": "Train a model to classify images into 10 categories.",
            "path": f"{trajectory_dir}/metagen-claude-35-sonnet__imageClassificationCifar10__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents_device_5/imageClassificationCifar10.traj",
            "full_description": f"{TASK_DESCRIPTIONS['cifar10']}"
        },
        {
            "name": "House Price Prediction (Kaggle)",
            "icon": "🏠",
            "description": "Predict house prices using regression.",
            "path": f"{trajectory_dir}/metagen-claude-35-sonnet__regressionKaggleHousePrice__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents_device_2/regressionKaggleHousePrice.traj",
            "full_description": f"{TASK_DESCRIPTIONS['house_price']}"
        },
        {
            "name": "Language Modeling (FineWeb)",
            "icon": "📝",
            "description": "Decrease perplexity on FineWeb.",
            "path": f"{trajectory_dir}/metagen-claude-35-sonnet-new__languageModelingFineWeb__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__submission_forma_test_run_0/languageModelingFineWeb.traj",
            "full_description": f"{TASK_DESCRIPTIONS['language_modeling']}"
        },
        {
            "name": "Reinforcement Learning (MountainCar)",
            "icon": "🚗",
            "description": "Maximize reward in MountainCar by controlling the car to drive up a steep hill.",
            "path": f"{trajectory_dir}/metagen-gemini-15-pro__rlMountainCarContinuous__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents_run_2/rlMountainCarContinuous.traj",
            "full_description": f"{TASK_DESCRIPTIONS['mountain_car']}"
        },
        {
            "name": "Game Theory (Battle of Sexes)",
            "icon": "🎮",
            "description": "Find a winning strategy in the Battle of Sexes game. Batle of Sexes is a coordination game between two players with different preferences (e.g. a couple deciding how to spend their weekend).",
            "path": f"{trajectory_dir}/metagen-gpt-o1__battleOfSexes__better_thought_action_parser_with_insert__t-0.00__p-0.95__c-4.00__install-0__parallel_agents_run_0/battleOfSexes.traj",
            "full_description": f"{TASK_DESCRIPTIONS['battle_of_sexes']}"
        },
    ]

    for traj in trajectories:
        st.markdown(f"""
        <div style='padding: 1rem; background: rgba(45, 45, 45, 0.5); border-radius: 8px; margin-bottom: 1rem;'>
            <h3>{traj['icon']} {traj['name']}</h3>
            <p style='color: #999; margin-bottom: 1rem;'>{traj['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        # Add expandable task description
        with st.expander("📋 View Full Task Description"):
            st.markdown(traj['full_description'])

        if st.button("▶️ Replay Experiment", key=traj["path"]):
            st.session_state.current_trajectory = traj["path"]
            st.session_state.index = 0
            st.session_state.is_playing = True
            st.rerun()

# Main content area
if st.session_state.current_trajectory:
    st.title("👩‍🔬 MLGym Agent")

    # Load and display trajectory
    data = load_trajectory(st.session_state.current_trajectory)
    if st.session_state.is_playing:
        display_steps(data, st.session_state.index)
        st.session_state.index = len(data)
        st.session_state.is_playing = False
    
    # # Progress indicator
    # progress_percentage = (len(data) / 50) * 100
    # st.markdown(f"""
    # <div class="step-progress">
    #     <h4>Experiment Progress</h4>
    #     Step {len(data)} of 50 ({progress_percentage:.1f}% complete)
    # </div>
    # """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem;'>
        <h1>👋 Welcome to the MLGym Demo</h1>
        <p style='font-size: 1.2rem; color: #e0e0e0; margin: 2rem 0;'>
            Select a task from the sidebar to watch the MLGym Agent in action.
        </p>
        <p style='color: #999;'>
            Note: This is a replay of previously generated experiments, not real-time execution.
        </p>
    </div>
    """, unsafe_allow_html=True)
