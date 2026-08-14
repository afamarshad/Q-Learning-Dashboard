import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import random
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Q-Learning Maze",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* ======================================================
       MAIN TITLE
       ====================================================== */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 20px;
    }


    /* ======================================================
       MAZE
       ====================================================== */

    .maze-container {
        background: #172033;
        padding: 8px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        width: 100%;
        
    }

    .maze-cell {
        height: 125px;

        display: flex;
        flex-direction: column;

        justify-content: center;
        align-items: center;

        border: 2px solid #475569;

        position: relative;

        transition: all 0.2s ease;
    }

    .maze-empty {
        background: #f8fafc;
    }

    .maze-path {
        background: #dbeafe;
        border: 3px solid #60a5fa;
    }

    .maze-start {
        background: #dcfce7;
        border: 3px solid #22c55e;
    }

    .maze-goal {
        background: #fef3c7;
        border: 3px solid #f59e0b;
    }

    .agent {
        font-size: 48px;
        line-height: 1;
    }

    .goal {
        font-size: 48px;
        line-height: 1;
    }

    .arrow {
        font-size: 42px;
        font-weight: bold;
        color: #2563eb;
        line-height: 1;
    }

    .cell-number {
        font-size: 12px;
        color: #64748b;
        margin-top: 8px;
    }

    .cell-label {
        font-size: 11px;
        font-weight: 700;
        color: #475569;
        margin-top: 3px;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .metric-title {
        font-size: 13px;
        color: #64748b;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 750;
        color: #1e293b;
        margin-top: 4px;
    }


    /* ======================================================
       PATH
       ====================================================== */

    .path-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 18px;
        font-size: 17px;
        text-align: center;
        margin-top: 15px;
    }


    /* ======================================================
       EQUATION
       ====================================================== */

    .equation-box {
        background: #111827;
        color: white;
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        font-size: 25px;
        font-weight: 600;
        margin: 20px 0;
    }


    /* ======================================================
       INFO CARDS
       ====================================================== */

    .info-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 20px;
        min-height: 150px;
    }


    /* ======================================================
       LEGEND
       ====================================================== */

    .legend {
        display: flex;
        justify-content: center;
        gap: 35px;
        margin: 20px 0;
        font-size: 15px;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        border-right: 1px solid #e5e7eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS
# ============================================================

GRID_SIZE = 4

START = (0, 0)

GOAL = (3, 3)

ACTIONS = ["↑", "↓", "←", "→"]

ACTION_NAMES = [
    "Up",
    "Down",
    "Left",
    "Right"
]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 Q-Learning Maze</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An interactive 4×4 reinforcement learning environment'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Training Controls")

st.sidebar.markdown(
    "Configure the Q-learning agent."
)

st.sidebar.divider()


episodes = st.sidebar.slider(
    "Training Episodes",
    min_value=100,
    max_value=5000,
    value=500,
    step=100
)


alpha = st.sidebar.slider(
    "Learning Rate (α)",
    min_value=0.01,
    max_value=1.0,
    value=0.10,
    step=0.01
)


gamma = st.sidebar.slider(
    "Discount Factor (γ)",
    min_value=0.0,
    max_value=1.0,
    value=0.90,
    step=0.05
)


epsilon = st.sidebar.slider(
    "Exploration Probability (ε)",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.05
)


st.sidebar.divider()


train_button = st.sidebar.button(
    "🚀 Train Q-Learning Agent",
    use_container_width=True,
    type="primary"
)


# ============================================================
# STATE NUMBER
# ============================================================

def state_number(position):

    row, col = position

    return row * GRID_SIZE + col


# ============================================================
# TAKE ACTION
# ============================================================

def take_action(position, action):

    row, col = position

    # Up
    if action == 0:
        row -= 1

    # Down
    elif action == 1:
        row += 1

    # Left
    elif action == 2:
        col -= 1

    # Right
    elif action == 3:
        col += 1

    # Keep agent inside grid

    row = max(
        0,
        min(row, GRID_SIZE - 1)
    )

    col = max(
        0,
        min(col, GRID_SIZE - 1)
    )

    return (row, col)


# ============================================================
# REWARD
# ============================================================

def get_reward(position):

    if position == GOAL:
        return 10

    return -1


# ============================================================
# Q-LEARNING TRAINING
# ============================================================

def train_q_learning(
    episodes,
    alpha,
    gamma,
    epsilon
):

    # 16 states × 4 actions

    q_table = np.zeros(
        (
            GRID_SIZE * GRID_SIZE,
            4
        )
    )

    episode_rewards = []

    episode_steps = []

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(episodes):

        position = START

        total_reward = 0

        steps_taken = 0

        # ====================================================
        # STEPS
        # ====================================================

        for step in range(100):

            current_state = state_number(
                position
            )

            # =================================================
            # EPSILON-GREEDY
            # =================================================

            if random.uniform(0, 1) < epsilon:

                # Explore

                action = random.randint(
                    0,
                    3
                )

            else:

                # Exploit

                action = np.argmax(
                    q_table[current_state]
                )

            # =================================================
            # TAKE ACTION
            # =================================================

            new_position = take_action(
                position,
                action
            )

            # =================================================
            # GET REWARD
            # =================================================

            reward = get_reward(
                new_position
            )

            total_reward += reward

            steps_taken += 1

            # =================================================
            # NEXT STATE
            # =================================================

            next_state = state_number(
                new_position
            )

            # =================================================
            # Q-LEARNING EQUATION
            # =================================================

            q_table[
                current_state,
                action
            ] = (

                q_table[
                    current_state,
                    action
                ]

                +

                alpha
                *
                (
                    reward

                    +

                    gamma
                    *
                    np.max(
                        q_table[next_state]
                    )

                    -

                    q_table[
                        current_state,
                        action
                    ]
                )
            )

            # Move

            position = new_position

            # =================================================
            # GOAL
            # =================================================

            if position == GOAL:

                break

        episode_rewards.append(
            total_reward
        )

        episode_steps.append(
            steps_taken
        )

    return (
        q_table,
        episode_rewards,
        episode_steps
    )


# ============================================================
# GET LEARNED PATH
# ============================================================

def get_learned_path(q_table):

    position = START

    path = [position]

    visited = set()

    for step in range(20):

        # Goal reached

        if position == GOAL:
            break

        # Prevent loops

        if position in visited:
            break

        visited.add(position)

        state = state_number(
            position
        )

        # Best learned action

        action = np.argmax(
            q_table[state]
        )

        new_position = take_action(
            position,
            action
        )

        # Cannot move

        if new_position == position:
            break

        position = new_position

        path.append(position)

    return path


# ============================================================
# CREATE PATH ARROWS
# ============================================================

def create_path_arrows(path):

    arrows = {}

    for i in range(
        len(path) - 1
    ):

        current = path[i]

        next_position = path[i + 1]

        current_row, current_col = current

        next_row, next_col = next_position

        # Down

        if next_row > current_row:

            arrows[current] = "↓"

        # Up

        elif next_row < current_row:

            arrows[current] = "↑"

        # Right

        elif next_col > current_col:

            arrows[current] = "→"

        # Left

        elif next_col < current_col:

            arrows[current] = "←"

    return arrows


# ============================================================
# DRAW MAZE
# ============================================================

def display_maze(path=None):

    # --------------------------------------------------------
    # If training has not happened yet
    # --------------------------------------------------------

    if path is None:
        path = []

    # --------------------------------------------------------
    # Create arrows for the learned path
    # --------------------------------------------------------

    arrows = create_path_arrows(path)

    # --------------------------------------------------------
    # Start HTML
    # --------------------------------------------------------

    maze_html = """
    <!DOCTYPE html>

    <html>

    <head>

        <style>

            * {
                box-sizing: border-box;
            }

            body {
                margin: 0;
                padding: 10px;
                font-family: Arial, sans-serif;
                background: white;
            }

            /* ==============================
               LEGEND
               ============================== */

            .legend {
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 25px;
                flex-wrap: wrap;

                margin-bottom: 18px;

                font-size: 14px;
                color: #374151;
            }

            /* ==============================
               MAZE
               ============================== */

            .maze-container {
                background: #172033;

                padding: 12px;

                border-radius: 18px;

                width: 100%;

                box-shadow:
                    0 8px 25px rgba(0,0,0,0.15);
            }

            .maze-grid {

                display: grid;

                grid-template-columns:
                    repeat(4, 1fr);

                gap: 3px;

                width: 100%;
            }

            /* ==============================
               CELL
               ============================== */

            .maze-cell {

                height: 105px;

                display: flex;

                flex-direction: column;

                justify-content: center;

                align-items: center;

                border-radius: 8px;

                position: relative;

                transition: 0.2s;
            }

            /* ==============================
               NORMAL CELL
               ============================== */

            .maze-empty {

                background: #f8fafc;

                border: 2px solid #64748b;
            }

            /* ==============================
               START
               ============================== */

            .maze-start {

                background: #dcfce7;

                border: 3px solid #22c55e;
            }

            /* ==============================
               GOAL
               ============================== */

            .maze-goal {

                background: #fef3c7;

                border: 3px solid #f59e0b;
            }

            /* ==============================
               LEARNED PATH
               ============================== */

            .maze-path {

                background: #dbeafe;

                border: 3px solid #3b82f6;
            }

            /* ==============================
               AGENT
               ============================== */

            .agent {

                font-size: 38px;

                line-height: 1;

                margin-bottom: 5px;
            }

            /* ==============================
               GOAL ICON
               ============================== */

            .goal {

                font-size: 38px;

                line-height: 1;

                margin-bottom: 5px;
            }

            /* ==============================
               ARROW
               ============================== */

            .arrow {

                font-size: 38px;

                font-weight: bold;

                color: #2563eb;

                line-height: 1;

                margin-bottom: 5px;
            }

            /* ==============================
               STATE NUMBER
               ============================== */

            .cell-number {

                font-size: 13px;

                color: #475569;

                margin-top: 5px;
            }

            /* ==============================
               LABEL
               ============================== */

            .cell-label {

                font-size: 11px;

                font-weight: bold;

                color: #475569;

                margin-top: 3px;

                letter-spacing: 0.5px;
            }

        </style>

    </head>


    <body>

        <!-- =================================
             LEGEND
             ================================= -->

        <div class="legend">

            <span>🤖 Agent</span>

            <span>🏆 Goal</span>

            <span>🔵 Learned Path</span>

            <span>⬜ Unvisited Cell</span>

        </div>


        <!-- =================================
             MAZE
             ================================= -->

        <div class="maze-container">

            <div class="maze-grid">
    """

    # ========================================================
    # CREATE 4 × 4 GRID
    # ========================================================

    for row in range(GRID_SIZE):

        for col in range(GRID_SIZE):

            position = (row, col)

            state = state_number(position)

            # ------------------------------------------------
            # START
            # ------------------------------------------------

            if position == START:

                css_class = "maze-start"

                symbol = "🤖"

                label = "START"

                symbol_class = "agent"

            # ------------------------------------------------
            # GOAL
            # ------------------------------------------------

            elif position == GOAL:

                css_class = "maze-goal"

                symbol = "🏆"

                label = "GOAL"

                symbol_class = "goal"

            # ------------------------------------------------
            # LEARNED PATH
            # ------------------------------------------------

            elif position in arrows:

                css_class = "maze-path"

                symbol = arrows[position]

                label = "PATH"

                symbol_class = "arrow"

            # ------------------------------------------------
            # NORMAL CELL
            # ------------------------------------------------

            else:

                css_class = "maze-empty"

                symbol = ""

                label = ""

                symbol_class = ""

            # ------------------------------------------------
            # ADD CELL
            # ------------------------------------------------

            maze_html += f"""
                <div class="maze-cell {css_class}">

                    <div class="{symbol_class}">
                        {symbol}
                    </div>

                    <div class="cell-number">
                        State {state}
                    </div>

                    <div class="cell-label">
                        {label}
                    </div>

                </div>
            """

    # ========================================================
    # CLOSE HTML
    # ========================================================

    maze_html += """
            </div>

        </div>

    </body>

    </html>
    """

    # ========================================================
    # RENDER COMPLETE HTML COMPONENT
    # ========================================================

    components.html(
        maze_html,
        height=650,
        scrolling=False
    )


# ============================================================
# INITIAL SESSION STATE
# ============================================================

if "trained" not in st.session_state:

    st.session_state.trained = False


# ============================================================
# TRAINING
# ============================================================

if train_button:

    with st.spinner(
        "🤖 Agent is learning..."
    ):

        (
            q_table,
            rewards,
            steps_history
        ) = train_q_learning(
            episodes,
            alpha,
            gamma,
            epsilon
        )

        path = get_learned_path(
            q_table
        )

        st.session_state.q_table = q_table

        st.session_state.rewards = rewards

        st.session_state.steps_history = (
            steps_history
        )

        st.session_state.path = path

        st.session_state.trained = True


# ============================================================
# TABS
# ============================================================

tab_summary, tab_maze, tab_qtable, tab_training, tab_about = st.tabs(
    [
        "🏠 Summary",
        "🗺️ Learned Maze",
        "📋 Q-Table",
        "📈 Training",
        "🧠 How It Works"
    ]
)


# ============================================================
# SUMMARY TAB
# ============================================================

with tab_summary:

    st.header("🏠 Q-Learning Summary")

    if not st.session_state.trained:

        st.info(
            "👈 Configure the parameters in the sidebar "
            "and click **Train Q-Learning Agent**."
        )

        st.markdown(
            "### 🎯 Objective"
        )

        st.write(
            "The agent must learn the best route from "
            "**Start (S)** to the **Goal (G)** in a 4×4 grid."
        )

        st.markdown(
            "### 🌎 Environment"
        )

        summary_cols = st.columns(4)

        with summary_cols[0]:

            st.metric(
                "States",
                "16"
            )

        with summary_cols[1]:

            st.metric(
                "Actions",
                "4"
            )

        with summary_cols[2]:

            st.metric(
                "Goal Reward",
                "+10"
            )

        with summary_cols[3]:

            st.metric(
                "Other Move",
                "-1"
            )

    else:

        q_table = st.session_state.q_table

        rewards = st.session_state.rewards

        steps_history = (
            st.session_state.steps_history
        )

        path = st.session_state.path

        # ----------------------------------------------------
        # SUMMARY METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Training Episodes",
                episodes
            )

        with col2:

            st.metric(
                "Path Length",
                len(path) - 1
            )

        with col3:

            st.metric(
                "Final Reward",
                rewards[-1]
            )

        with col4:

            if path[-1] == GOAL:

                st.metric(
                    "Goal Status",
                    "Reached 🎯"
                )

            else:

                st.metric(
                    "Goal Status",
                    "Not Reached"
                )

        st.divider()

        # ----------------------------------------------------
        # SHORT DESCRIPTION
        # ----------------------------------------------------

        st.markdown(
            "### 🤖 What did the agent learn?"
        )

        if path[-1] == GOAL:

            st.success(
                f"The agent successfully learned a route "
                f"from the starting point to the goal in "
                f"**{len(path) - 1} steps**."
            )

        else:

            st.warning(
                "The agent has not yet learned a complete "
                "route to the goal."
            )

        st.markdown(
            "### 📍 Learned Route"
        )

        path_text = " → ".join(
            [
                f"({r},{c})"
                for r, c in path
            ]
        )

        st.code(
            path_text
        )

        st.markdown(
            "### ⚙️ Current Parameters"
        )

        parameter_df = pd.DataFrame(
            {
                "Parameter": [
                    "Episodes",
                    "Learning Rate (α)",
                    "Discount Factor (γ)",
                    "Exploration (ε)"
                ],
                "Value": [
                    episodes,
                    alpha,
                    gamma,
                    epsilon
                ]
            }
        )

        st.dataframe(
            parameter_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# LEARNED MAZE TAB
# ============================================================

with tab_maze:

    st.header("🗺️ Final Learned Maze")

    if not st.session_state.trained:

        st.info(
            "Train the agent first to see its learned path."
        )

        # Preview maze

        display_maze([])

    else:

        path = st.session_state.path

        # ----------------------------------------------------
        # RESULT MESSAGE
        # ----------------------------------------------------

        if path[-1] == GOAL:

            st.success(
                f"🏆 The agent reached the goal in "
                f"**{len(path) - 1} steps!**"
            )

        else:

            st.warning(
                "The agent did not reach the goal."
            )

        # ----------------------------------------------------
        # MAZE
        # ----------------------------------------------------

        display_maze(
            path
        )

        # ----------------------------------------------------
        # PATH
        # ----------------------------------------------------

        st.subheader(
            "🤖 Agent Journey"
        )

        path_text = "  →  ".join(
            [
                f"({r},{c})"
                for r, c in path
            ]
        )

        st.markdown(
            f"""
            <div class="path-box">

            🤖 {path_text} 🏆

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# Q-TABLE TAB
# ============================================================

with tab_qtable:

    st.header("📋 Learned Q-Table")

    if not st.session_state.trained:

        st.info(
            "Train the agent to generate the Q-table."
        )

    else:

        q_table = st.session_state.q_table

        st.write(
            "Each row represents a state and each column "
            "represents the Q-value for an action."
        )

        q_df = pd.DataFrame(
            np.round(
                q_table,
                2
            ),
            columns=[
                "↑ Up",
                "↓ Down",
                "← Left",
                "→ Right"
            ]
        )

        q_df.insert(
            0,
            "State",
            range(16)
        )

        st.dataframe(
            q_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # ----------------------------------------------------
        # BEST ACTION
        # ----------------------------------------------------

        st.subheader(
            "🧭 Best Action for Each State"
        )

        best_actions = []

        for row in range(GRID_SIZE):

            row_data = []

            for col in range(GRID_SIZE):

                position = (
                    row,
                    col
                )

                if position == GOAL:

                    row_data.append(
                        "🏆 Goal"
                    )

                else:

                    state = state_number(
                        position
                    )

                    best_action = np.argmax(
                        q_table[state]
                    )

                    row_data.append(
                        ACTIONS[best_action]
                        +
                        " "
                        +
                        ACTION_NAMES[
                            best_action
                        ]
                    )

            best_actions.append(
                row_data
            )

        best_df = pd.DataFrame(
            best_actions,
            columns=[
                "Column 0",
                "Column 1",
                "Column 2",
                "Column 3"
            ]
        )

        st.dataframe(
            best_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TRAINING TAB
# ============================================================

with tab_training:

    st.header("📈 Training Performance")

    if not st.session_state.trained:

        st.info(
            "Train the agent to see the learning graphs."
        )

    else:

        rewards = st.session_state.rewards

        steps_history = (
            st.session_state.steps_history
        )

        # ----------------------------------------------------
        # REWARD GRAPH
        # ----------------------------------------------------

        st.subheader(
            "💰 Reward per Episode"
        )

        reward_df = pd.DataFrame(
            {
                "Reward": rewards
            }
        )

        st.line_chart(
            reward_df,
            height=350
        )

        # ----------------------------------------------------
        # STEPS GRAPH
        # ----------------------------------------------------

        st.subheader(
            "👣 Steps per Episode"
        )

        steps_df = pd.DataFrame(
            {
                "Steps": steps_history
            }
        )

        st.line_chart(
            steps_df,
            height=350
        )


# ============================================================
# HOW IT WORKS TAB
# ============================================================

with tab_about:

    st.header("🧠 How Q-Learning Works")

    st.write(
        "Q-Learning is a reinforcement learning algorithm "
        "that learns which action is best to take in each state."
    )

    # --------------------------------------------------------
    # EQUATION
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="equation-box">

        Q(s,a) = Q(s,a) + α [R + γ max Q(s',a') − Q(s,a)]

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### Q(s,a)

            Current Q-value for taking action `a`
            from state `s`.

            ### α — Learning Rate

            Controls how strongly new information
            changes the existing Q-value.

            ### R — Reward

            Immediate feedback received after
            taking an action.
            """
        )

    with col2:

        st.markdown(
            """
            ### γ — Discount Factor

            Determines how much future rewards matter.

            ### s'

            The next state after taking an action.

            ### ε — Exploration

            Determines how often the agent tries
            random actions instead of using its
            current best action.
            """
        )

    st.divider()

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    st.subheader(
        "🔄 Learning Process"
    )

    process_cols = st.columns(5)

    process = [
        ("1️⃣", "Observe", "Current state"),
        ("2️⃣", "Choose", "Select action"),
        ("3️⃣", "Move", "Enter next state"),
        ("4️⃣", "Reward", "Receive feedback"),
        ("5️⃣", "Learn", "Update Q-value")
    ]

    for col, item in zip(
        process_cols,
        process
    ):

        icon, title, description = item

        with col:

            st.markdown(
                f"""
                <div class="info-card">

                <h2>{icon}</h2>

                <h4>{title}</h4>

                <p>{description}</p>

                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.subheader(
        "🎮 Environment Rules"
    )

    rules_df = pd.DataFrame(
        {
            "Element": [
                "Grid",
                "States",
                "Actions",
                "Start",
                "Goal",
                "Normal Move",
                "Goal Reached"
            ],
            "Value": [
                "4 × 4",
                "16",
                "Up, Down, Left, Right",
                "(0, 0)",
                "(3, 3)",
                "-1 reward",
                "+10 reward"
            ]
        }
    )

    st.dataframe(
        rules_df,
        use_container_width=True,
        hide_index=True
    )