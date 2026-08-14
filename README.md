# 🤖 Q-Learning Maze Dashboard

An interactive **Streamlit application** that demonstrates how a Q-Learning reinforcement learning agent learns to navigate a 4×4 maze from a starting point to a goal.

## ✨ Features

* Interactive 4×4 maze
* Q-Learning reinforcement learning algorithm
* Adjustable training episodes
* Adjustable learning rate (α)
* Adjustable discount factor (γ)
* Adjustable exploration probability (ε)
* Visual learned path
* Q-Table visualization
* Training reward graph
* Steps-per-episode graph
* Explanation of how Q-Learning works

## 🛠️ Technologies

* Python
* Streamlit
* NumPy
* Pandas

## 🚀 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
cd <your-project-folder>
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Run the App

```bash
streamlit run app.py
```

The application will open in your web browser.

## 🧠 How It Works

The agent starts at **(0, 0)** and learns to reach the goal at **(3, 3)**.

The agent can move:

* ↑ Up
* ↓ Down
* ← Left
* → Right

The reward system is:

* Normal move: **-1**
* Reaching the goal: **+10**

After training, the application displays the agent's learned route, Q-Table, and training performance.

## 📁 Project Files

```text
├── app.py
├── requirements.txt
└── README.md
```

## 👩‍💻 Author

**Afsah Arshad**
