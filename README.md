# 🕸️ LangGraph Fundamentals — State, Conditional Routing & a Real Tool-Using ReAct Agent (Agentic AI #6)

The third distinct framework in this Agentic AI series, after AutoGen's conversation-centric model and CrewAI's task/crew model: **LangGraph**, which represents agentic workflows as an explicit **graph of nodes and edges**, with a typed **state object** flowing through every step. Built up from a single-node graph, through conditional branching, to a genuine tool-using **ReAct agent** that searches the live web and does real arithmetic — deciding for itself when a tool is needed.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-1C3C3C)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)
![Agentic AI](https://img.shields.io/badge/Series-Agentic%20AI%20%2306-8A2BE2)

---

## 📌 Overview

AutoGen models agents as **conversation participants**; CrewAI models them as **role-based task executors**. LangGraph takes a third, distinct approach: an agentic system is a **directed graph**, where each node is a function that receives and returns state, and edges — fixed or conditional — decide where execution goes next. This is the framework closest to how a state machine or workflow engine actually works under the hood, and this notebook builds that mental model from the ground up before arriving at a genuine ReAct-pattern tool-using agent.

---

## 🏗️ The Progression, in Three Stages

```
STAGE 1 — Single node
  TypedDict state → StateGraph → add_node → add_edge(node, END) → compile → invoke

STAGE 2 — Multi-node with conditional routing
  first_node → decide_location() [random routing function] → second_node OR third_node → END

STAGE 3 — A real tool-using ReAct agent
  MessagesState → chatbot node (LLM + bound tools) → tools_condition
        ↑                                                    │
        └──────────────── ToolNode(tools) ←──────────────────┘
  (loops between reasoning and tool execution until no tool call remains)
```

---

## 🔬 Stage 1 — The Core Mental Model: State → Node → Updated State

```python
class SomeState(TypedDict):
    att1: str
    att2: str

def some_function(state: SomeState):
    state['att1'] = "Value Changed by node some_function()"
    return state

graph = StateGraph(SomeState)
graph.add_node('node1', some_function)
graph.add_edge("node1", END)
graph.set_entry_point('node1')
compiled_graph = graph.compile()

result = compiled_graph.invoke({"att1": "Original Value", "att2": "Hello"})
```

Every LangGraph concept in this notebook builds on one repeated cycle: a **`TypedDict`** defines the state's shape, a **node** is just a plain function that takes state in and returns an update, and **edges** decide what runs next. `add_node()` defines *what* work happens; `add_edge()` defines *what happens after*; `set_entry_point()` defines *where things start*; `compile()` turns the definition into something runnable; `invoke()` actually runs it.

---

## 🔬 Stage 2 — Multiple Nodes and Conditional Routing

```python
def decide_location(state) -> Literal["second_node", "third_node"]:
    if random.random() < 0.5:
        return "second_node"
    else:
        return "third_node"

builder.add_edge(START, 'first_node')
builder.add_conditional_edges("first_node", decide_location)
builder.add_edge("second_node", END)
builder.add_edge("third_node", END)
```

Two genuinely different function roles emerge here: **normal nodes** process and update state, while **routing functions** like `decide_location()` inspect the state and decide *where the graph goes next*, without changing the state themselves. `add_conditional_edges()` is what wires a routing function into the graph instead of a fixed, single next-step edge — the mechanism that turns a straight-line pipeline into a genuine branching workflow. Multiple test invocations (`"Hi, I am Batman"`, `"Hi, my name is Andrew Ng"`, etc.) each take a randomly different path through the graph, visibly proving the conditional routing works.

---

## 🔬 Stage 3 — A Real ReAct Agent: Reasoning, Tool Calls, and a Loop

### Purpose-Built Message State
```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
```
Rather than hand-rolling a state shape, `add_messages` is LangGraph's built-in **reducer** for conversation history — new messages are *merged* into the existing list, not overwritten, exactly the semantics a multi-turn chat needs.

### Real Tools: Web Search and Arithmetic
```python
def search_duckduckgo(query: str):
    """Searches DuckDuckGo for the given query."""
    return DuckDuckGoSearchRun().invoke(query)

def add(a: int, b: int) -> int:
    """Adds a and b"""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply a and b"""
    return a * b

tools = [search_duckduckgo, add, multiply]
llm_with_tools = llm.bind_tools(tools)
```
`bind_tools()` doesn't execute anything — it gives the LLM *knowledge* of what tools exist, so it can request one when reasoning determines it's needed. Note the docstrings on every tool function: LangGraph/LangChain use them as the tool's description for the LLM, so an undocumented function is effectively an invisible tool.

### The ReAct Loop Itself
```python
graph_builder.add_node("assistant", chatbot)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_edge(START, "assistant")
graph_builder.add_conditional_edges("assistant", tools_condition)
graph_builder.add_edge("tools", "assistant")   # ← the loop-back edge

react_graph = graph_builder.compile()
```
This is the notebook's payoff structure: `tools_condition` (a **prebuilt** LangGraph routing function) checks whether the LLM's last response requested a tool call. If yes, control passes to `ToolNode`, which actually executes the tool — and then, critically, `add_edge("tools", "assistant")` sends the result **back** to the assistant node, not to `END`. This loop is what lets the agent reason, call a tool, see the result, and reason again — repeating until no further tool call is needed.

### Tested on a Genuinely Multi-Step Query
```python
response = react_graph.invoke({"messages": [HumanMessage(
    content="What is the weather in Hydrabad. Multiply it by 2 and add 5."
)]})
```
This single question requires **three separate tool calls in sequence** (a search, then two arithmetic operations) — the agent must reason about *which* tool to call, *when*, and use one tool's result as input to the next, entirely on its own via the loop structure above.

---

## 🗂️ Repository Structure

```
langgraph-fundamentals-react-agent/
├── AgenticAI_06_LangGraph_01.ipynb   # Main notebook
├── requirements.txt                    # Dependencies
├── .gitignore                          # Keeps secrets out of git
├── .env.example                        # Template for required environment variables
└── README.md                           # This documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- No key needed for DuckDuckGo search — `ddgs`/`duckduckgo-search` work without authentication

### Installation

```bash
git clone https://github.com/Kailaswadje/langgraph-fundamentals-react-agent.git
cd langgraph-fundamentals-react-agent

pip install -r requirements.txt

jupyter notebook AgenticAI_06_LangGraph_01.ipynb
```

> ⚠️ The notebook already uses `getpass()` for the API key — keep it. `draw_mermaid_png()` calls require internet access to render graph diagrams; if offline, comment those cells out and the graphs will still execute correctly. Clear notebook outputs before pushing.

---

## 🧠 Key Takeaways

- **LangGraph's core unit is State → Node → Updated State** — every pattern in this notebook, from a single node to a full ReAct loop, is a variation on this one cycle
- **Routing functions decide, nodes do** — `decide_location()`/`tools_condition` never change state, they only choose the next edge; keeping "what happens" and "what happens next" as separate concerns is a clean design principle worth carrying into any graph-based system
- **`add_messages` is a reducer, not a replacement** — understanding that new state merges rather than overwrites is essential the moment conversation history matters
- **`bind_tools()` informs; `ToolNode` executes** — the LLM only ever *requests* a tool call, and a separate graph node is what actually runs it, keeping reasoning and execution cleanly separated
- **The loop-back edge (`"tools" → "assistant"`) is what makes this a genuine ReAct agent**, not a one-shot tool call — the agent can chain multiple tool calls across multiple reasoning steps for a single user question
- Of the three frameworks covered in this series, LangGraph's explicit, inspectable graph structure is the closest fit to the auditable, debuggable orchestration my dissertation's agentic intelligence platform needs

---

## 📚 Agentic AI Series Context

| Part | Project | Framework | Focus |
|---|---|---|---|
| 01 | [AutoGen Agent Fundamentals](https://github.com/Kailaswadje/agentic-ai-autogen-introduction) | AutoGen | `ConversableAgent`, peer-to-peer negotiation |
| 02 | [UserProxyAgent & Sequential Chat](https://github.com/Kailaswadje/agentic-ai-userproxyagent-sequential-chat) | AutoGen | Human-facing coordination, pipeline handoffs |
| 03 | [Group Chat, State Flow & Nested Chat](https://github.com/Kailaswadje/agentic-ai-group-chat-state-flow-nested-chat) | AutoGen | Multi-agent teams, deterministic orchestration |
| 04 | [CrewAI Fundamentals](https://github.com/Kailaswadje/crewai-fundamentals-recipe-crew) | CrewAI | Structured agents, task/agent separation, planning |
| 05 | [CrewAI with Real Web Search](https://github.com/Kailaswadje/crewai-web-search-market-research-crew) | CrewAI | Tool-equipped agents, automatic sequential context |
| **06 (this repo)** | LangGraph Fundamentals | **LangGraph** | Explicit state graphs, conditional routing, ReAct tool loop |

---

## 🔮 Possible Extensions

- [ ] Add a fourth tool (e.g. a calculator for division/subtraction) and test more complex chained queries
- [ ] Replace the random `decide_location()` router with a genuine LLM-based routing decision
- [ ] Add persistent memory/checkpointing across multiple `invoke()` calls for true multi-turn conversation
- [ ] Rebuild one of the AutoGen or CrewAI scenarios from earlier in this series as a LangGraph, for a direct three-way framework comparison

---

## 👤 Author

**Kailas Wadje**
MSc Data Science & AI, University of Liverpool

- GitHub: [@Kailaswadje](https://github.com/Kailaswadje)
- LinkedIn: [linkedin.com/in/kwadaje](https://www.linkedin.com/in/kwadaje/)

---

⭐ If the State → Node → Updated State model finally made graph-based agents click, consider giving it a star!
