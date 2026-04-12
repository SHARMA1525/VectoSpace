from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState

# --- MOCK NODES FOR DEVELOPMENT ---
# Developer 2 will implement the real diagnosis logic
def diagnose_node(state: AgentState):
    print("[Node] Executing diagnosis...")
    return {"learning_gaps": "Mocked learning gaps based on performance data."}

# Developer 3 will implement the real RAG retrieval logic
def retrieve_node(state: AgentState):
    print("[Node] Executing retrieval...")
    return {"resources": [{"title": "Calculus 101", "url": "http://example.com/calc"}]}

# Developer 4 will implement the real planner logic
def planner_node(state: AgentState):
    print("[Node] Executing planner...")
    plan = "Mocked 4-week study plan focusing on Calculus."
    return {"study_plan": plan, "final_report": {"status": "Complete", "plan": plan}}

def build_graph():
    """
    Builds the LangGraph state machine workflow.
    """
    workflow = StateGraph(AgentState)
    
    # 1. Add Nodes
    workflow.add_node("diagnose", diagnose_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("plan", planner_node)
    
    # 2. Add Edges (Linear Flow)
    workflow.set_entry_point("diagnose")
    workflow.add_edge("diagnose", "retrieve")
    workflow.add_edge("retrieve", "plan")
    workflow.add_edge("plan", END)
    
    # 3. Setup checkpointer for memory
    memory = MemorySaver()
    
    # 4. Compile the graph
    app = workflow.compile(checkpointer=memory)
    return app

agent_app = build_graph()
