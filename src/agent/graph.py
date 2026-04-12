from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from src.agent.state import AgentState

# --- MOCK NODES FOR DEVELOPMENT ---
# Developer 2 will implement the real diagnosis logic
def diagnose_node(state: AgentState):
    print("[Node] Executing diagnosis...")
    return {"learning_gaps": "Mocked learning gaps based on performance data."}

# --- REAL RAG RETRIEVAL NODE (Developer 3) ---
def retrieve_node(state: AgentState):
    """
    Real Retrieval Node — delegates to rag.retriever.run_retrieval_node.

    Reads state["learning_gaps"] (str | list[dict] | DiagnosisReport),
    queries the vectorstore for relevant educational materials, and returns
    a populated resources list for state["resources"].
    """
    print("[Node] Executing RAG retrieval...")
    try:
        from rag.retriever import run_retrieval_node
        result = run_retrieval_node(state)
        print(f"[Node] RAG retrieval complete — {len(result.get('resources', []))} resource(s) found.")
        return result
    except Exception as exc:
        print(f"[Node] RAG retrieval failed ({exc}). Returning empty resources.")
        return {"resources": []}

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
