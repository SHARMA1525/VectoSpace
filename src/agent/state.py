from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    """
    The shared state for the Agentic AI Study Coach.
    All nodes in the LangGraph workflow read from and write to this state.
    """
    messages: List[Any]  # Conversation history
    student_goals: str
    performance_data: Dict[str, Any]
    learning_gaps: Optional[str]
    resources: List[Dict[str, str]]
    study_plan: Optional[str]
    final_report: Optional[Dict[str, Any]]
