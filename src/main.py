from src.agent.graph import agent_app

def run_interaction(session_id: str, goals: str, perf_data: dict):
    print(f"\n--- Starting Session: {session_id} ---")
    config = {"configurable": {"thread_id": session_id}}
    
    # Define the initial inputs for the state
    initial_state = {
        "messages": [],
        "student_goals": goals,
        "performance_data": perf_data,
        "learning_gaps": None,
        "resources": [],
        "study_plan": None,
        "final_report": None
    }
    
    # Invoke the graph
    print("Invoking Agentic Study Coach...")
    result = agent_app.invoke(initial_state, config=config)
    
    print("\n--- Final Agent State ---")
    print(f"Goals: {result.get('student_goals')}")
    print(f"Gaps Detected: {result.get('learning_gaps')}")
    print(f"Resources Found: {result.get('resources')}")
    print(f"Study Plan Generated: {result.get('study_plan')}")
    print("------------------------\n")

if __name__ == "__main__":
    # Example usage mimicking a student interaction
    student_id = "student_123"
    
    # Simulated input from the Frontend/User
    mock_goals = "I want to prepare for my End-Sem exam in Advanced Mathematics."
    mock_perf = {"quiz_1_score": 45, "quiz_2_score": 60, "weak_areas": ["integration"]}
    
    run_interaction(session_id=student_id, goals=mock_goals, perf_data=mock_perf)
