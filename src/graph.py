from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.nodes import (
    planner_node, 
    research_node, 
    reflection_node, 
    report_node, 
    should_continue
)

def create_market_intelligence_graph():
    """
    Compiles the agentic state graph with a self-reflection loop.
    """
    workflow = StateGraph(AgentState)

    # 1. Register all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("reflection", reflection_node)
    workflow.add_node("synthesizer", report_node)

    # 2. Add fixed edges
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "reflection")

    # 3. Add the conditional loop edge
    # After reflection, call should_continue() to decide: loop back to researcher OR go to synthesizer
    workflow.add_conditional_edges(
        "reflection",
        should_continue,
        {
            "researcher": "researcher",
            "synthesizer": "synthesizer"
        }
    )

    # 4. Synthesizer completes the run
    workflow.add_edge("synthesizer", END)

    return workflow.compile()