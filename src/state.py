from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    # 1. User Input
    company: str                          # e.g., "Notion" or "Stripe"
    
    # 2. Planning Phase
    search_queries: List[str]             # List of search queries the planner generates
    
    # 3. Research Phase (Memory)
    # Annotated with operator.add means when nodes return new results, 
    # LangGraph APPENDS them to the list instead of overwriting existing findings.
    search_results: Annotated[List[dict], operator.add] 
    
    # 4. Evaluation & Reflection Phase
    is_sufficient: bool                   # True if we have enough info, False if gaps exist
    critique: str                         # What is missing? (used to refine the next loop)
    iteration_count: int                  # Safety counter to stop infinite loops (e.g., max 2 or 3)
    
    # 5. Final Output
    final_report: str                     # The compiled executive markdown brief