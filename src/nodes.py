import os
from typing import List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from src.state import AgentState
from src.tools import search_web

load_dotenv()

# 1. Initialize the Groq LLM (High-speed, top-tier reasoning)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2  # Low temperature for factual consistency
)

# 2. Pydantic schema to force structured output from the LLM
class SearchPlan(BaseModel):
    queries: List[str] = Field(
        description="A list of exactly 3 distinct search queries targeting: 1) recent product releases/features, 2) pricing or packaging updates, 3) key strategic moves or leadership/acquisition news."
    )
# --- Add this Pydantic Schema near your SearchPlan schema in src/nodes.py ---

class ReflectionEvaluation(BaseModel):
    is_sufficient: bool = Field(
        description="True if the findings cleanly answer recent products, pricing, and strategic moves for the specific target company. False if there are entity ambiguities, missing pricing data, or sparse results."
    )
    critique: str = Field(
        description="Summary of what is missing, dubious, or conflicting (e.g. 'Found another company with the same name; need queries specific to Notion Labs productivity tool')."
    )
    suggested_follow_up_query: str = Field(
        description="A refined search query to fill the missing gap if is_sufficient is False. Empty string if True."
    )


# --- Add the Reflection Node ---

def reflection_node(state: AgentState) -> dict:
    """
    Evaluates current evidence for completeness, entity accuracy, and gaps.
    Decides whether to trigger an extra targeted search loop.
    """
    company = state["company"]
    findings = state["search_results"]
    current_iteration = state.get("iteration_count", 0) + 1

    print(f"\n🧠 [Reflection] Auditing evidence for {company} (Iteration {current_iteration})...")

    # Format findings snippet titles for evaluation
    summaries = [f"- {item['title']}: {item['content'][:180]}..." for item in findings[:8]]
    evidence_text = "\n".join(summaries)

    prompt = f"""Target Company: {company}
Current Research Evidence:
{evidence_text}

Evaluate if this evidence is sufficient, current, and genuinely refers to '{company}'.
Check for:
1. Entity confusion (e.g. Another company with the same name).
2. Missing details on pricing or recent product moves.

If gaps exist, mark is_sufficient as False and suggest ONE precise follow-up query."""

    eval_llm = llm.with_structured_output(ReflectionEvaluation)
    evaluation: ReflectionEvaluation = eval_llm.invoke([
        SystemMessage(content="You are a strict fact-checker and research quality auditor."),
        HumanMessage(content=prompt)
    ])

    print(f"📊 [Reflection Verdict] Sufficient: {evaluation.is_sufficient}")
    print(f"💬 [Critique]: {evaluation.critique}")

    updates = {
        "is_sufficient": evaluation.is_sufficient,
        "critique": evaluation.critique,
        "iteration_count": current_iteration
    }

    # If gaps are flagged, queue the follow-up query for the research node
    if not evaluation.is_sufficient and evaluation.suggested_follow_up_query:
        print(f"🔄 [Follow-up Planned]: {evaluation.suggested_follow_up_query}")
        updates["search_queries"] = [evaluation.suggested_follow_up_query]

    return updates


# --- Routing Logic (Conditional Edge) ---

def should_continue(state: AgentState) -> str:
    """
    Directs the graph track:
    - Loop back to 'researcher' if evidence has gaps and iterations < 2.
    - Otherwise, proceed to 'synthesizer'.
    """
    if state["iteration_count"] >= 2:
        print("🛑 [Router] Max iteration budget reached. Proceeding to report.")
        return "synthesizer"

    if not state.get("is_sufficient", True):
        print("🔁 [Router] Information gaps detected. Looping back to search...")
        return "researcher"

    print("✅ [Router] Research verified. Proceeding to report.")
    return "synthesizer"

# -------------------------------------------------------------------------
# Node 1: The Planner
# -------------------------------------------------------------------------
def planner_node(state: AgentState) -> dict:
    """
    Analyzes the target company and generates 3 targeted search queries.
    Uses Pydantic structured output so the LLM returns a strict Python list.
    """
    company = state["company"]
    print(f"\n🔍 [Planner] Designing research strategy for: {company}...")

    # Bind the Pydantic schema to the LLM
    structured_llm = llm.with_structured_output(SearchPlan)

    prompt = f"""You are a senior market intelligence analyst. 
Target Company: {company}

Formulate exactly 3 distinct, high-impact web search queries to gather recent intelligence on:
1. Latest product/feature launches and roadmap updates.
2. Pricing changes, packaging tier shifts, or monetization updates.
3. Major strategic announcements, leadership hires, or acquisitions.

Keep queries concise, targeted, and optimized for search engines."""

    response: SearchPlan = structured_llm.invoke([
        SystemMessage(content="You generate strategic search queries for competitive intelligence."),
        HumanMessage(content=prompt)
    ])

    print(f"📋 [Planner] Generated Queries: {response.queries}")
    return {"search_queries": response.queries}


# -------------------------------------------------------------------------
# Node 2: The Researcher
# -------------------------------------------------------------------------
def research_node(state: AgentState) -> dict:
    """
    Iterates through the generated queries, runs web searches using Tavily,
    and returns the collected findings.
    """
    queries = state["search_queries"]
    all_findings = []

    print(f"\n🌐 [Researcher] Executing {len(queries)} search tasks...")

    for query in queries:
        print(f"   -> Searching: '{query}'")
        results = search_web(query=query, max_results=2)
        all_findings.extend(results)

    print(f"✅ [Researcher] Collected {len(all_findings)} verified search snippets.")
    return {"search_results": all_findings}


# -------------------------------------------------------------------------
# Node 3: The Report Synthesizer
# -------------------------------------------------------------------------
def report_node(state: AgentState) -> dict:
    """
    Ingests all accumulated search snippets and synthesizes an executive
    competitive intelligence brief with citations.
    """
    company = state["company"]
    findings = state["search_results"]

    print(f"\n✍️  [Synthesizer] Compiling executive report for {company}...")

    # Format the raw snippets into clean text context with source numbering
    context_blocks = []
    for i, item in enumerate(findings, start=1):
        context_blocks.append(
            f"Source [{i}]: {item['title']}\nURL: {item['url']}\nContent: {item['content']}\n"
        )
    formatted_context = "\n---\n".join(context_blocks)

    prompt = f"""You are an executive competitive intelligence advisor.
Analyze the provided web findings for {company} and generate a structured executive brief in Markdown.

RAW EVIDENCE:
{formatted_context}

INSTRUCTIONS:
1. Synthesize the findings into clear, high-density business insights.
2. Structure the brief using these Markdown headers:
   - ## Executive Summary
   - ## Product & Feature Releases
   - ## Pricing & Packaging Strategy
   - ## Strategic Moves & Leadership
   - ## Key Sources & References (Include markdown links: [Title](URL))
3. If specific information for a section was not found in the evidence, explicitly state: "No recent updates detected in public disclosures."
4. Maintain a professional, objective tone."""

    response = llm.invoke([
        SystemMessage(content="You are an expert market intelligence analyst writing executive briefs."),
        HumanMessage(content=prompt)
    ])

    return {"final_report": response.content}