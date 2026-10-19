import os
import streamlit as st
from dotenv import load_dotenv
from src.graph import create_market_intelligence_graph

# 1. Page Configuration
load_dotenv()
st.set_page_config(
    page_title="Market & Competitive Intelligence Agent",
    page_icon="💼",
    layout="wide"
)

# 2. UI Header & Description
st.title("💼 Autonomous Market & Competitive Intelligence Agent")
st.markdown(
    "An autonomous, stateful agent built with **LangGraph**, **Groq**, and **Tavily**. "
    "It autonomously formulates research hypotheses, gathers live corporate disclosures, "
    "audits evidence for entity conflicts, and synthesizes an executive brief."
)

# 3. Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    target_company = st.text_input("Target Company / Entity:", value="Figma", placeholder="e.g. Stripe, Databricks, Notion")
    
    st.markdown("---")
    st.markdown("### Agent Architecture")
    st.markdown(
        """
        - **Planner:** Generates search angles via structured schemas.
        - **Researcher:** Live search retrieval via Tavily.
        - **Auditor:** Self-reflection loop detecting entity collisions & gaps.
        - **Synthesizer:** Markdown report generation with cited sources.
        """
    )
    
    st.caption("Powered by LPU inference on Groq & LangGraph state machines.")

# 4. Agent Execution Trigger
if st.button("🚀 Run Competitive Intelligence Scan", type="primary", use_container_width=True):
    if not target_company.strip():
        st.warning("Please enter a valid company name.")
        st.stop()

    # Initialize Graph & Session Data
    agent = create_market_intelligence_graph()
    initial_state = {
        "company": target_company.strip(),
        "search_queries": [],
        "search_results": [],
        "is_sufficient": False,
        "critique": "",
        "iteration_count": 0,
        "final_report": ""
    }

    st.subheader(f"🔍 Live Intelligence Audit: {target_company}")

    # Interactive progress timeline
    with st.status("Agent workflow initializing...", expanded=True) as status_box:
        final_report = ""

        # Stream node-by-node updates from LangGraph
        for step_output in agent.stream(initial_state):
            for node_name, node_state in step_output.items():

                if node_name == "planner":
                    status_box.update(label="Planning research strategy...", state="running")
                    st.markdown("#### 📋 Strategic Search Plan")
                    queries = node_state.get("search_queries", [])
                    for q in queries:
                        st.markdown(f"- 🔎 `{q}`")

                elif node_name == "researcher":
                    status_box.update(label="Querying live internet sources...", state="running")
                    st.markdown("#### 🌐 Live Evidence Retrieval")
                    findings = node_state.get("search_results", [])
                    st.success(f"Retrieved and processed {len(findings)} source snippets.")

                elif node_name == "reflection":
                    status_box.update(label="Auditing evidence consistency & gaps...", state="running")
                    st.markdown("#### 🧠 Self-Reflection & Critique")
                    is_sufficient = node_state.get("is_sufficient", False)
                    critique = node_state.get("critique", "No critique provided.")
                    
                    if is_sufficient:
                        st.info(f"**Verification Result:** Sufficient Evidence\n\n{critique}")
                    else:
                        st.warning(f"**Verification Result:** Gaps / Conflicts Flagged\n\n{critique}")
                        if "search_queries" in node_state and node_state["search_queries"]:
                            st.markdown(f"**Planned Loop-Back Query:** `{node_state['search_queries'][0]}`")

                elif node_name == "synthesizer":
                    status_box.update(label="Drafting final executive briefing...", state="running")
                    final_report = node_state.get("final_report", "")

        status_box.update(label="✅ Competitive Intelligence Audit Completed!", state="complete", expanded=False)

    # 5. Display Final Synthesized Output
    if final_report:
        st.markdown("---")
        st.header("📑 Executive Intelligence Brief")
        st.markdown(final_report)

        # Download capability for generated artifact
        clean_filename = f"{target_company.lower().replace(' ', '_')}_intel_brief.md"
        st.download_button(
            label="📥 Download Executive Brief (Markdown)",
            data=final_report,
            file_name=clean_filename,
            mime="text/markdown",
            use_container_width=True
        )