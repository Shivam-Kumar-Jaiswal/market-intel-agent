import os
from datetime import datetime
from dotenv import load_dotenv
from src.graph import create_market_intelligence_graph

load_dotenv()

def run_agent():
    print("🚀 Initializing Autonomous Market Intelligence Agent...")
    agent = create_market_intelligence_graph()

    target_company = "Figma"  # Try testing another company like Figma, Stripe, or Databricks

    initial_state = {
        "company": target_company,
        "search_queries": [],
        "search_results": [],
        "is_sufficient": False,
        "critique": "",
        "iteration_count": 0,
        "final_report": ""
    }

    print(f"\n🎯 Target: Investigating '{target_company}'...")
    final_output = agent.invoke(initial_state)

    # 1. Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)

    # 2. Generate a clean filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = target_company.lower().replace(" ", "_")
    filename = f"outputs/{clean_name}_intel_{timestamp}.md"

    # 3. Save the report to disk
    with open(filename, "w", encoding="utf-8") as f:
        f.write(final_output["final_report"])

    print("\n" + "="*80)
    print(f"✅ Intelligence Report successfully generated and saved to: {filename}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_agent()