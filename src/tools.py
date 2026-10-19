import os
from typing import List, Dict
from tavily import TavilyClient
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# Initialize client using the environment key
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY is not set. Please add it to your .env file.")

tavily_client = TavilyClient(api_key=api_key)


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Executes a web search query and extracts the cleanest text snippets
    along with sources to save LLM context window.
    """
    try:
        response = tavily_client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_raw_content=False
        )
        
        # Clean the response to only what the agent actually needs
        cleaned_results = []
        for item in response.get("results", []):
            cleaned_results.append({
                "title": item.get("title", "No Title"),
                "url": item.get("url", ""),
                "content": item.get("content", "")
            })
            
        return cleaned_results

    except Exception as e:
        print(f"Error performing search for query '{query}': {e}")
        return []