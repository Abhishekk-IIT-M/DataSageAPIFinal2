import logging
from typing import Dict, Any
import json
from langchain_agent import LangChainAgent, analyze_sales_data, analyze_scraped_movie_data

logger = logging.getLogger(__name__)


class DataAnalystAgent:

    def __init__(self):
        self.langchain_agent = LangChainAgent()
        logger.info("DataAnalystAgent initialized.")

    def run(self, question: str, files: Dict[str, str]) -> Any:
        logger.info("Executing task with manual routing logic.")

        # Manual Routing Logic
        try:
            if "sales" in question.lower(
            ) or "sample-sales.csv" in question.lower():
                logger.info("ROUTING: Sales analysis task detected.")
                file_path = files.get("sample-sales.csv")
                if not file_path:
                    return {
                        "error":
                        "Required file 'sample-sales.csv' was not provided."
                    }
                return json.loads(analyze_sales_data(file_path))

            elif "network" in question.lower(
            ) or "edges.csv" in question.lower():
                logger.info("ROUTING: Network analysis task detected.")
                file_path = files.get("edges.csv")
                if not file_path:
                    return {
                        "error": "Required file 'edges.csv' was not provided."
                    }
                # We need to create analyze_network_data in the other file
                from langchain_agent import analyze_network_data
                return json.loads(analyze_network_data(file_path))

            elif "films" in question.lower() or "wikipedia" in question.lower(
            ):
                logger.info(
                    "ROUTING: Movie analysis task detected. Using agent.")
                response = self.langchain_agent.agent_executor.invoke(
                    {"input": question})
                final_output = response.get('output', '')

            else:
                logger.warning(
                    "ROUTING: No specific keywords found. Using default agent."
                )
                response = self.langchain_agent.agent_executor.invoke(
                    {"input": question})
                final_output = response.get('output', '')

            return json.loads(final_output)

        except Exception as e:
            logger.error(f"An error occurred in the agent's run method: {e}",
                         exc_info=True)
            raise
