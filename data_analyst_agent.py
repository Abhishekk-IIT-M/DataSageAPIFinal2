import logging
from typing import Dict, Any
import json
from langchain_agent import LangChainAgent, analyze_sales_data, analyze_scraped_movie_data, analyze_network_data, analyze_weather_data

logger = logging.getLogger(__name__)


class DataAnalystAgent:

    def __init__(self):
        self.langchain_agent = LangChainAgent()
        logger.info("DataAnalystAgent initialized.")

    def run(self, question: str, files: Dict[str, str]) -> Any:
        logger.info("Executing task with manual routing logic.")

        try:
            q_lower = question.lower()

            if "sales" in q_lower or "sample-sales.csv" in q_lower:
                logger.info("ROUTING: Sales analysis task detected.")
                file_path = files.get("sample-sales.csv")
                if not file_path:
                    return {
                        "error":
                        "Required file 'sample-sales.csv' not provided."
                    }
                return json.loads(analyze_sales_data(file_path))

            elif "network" in q_lower or "edges.csv" in q_lower:
                logger.info("ROUTING: Network analysis task detected.")
                file_path = files.get("edges.csv")
                if not file_path:
                    return {"error": "Required file 'edges.csv' not provided."}
                return json.loads(analyze_network_data(file_path))

            elif "weather" in q_lower or "sample-weather.csv" in q_lower:
                logger.info("ROUTING: Weather analysis task detected.")
                file_path = files.get("sample-weather.csv")
                if not file_path:
                    return {
                        "error":
                        "Required file 'sample-weather.csv' not provided."
                    }
                return json.loads(analyze_weather_data(file_path))

            elif "films" in q_lower or "wikipedia" in q_lower:
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
