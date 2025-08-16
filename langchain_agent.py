import os
import pandas as pd
import logging
import io
import base64
import json
import requests
from typing import Dict, Any

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages, )
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.tools import tool

# Other Module Imports
import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates  # <-- Make sure this import is here
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# --- Tool Definitions ---


@tool
def analyze_weather_data(file_path: str) -> str:
    """
    Reads a CSV file of weather data, performs a complete analysis, and returns a
    JSON object with keys for temperature, precipitation, correlation, and charts.
    Use this tool when the user asks to analyze 'sample-weather.csv'.
    """
    logger.info(f"Using analyze_weather_data tool on file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        average_temp_c = df['temperature_celsius'].mean()
        max_precip_date = df.loc[
            df['precipitation_mm'].idxmax()]['date'].strftime('%Y-%m-%d')
        min_temp_c = df['temperature_celsius'].min()
        temp_precip_correlation = df['temperature_celsius'].corr(
            df['precipitation_mm'])
        average_precip_mm = df['precipitation_mm'].mean()

        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df['temperature_celsius'], color='red')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.gcf().autofmt_xdate()
        plt.title('Temperature Over Time')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        temp_line_chart_base64 = base64.b64encode(
            buf.getvalue()).decode('utf-8')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(df['precipitation_mm'], bins=10, color='orange')
        plt.title('Precipitation Distribution')
        plt.xlabel('Precipitation (mm)')
        plt.ylabel('Frequency')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        precip_histogram_base64 = base64.b64encode(
            buf.getvalue()).decode('utf-8')
        plt.close()

        result = {
            "average_temp_c": float(average_temp_c),
            "max_precip_date": max_precip_date,
            "min_temp_c": int(min_temp_c),
            "temp_precip_correlation": float(temp_precip_correlation),
            "average_precip_mm": float(average_precip_mm),
            "temp_line_chart": temp_line_chart_base64,  # Corrected base64
            "precip_histogram": precip_histogram_base64  # Corrected base64
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error during weather analysis: {e}"


@tool
def analyze_sales_data(file_path: str) -> str:
    """
    Reads a CSV file of sales data and returns a JSON object with total_sales,
    top_region, correlation, and two charts. Use for 'sample-sales.csv'.
    """
    logger.info(f"Using analyze_sales_data tool on file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_month'] = df['date'].dt.day

        total_sales = df['sales'].sum()
        region_sales = df.groupby('region')['sales'].sum()
        top_region = region_sales.idxmax()
        day_sales_correlation = df['day_of_month'].corr(df['sales'])
        median_sales = df['sales'].median()
        total_sales_tax = total_sales * 0.10

        plt.figure(figsize=(8, 6))
        region_sales.plot(kind='bar', color='blue')
        plt.title('Total Sales by Region')
        plt.xlabel('Region')
        plt.ylabel('Total Sales')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        bar_chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        df_sorted = df.sort_values('date')
        df_sorted['cumulative_sales'] = df_sorted['sales'].cumsum()
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['date'], df_sorted['cumulative_sales'], color='red')
        plt.title('Cumulative Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        cumulative_chart_base64 = base64.b64encode(
            buf.getvalue()).decode('utf-8')
        plt.close()

        result = {
            "total_sales": int(total_sales),
            "top_region": top_region,
            "day_sales_correlation": float(day_sales_correlation),
            "bar_chart": bar_chart_base64,  # Corrected base64
            "median_sales": int(median_sales),
            "total_sales_tax": int(total_sales_tax),
            "cumulative_sales_chart":
            cumulative_chart_base64  # Corrected base64
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error during sales analysis: {e}"


@tool
def analyze_scraped_movie_data(file_path: str) -> str:
    """
    Reads a CSV of movie data, performs analysis for the four specific movie questions,
    and returns the final formatted JSON array. Use for the Wikipedia movie data task.
    """
    logger.info(f"Using analyze_scraped_movie_data tool on file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        df['Worldwide gross'] = df['Worldwide gross'].astype(str).str.replace(
            r'[$,]', '', regex=True)
        df['Worldwide gross'] = df['Worldwide gross'].str.replace(r'^[A-Z]+',
                                                                  '',
                                                                  regex=True)
        df['Worldwide gross'] = pd.to_numeric(df['Worldwide gross'],
                                              errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df.dropna(subset=['Year', 'Worldwide gross'], inplace=True)
        df['Year'] = df['Year'].astype(int)
        df['Rank'] = pd.to_numeric(df['Rank'], errors='coerce')
        df['Peak'] = pd.to_numeric(df['Peak'], errors='coerce')

        answer1 = len(df[(df['Worldwide gross'] >= 2_000_000_000)
                         & (df['Year'] < 2000)])
        movies_over_1_5bn = df[df['Worldwide gross'] > 1_500_000_000]
        answer2 = movies_over_1_5bn.loc[movies_over_1_5bn['Year'].idxmin(
        )]['Title'] if not movies_over_1_5bn.empty else "No film found"
        answer3 = df['Rank'].corr(df['Peak'])

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='Rank', y='Peak')
        m, b = np.polyfit(df['Rank'], df['Peak'], 1)
        plt.plot(df['Rank'], m * df['Rank'] + b, color='red', linestyle='--')
        plt.title('Rank vs. Peak of Highest-Grossing Films')
        plt.xlabel('Rank')
        plt.ylabel('Peak')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        answer4 = f"data:image/png;base64,{image_base64}"
        plt.close()

        final_result = [answer1, answer2, float(answer3), answer4]
        return json.dumps(final_result)
    except Exception as e:
        return f"Error during movie analysis: {e}"


@tool
def analyze_network_data(file_path: str) -> str:
    """
    Reads an edge list from a CSV file, analyzes network properties, and returns a
    JSON object. Use for the 'sample-network' task.
    """
    logger.info(f"Using analyze_network_data tool on file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        G = nx.from_pandas_edgelist(df, 'source', 'target')

        edge_count = G.number_of_edges()
        degrees = dict(G.degree())
        highest_degree_node = max(degrees, key=degrees.get)
        average_degree = sum(degrees.values()) / G.number_of_nodes()
        density = nx.density(G)
        shortest_path_alice_eve = nx.shortest_path_length(G,
                                                          source='Alice',
                                                          target='Eve')

        plt.figure(figsize=(8, 6))
        nx.draw(G,
                with_labels=True,
                node_color='skyblue',
                node_size=2000,
                edge_color='gray')
        plt.title('Network Graph')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        network_graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.hist(list(degrees.values()),
                 bins=range(1,
                            max(degrees.values()) + 2),
                 align='left',
                 color='green')
        plt.title('Degree Distribution')
        plt.xlabel('Degree')
        plt.ylabel('Number of Nodes')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=75)
        buf.seek(0)
        degree_histogram_base64 = base64.b64encode(
            buf.getvalue()).decode('utf-8')
        plt.close()

        result = {
            "edge_count": edge_count,
            "highest_degree_node": highest_degree_node,
            "average_degree": float(average_degree),
            "density": float(density),
            "shortest_path_alice_eve": shortest_path_alice_eve,
            "network_graph": network_graph_base64,  # Corrected base64
            "degree_histogram": degree_histogram_base64  # Corrected base64
        }
        return json.dumps(result)
    except Exception as e:
        return f"Error during network analysis: {e}"


@tool
def web_scraper(url: str) -> str:
    """
    Scrapes the first table from a URL, saves it to 'temp_data.csv',
    and returns a summary including the filename.
    """
    cleaned_url = url.strip().strip("'\"")
    logger.info(f"Using web_scraper tool for cleaned URL: {cleaned_url}")
    try:
        headers = {
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(cleaned_url, headers=headers, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        if tables:
            df = tables[0]
            temp_file_path = "temp_data.csv"
            df.to_csv(temp_file_path, index=False)
            summary = {
                "message":
                f"Successfully scraped and saved data to {temp_file_path}"
            }
            return json.dumps(summary)
        else:
            return "Error: No tables found at the specified URL."
    except Exception as e:
        return f"Error: Failed to scrape the URL. Reason: {e}"


@tool
def duckdb_sql_querier(query: str) -> str:
    """
    Executes a DuckDB SQL query.
    """
    logger.info(f"Using duckdb_sql_querier tool with query: {query[:100]}...")
    try:
        con = duckdb.connect(database=':memory:')
        result = con.execute(query).fetchall()
        return str(result)
    except Exception as e:
        return f"Error: DuckDB query failed. Reason: {e}"


class LangChainAgent:

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini",
                              temperature=0,
                              openai_api_key=os.getenv('OPENAI_API_KEY'))
        self.tools = [
            web_scraper, analyze_scraped_movie_data, analyze_sales_data,
            analyze_network_data, analyze_weather_data, duckdb_sql_querier
        ]

        llm_with_tools = self.llm.bind_tools(self.tools)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a helpful data analyst. You must use the provided tools to answer the user's question. For multi-step tasks like web scraping, first use the web_scraper tool, then use the appropriate analysis tool on the resulting 'temp_data.csv' file."
             ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = ({
            "input":
            lambda x: x["input"],
            "agent_scratchpad":
            lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        }
                 | self.prompt
                 | llm_with_tools
                 | OpenAIToolsAgentOutputParser())

        self.agent_executor = AgentExecutor(agent=agent,
                                            tools=self.tools,
                                            verbose=True,
                                            handle_parsing_errors=True)
        logger.info("LangChain Tool-Calling Agent initialized.")
