from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt_file(data: str, filename: str = "output.txt"):
    timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"---- Final Output ---- \nTimestamp: {timestamp_now}\n\n{data}\n\n"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(formatted_text)
        
        return f"Data saved successfully to {filename}."
    
    except Exception as e:
        return f"Failed to save data: {str(e)}"
    

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt_file,
    description="Saves structured research data to a text file."
)

search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for info."
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)