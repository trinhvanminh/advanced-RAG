from langchain.tools.render import render_text_description
import json
from pathlib import Path

from langchain_community.agent_toolkits import JsonToolkit, create_json_agent
from langchain_community.tools.json.tool import JsonSpec

from src.config import llm_options

directory_path = './data/preprocessed/json'
fn = 'bridging_loans_data.json'

path = Path(directory_path, fn)
"""Create a JsonSpec from a file."""
if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")
dict_ = json.loads(path.read_text())[0]
json_spec = JsonSpec(dict_=dict_)
# print(json_spec)
# json_spec = JsonSpec.from_file(path=path)
json_toolkit = JsonToolkit(spec=json_spec)


llm = llm_options['fireworks'].get('llm')
json_agent_executor = create_json_agent(
    llm=llm,
    toolkit=json_toolkit,
    verbose=True,
    max_iterations=15

)


response = json_agent_executor.run(
    "give me bridging loans fees of the lender name Adelaide")


print(response)
