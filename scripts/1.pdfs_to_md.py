
from pathlib import Path
from dotenv import load_dotenv
import os
from llama_parse import LlamaParse
load_dotenv()

# instruction = """This pdf file contains many tables. Try to be precise while answering the questions"""

parser = LlamaParse(
    api_key=os.getenv('LLAMA_PARSE'),
    result_type="markdown",
    # parsing_instruction=instruction,
    max_timeout=5000,
)

folder_path = './data/'
for file in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file)
    if file.endswith('.pdf'):
        llama_parse_documents = parser.load_data(file_path)

        parsed_doc = "\n\n".join([doc.text for doc in llama_parse_documents])

        document_path = Path(file_path + '.md')
        with document_path.open("w", encoding="utf-8") as f:
            f.write(parsed_doc)
