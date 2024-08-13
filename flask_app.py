import json
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

import src.config as cfg
from src.csv_retriever import CSVRetriever
from src.qna import QnA, QnAResponse
from src.rag import RAG
from src.utils.conversation import parse_llm_messages

app = Flask(__name__)
CORS(app)


def get_qna():
    default_model = cfg.llm_options['azure-openai'].get('llm')

    rag = RAG(model=default_model, rerank=cfg.rerank)

    csv_retriever = CSVRetriever(
        llm=default_model,
        directory_path='./data/preprocessed/csv/'
    )

    qa = QnA(
        model=default_model,
        retriever=rag.retriever,
        data_retriever=csv_retriever
    )

    return qa


@app.route('/chat', methods=['POST'])
def chat():
    input_json = request.get_json(force=True)
    query = input_json["query"]
    session_id = input_json.get('session_id', '')

    qa: QnA = get_qna()

    response: QnAResponse = qa.ask_question(
        query=query,
        session_id=session_id,
        stream=False
    )

    return jsonify({
        "response": {
            "answer": response['answer'],
            "chat_history": parse_llm_messages(response['chat_history']),
            "context": [
                {
                    "source": doc.metadata['source'],
                    "relevance_score": doc.metadata['relevance_score'],
                    "page_content": doc.page_content[:100]
                }
                for doc in response['context']
            ]
        },
        "status": "Success"
    })


def fake_stream():
    for i in range(10):
        data = json.dumps({
            "response": {
                "answer": f"Answer {i}",
                "chat_history": [],
                "context": []
            },
            "status": "Success"
        })

        #  Convert the event object to a string
        msg = f'id: {1}\ndata: {data}\n\n'

        yield msg


def stream_parser(response):
    context = []
    chat_history = []

    for chunk in response:
        if 'context' in chunk:
            for doc in chunk['context']:
                context.append([
                    {
                        "source": doc.metadata['source'],
                        "relevance_score": doc.metadata['relevance_score'],
                        "page_content": doc.page_content[:100]
                    }
                ])

        if 'chat_history' in chunk:
            chat_history = parse_llm_messages(chunk['chat_history'])

        answer = chunk.get('answer', '')
        data = json.dumps({
            "response": {
                "answer": answer,
                "chat_history": chat_history,
                "context": context
            },
            "status": "Success"
        })

        #  Convert the event object to a string
        msg = f'id: {1}\ndata: {data}\n\n'

        yield msg


@app.route('/stream', methods=['POST'])
def stream():
    input_json = request.get_json(force=True)
    query = input_json["query"]
    session_id = input_json.get('session_id', '')

    qa: QnA = get_qna()

    # response = None
    response: QnAResponse = qa.ask_question(
        query=query,
        session_id=session_id,
        stream=True
    )

    if response:
        return app.response_class(stream_parser(response), content_type='text/event-stream')
    else:
        return app.response_class(fake_stream(), content_type='text/event-stream')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=os.getenv('PORT', 5000), debug=True)
