import json
import os
import tempfile
from datetime import datetime

from flask import Request, abort
from google.cloud import firestore, storage
from langchain.chat_models import ChatOpenAI
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from llama_index import (LLMPredictor, ServiceContext, PromptTemplate,
                         StorageContext, load_index_from_storage)
from llama_index.indices.base import BaseIndex

FILES_TO_DOWNLOAD = ["docstore.json", "index_store.json", "vector_store.json"]
BUCKET_NAME = "udemy-vector-store"

line_bot_api = LineBotApi(os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])
db = firestore.Client()

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

USE_HISTORY = False


def get_object_updated_time(object_name: str) -> datetime:
    blob = bucket.blob(object_name)
    blob.reload()
    return blob.updated


def download_files_from_storage(temp_dir: str) -> None:
    for file in FILES_TO_DOWNLOAD:
        blob = bucket.blob(file)
        blob.download_to_filename(f"{temp_dir}/{file}")


def setup_storage_and_load_index() -> BaseIndex:
    with tempfile.TemporaryDirectory() as temp_dir:
        download_files_from_storage(temp_dir)
        service_context = ServiceContext.from_defaults(
            llm_predictor=LLMPredictor(
                llm=ChatOpenAI(
                    temperature=0, model_name="gpt-3.5-turbo", max_tokens=512
                )
            )
        )

        storage_context = StorageContext.from_defaults(persist_dir=temp_dir)
        return load_index_from_storage(
            storage_context=storage_context, service_context=service_context
        )


index = setup_storage_and_load_index()
updated_time = get_object_updated_time(FILES_TO_DOWNLOAD[0])


def reload_index_if_updated() -> None:
    global index, updated_time
    latest_updated_time = get_object_updated_time(FILES_TO_DOWNLOAD[0])
    if updated_time != latest_updated_time:
        index = setup_storage_and_load_index()
        updated_time = latest_updated_time


def get_previous_messages(user_id: str) -> list:
    query = (
        db.collection("users")
        .document(user_id)
        .collection("messages")
        .order_by("timestamp", direction=firestore.Query.DESCENDING)
        .limit(3)
    )
    return [
        {
            "chatgpt_response": doc.to_dict()["chatgpt_response"],
            "user_message": doc.to_dict()["user_message"],
        }
        for doc in query.stream()
    ]


def format_history(previous_messages: list) -> str:
    return "".join(
        f"ユーザー: {d['user_message']}\nアシスタント: {d['chatgpt_response']}\n"
        for d in previous_messages[::-1]
    )


def generate_response(user_message: str, history: str) -> str:
    COMMON_PROMPT = """
    あなたは親切なアシスタントです。
    {history_section}
    以下に文献の情報を提供します。

    ---------------------
    {{context_str}}
    ---------------------

    与えられた情報を元にユーザーへのアドバイスを200文字以内で出力してください。
    文献の情報から回答できない入力の場合は、そのように出力してください。

    入力：{{query_str}}

    出力：
    """

    if USE_HISTORY:
        history_section = f"""
        これまでのユーザーとアシスタントの会話の履歴は以下のようになっています。

        ---------------------
        {history}
        ---------------------
        """
    else:
        history_section = ""

    PROMPT = COMMON_PROMPT.format(history_section=history_section)

    query_engine = index.as_query_engine(text_qa_template=PromptTemplate(PROMPT))
    return str(query_engine.query(user_message))


def reply_to_user(reply_token: str, chatgpt_response: str) -> None:
    line_bot_api.reply_message(reply_token, TextSendMessage(text=chatgpt_response))


def save_message_to_db(user_id: str, user_message: str, chatgpt_response: str) -> None:
    doc_ref = db.collection("users").document(user_id).collection("messages").document()
    doc_ref.set(
        {
            "user_message": user_message,
            "chatgpt_response": chatgpt_response,
            "timestamp": datetime.now(),
        }
    )


###################


def main(request: Request) -> str:
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent) -> None:
    reload_index_if_updated()

    user_id = event.source.user_id
    user_message = event.message.text

    if USE_HISTORY:
        previous_messages = get_previous_messages(user_id)
        history = format_history(previous_messages)
    else:
        history = None
    chatgpt_response = generate_response(user_message, history)

    reply_to_user(event.reply_token, chatgpt_response)
    if USE_HISTORY:
        save_message_to_db(user_id, user_message, chatgpt_response)
