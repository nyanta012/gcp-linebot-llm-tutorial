import os

import openai
from flask import Request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

line_bot_api = LineBotApi(os.environ["LINE_CHANNEL_ACCESS_TOKEN"])
handler = WebhookHandler(os.environ["LINE_CHANNEL_SECRET"])


# Cloud Functionsエントリポイント
def main(request: Request) -> str:
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"


# LINEにRequestがあった時に実行される。
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event: MessageEvent) -> None:
    messages = [
        {"role": "system", "content": "あなたは親切な人工知能です"},
        {"role": "user", "content": event.message.text},
    ]
    chatgpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )

    reply_text = chatgpt_response.choices[0].message["content"]

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_text))
