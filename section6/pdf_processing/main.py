import glob
import json
import os
import tempfile

from google.cloud import storage
from google.cloud.functions_v1.context import Context
from langchain.chat_models import ChatOpenAI
from llama_index import (GPTVectorStoreIndex, LLMPredictor, ServiceContext,
                         StorageContext, download_loader)

SOURCE_BUCKET_NAME = "udemy-pdf-store"
TARGET_BUCKET_NAME = "udemy-vector-store"


def download_pdf_from_bucket(
    bucket: storage.Bucket, filename: str, destination_path: str
) -> None:
    blob = bucket.blob(filename)
    blob.download_to_filename(destination_path)


def vectorize_pdf(pdf_filepath: str, vector_filepath: str) -> None:
    CJKPDFReader = download_loader("CJKPDFReader")
    loader = CJKPDFReader()
    documents = loader.load_data(file=pdf_filepath)
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        )
    )
    index = GPTVectorStoreIndex.from_documents(
        documents, service_context=service_context
    )
    index.storage_context.persist(persist_dir=vector_filepath)


def upload_to_bucket(bucket: storage.Bucket, directory: str) -> None:
    files = glob.glob(os.path.join(directory, "*.json"))
    for file_name in files:
        blob = bucket.blob(os.path.basename(file_name))
        blob.upload_from_filename(file_name)


###################


def main(event: dict, context: Context) -> str:
    client = storage.Client()
    source_bucket = client.get_bucket(SOURCE_BUCKET_NAME)

    pdf_name = os.path.basename(event["name"])

    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_filepath = os.path.join(temp_dir, pdf_name)
        vector_filepath = os.path.join(temp_dir, "storage")

        download_pdf_from_bucket(source_bucket, pdf_name, pdf_filepath)
        vectorize_pdf(pdf_filepath, vector_filepath)

        target_bucket = client.get_bucket(TARGET_BUCKET_NAME)
        upload_to_bucket(target_bucket, os.path.join(temp_dir, "storage"))

    return "OK"
