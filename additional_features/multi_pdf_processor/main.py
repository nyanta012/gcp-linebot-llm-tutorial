import glob
import os
import tempfile
from typing import List

from google.cloud import storage
from google.cloud.functions_v1.context import Context
from langchain.chat_models import ChatOpenAI
from llama_index import (GPTVectorStoreIndex, LLMPredictor, ServiceContext,
                         SimpleDirectoryReader, StorageContext, download_loader)

SOURCE_BUCKET_NAME = "udemy-pdf-store"
TARGET_BUCKET_NAME = "udemy-vector-store"


def download_pdf_from_bucket(bucket: storage.Bucket, filename: str, destination_path: str) -> None:
    blob = bucket.blob(filename)
    blob.download_to_filename(destination_path)


def vectorize_all_pdfs_in_directory(directory: str, vector_filepath: str) -> None:
    reader = SimpleDirectoryReader(directory)
    documents = reader.load_data()
    
    service_context = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        )
    )
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
    index.storage_context.persist(persist_dir=vector_filepath)


def upload_to_bucket(bucket: storage.Bucket, directory: str) -> None:
    files = glob.glob(os.path.join(directory, "*.json"))
    for file_name in files:
        blob = bucket.blob(os.path.basename(file_name))
        blob.upload_from_filename(file_name)


def list_pdfs_in_bucket(bucket: storage.Bucket) -> List[storage.Blob]:
    blobs = bucket.list_blobs()
    return [blob for blob in blobs if blob.name.endswith('.pdf')]


def download_blobs_to_folder(blobs: List[storage.Blob], destination_folder: str) -> None:
    for blob in blobs:
        destination_file_name = os.path.join(destination_folder, os.path.basename(blob.name))
        blob.download_to_filename(destination_file_name)


###################


def main(event: dict, context: Context) -> str:
    client = storage.Client()
    source_bucket = client.bucket(SOURCE_BUCKET_NAME)

    with tempfile.TemporaryDirectory() as temp_dir:
        vector_filepath = os.path.join(temp_dir, "storage")
        pdfs_filepath = os.path.join(temp_dir, "all_pdf")
        os.makedirs(pdfs_filepath, exist_ok=True)

        pdf_blobs = list_pdfs_in_bucket(source_bucket)

        download_blobs_to_folder(pdf_blobs, pdfs_filepath)

        vectorize_all_pdfs_in_directory(pdfs_filepath, vector_filepath)

        target_bucket = client.bucket(TARGET_BUCKET_NAME)
        upload_to_bucket(target_bucket, vector_filepath)

    return "OK"
