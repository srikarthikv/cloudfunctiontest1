#required constants
#KEY_FILE = '/home/azureuser/cloudfiles/code/keys/openai_key.txt'
#KNOWLEDGE_DOCS = '/home/azureuser/cloudfiles/code/Users/richard_malchi/Database/documents'
DEFAULT_EMBED_BATCH_SIZE = 1
EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'
RESOURCE_ENDPOINT = ""
# In[ ]:


# """
# This class provides access to OpenAI models and functionalities for language processing tasks.
# It includes methods for initializing models, embeddings, and chains, loading and processing documents,
# generating responses, and more.
# """
class OpenAI:
    def __init__(self):
        import os
        import openai
        import langchain
        #with open(KEY_FILE, 'r') as file:
            #api_key = file.read().replace('\n', '')

        # Set OpenAI API configuration
        api_key = "d6420559b1154f5d82f8364ec4c77b55"
        openai.api_type = "azure"
        openai.api_base = "https://gpt-demo1.openai.azure.com/"
        openai.api_version = "2022-12-01"
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_API_KEY"] = api_key
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://gpt-demo1.openai.azure.com/"
        os.environ["OPENAI_EMBEDDINGS_ENGINE_DOC"] = "text-embedding-ada"
        os.environ["OPENAI_EMBEDDINGS_ENGINE_QUERY"] = "text-embedding-ada"
        os.environ["OPENAI_API_BASE"] = "https://gpt-demo1.openai.azure.com"
        os.environ["OPENAI_ENGINES"] = "GPT35-Demo1"
        os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
        RESOURCE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

    # """
    # Initialize the OpenAI model.
    # Args:
    #     model_type (str): The type of model to initialize. Defaults to "Chat".
    # Returns:
    #     object: The initialized model object.
    # """
    def init_model(self, model_type="Chat"):
        import openai
        import os
        from langchain.llms import AzureOpenAI
        from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
        api_key="d6420559b1154f5d82f8364ec4c77b55"
        if model_type == "Chat":
            model = AzureChatOpenAI(
                openai_api_base=openai.api_base,
                openai_api_version="2023-03-15-preview",
                deployment_name=os.environ["OPENAI_ENGINES"],
                openai_api_key=api_key,
                openai_api_type=openai.api_type
            )
        else:
            model = AzureOpenAI(
                openai_api_base=openai.api_base,
                deployment_name="text-davinci-003",
                openai_api_key=api_key,
            )
        return model

    # """
    # Initialize the OpenAI embeddings.
    # Returns:
    #     object: The initialized embeddings object.
    # """
    def init_embeddings(self):

        from langchain.embeddings import OpenAIEmbeddings

        embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=1)

        return embedding

    # """
    # Initialize the language model chain.
    # Args:
    #     prompt (str): The prompt to use for the chain.
    # Returns:
    #     object: The initialized language model chain object.
    # """
    def init_chain(self, prompt):
        from langchain.chains import LLMChain

        model = self.init_model(model_type="Chat")

        chatgpt_chain = LLMChain(
            llm=model,
            prompt=prompt,
            verbose=True
        )

        return chatgpt_chain

    # """
    # Load a PDF document.
    # Args:
    #     docpath (str): The path to the PDF document.
    #     online (bool): Indicates whether the document is loaded online. Defaults to False.
    # Returns:
    #     object: The loaded PDF document data.
    # """
    def pdf_loader(self, docpath, online=False):
        if not online:
            from langchain.document_loaders import PyMuPDFLoader
            loader = PyMuPDFLoader(docpath)
            data = loader.load()
            data[0]
        else:
            from langchain.document_loaders import OnlinePDFLoader
            loader = OnlinePDFLoader(docpath)
            data = loader.load()
            data[0]

        return data

    # """
    # Get the appropriate document loader based on the file type.
    # Args:
    #     file_path_or_url (str): The path or URL of the document.
    # Returns:
    #     object: The appropriate document loader object.
    # """
    def get_loader(self, file_path_or_url):
        import mimetypes
        from langchain.document_loaders import TextLoader, BSHTMLLoader, WebBaseLoader, PyMuPDFLoader, CSVLoader, UnstructuredWordDocumentLoader, WebBaseLoader

        if file_path_or_url.startswith("http://") or file_path_or_url.startswith("https://"):
            handle_website = URLHandler()
            return WebBaseLoader(handle_website.extract_links_from_websites([file_path_or_url]))
        else:
            mime_type, _ = mimetypes.guess_type(file_path_or_url)

            if mime_type == 'application/pdf':
                return PyMuPDFLoader(file_path_or_url)
            elif mime_type == 'text/csv':
                return CSVLoader(file_path_or_url)
            elif mime_type in ['application/msword',
                               'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                return UnstructuredWordDocumentLoader(file_path_or_url)
            elif mime_type == 'text/plain':
                return TextLoader(file_path_or_url)
            elif mime_type == 'text/html':
                return BSHTMLLoader(file_path_or_url)
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")

    # """
    # Ingest and process documents.
    # Args:
    #     file_path_or_url (str): The path or URL of the document.
    # Returns:
    #     list: A list of processed documents.
    # """
    def ingest_docs(self, file_path_or_url):
        from langchain.text_splitter import TokenTextSplitter

        loader = self.get_loader(file_path_or_url)
        raw_documents = loader.load()
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents = text_splitter.split_documents(raw_documents)

        return documents

    # """
    # Generate embeddings for the document.
    # Args:
    #     file_path_or_url (str): The path or URL of the document.
    # Returns:
    #     object: The generated embeddings.
    # """
    def embed(self, file_path_or_url):
        from langchain.vectorstores import FAISS
        import os
        import pickle

        embeddings = self.init_embeddings()
        documents = self.ingest_docs(file_path_or_url)
        vectorstore = FAISS.from_documents(documents, embeddings)

        return vectorstore

    # """
    # Generate a response based on the query.
    # Args:
    #     query (str): The query text.
    #     vectorstore (object): The vectorstore object containing document embeddings. Defaults to None.
    #     use_merged (bool): Indicates whether to use merged documents for retrieval. Defaults to True.
    # Returns:
    #     str: The generated response.
    # """
    def generate_response(self, query, vectorstore=None, use_merged=True):
        from langchain.chains import ConversationalRetrievalChain
        from langchain.chains.question_answering import load_qa_chain
        from langchain.prompts import PromptTemplate

        if vectorstore is not None:
            if use_merged:
                model = self.init_model("Chat")
                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
                    {context}
                    Question: {question}
                    Helpful Answer: """
                )
                chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
                docs = vectorstore.similarity_search(query)
                response = chain.run(input_documents=docs, question=query)
            else:
                llm = self.init_model("Chat")
                qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())
                response = qa({"question": query, "chat_history": ""})['answer']
        else:
            llm = self.init_model("native")
            response = llm(query)

        return response


# In[ ]:


class Blob_storage_access:
    """
    This class provides access to Azure Blob storage using BlobServiceClient.
    It allows listing files, downloading blobs, uploading blobs, and reading vector data from the storage.
    """

    def __init__(self):
        """
        Initialize the Blob_storage_access class with the connection string and container name.
        Args:
            connection_string (str): The connection string for the Azure Blob Storage account.
            container_name (str): The name of the container in the Blob Storage account.
        """
        from azure.storage.blob import BlobServiceClient

        connection_string = "DefaultEndpointsProtocol=https;AccountName=gptdemo7020140432;AccountKey=k3z0/JCQH3yV/9iSceGe+s1dtdIUbp8anSUQ/a0sDsrw34tuFHfd7usPn42bCvjaUdzlfpNvA09O+AStCRDO3w==;EndpointSuffix=core.windows.net"
        container_name = "azureml-blobstore-fa29c537-9f94-4f15-8679-5f1e2fd597e4"

        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def list_files(self, path_on_container):
        """
        List files in the specified path in the container.
        Args:
            path_on_container (str): The path in the container.
        Returns:
            list: A list of file paths.
        """
        files = []
        for blob in self.container_client.list_blobs(name_starts_with=path_on_container):
            files.append(blob.name)
        return files

    def download_blob(self, local_path, remote_path):
        """
        Download a blob from the remote path to the local path.
        Args:
            local_path (str): The local path to save the downloaded blob.
            remote_path (str): The remote path of the blob to download.
        Returns:
            str: The local path of the downloaded blob.
        """
        blob_client = self.container_client.get_blob_client(remote_path)
        with open(local_path, "wb") as file:
            blob_data = blob_client.download_blob().readall()
            file.write(blob_data)
        return local_path

    def upload_blob(self, local_path, remote_path):
        """
        Upload a blob from the local path to the remote path.
        Args:
            local_path (str): The local path of the blob to upload.
            remote_path (str): The remote path to save the uploaded blob.
        Returns:
            str: The remote path of the uploaded blob.
        """
        blob_client = self.container_client.get_blob_client(remote_path)
        with open(local_path, "rb") as file:
            blob_client.upload_blob(file,overwrite=True)
        return remote_path


    # """
    # Read the vector data from the specified path on the datastore.
    # Args:
    #     path_on_datastore (str): The path on the datastore to read the vector data from.
    # Returns:
    #     object: The vectorstore object.
    # """
    def read_vector(self,blob_name):
        import pickle

        blob_client = self.container_client.get_blob_client(blob_name)
        blob_data = blob_client.download_blob().readall()
        vectorstore = pickle.loads(blob_data)

        return vectorstore


# In[ ]:


def embed_and_upload(file_list):
    """
    Embeds the provided files and uploads the corresponding vector files to a remote storage.

    Args:
        file_list (str or list): A single file path as a string or a list of file paths.

    Returns:
        str: The remote path of the uploaded vector file.
    """
    import os
    import tempfile
    import pickle

    # Initialize instance of the Blob_storage_access class
    blob_instance = Blob_storage_access()
    openai_instance = OpenAI()

    # Create a temporary directory to store the downloaded documents and vector files
    with tempfile.TemporaryDirectory() as temp_dir:
        if isinstance(file_list, str):  # If a single file is provided
            file_list = [file_list]  # Convert it to a list

        for file in file_list:
            remote_doc = file  # Remote document path
            local_download = os.path.join(temp_dir, os.path.basename(remote_doc))  # Local download path

            # Download the document locally
            blob_instance.download_blob(local_download, remote_doc)

            # Perform embedding on the downloaded document (Replace with appropriate code for OpenAI embedding)
            vectorstore = openai_instance.embed(local_download)

            # Extract document and vector names
            doc_name_w_ext = os.path.basename(local_download)
            doc_name = os.path.splitext(doc_name_w_ext)[0]
            vector_name = doc_name + "_vectorstore.pkl"

            local_vector = os.path.join(temp_dir, vector_name)  # Local vector path

            # Save vectorstore to local file
            with open(local_vector, "wb") as f:
                pickle.dump(vectorstore, f)

            remote_vector = "vectors/" + vector_name
            # Upload vectorstore to remote storage
            remote_upload = blob_instance.upload_blob(local_vector, remote_vector)

    return remote_upload



# In[ ]:


# """
# Method: get_response
# Generates a response to a given query using OpenAI, optionally using a specific vectorstore
# Parameters:
# - query: The query to generate a response for
# - vector_path: Path to a specific vectorstore (default: None)
# - is_merged: Boolean flag indicating whether the vectorstore is merged or not (default: True)
# Returns:
# - response: The generated response
# """


def get_response(query, vector_path=None, use_merged=False):
    blob_instance = Blob_storage_access()  # Blob storage access instance
    openai_instance = OpenAI()  # OpenAI instance

    # Check if vector_path is provided
    if vector_path is not None:
        # Determine the path based on whether it is a merged vectorstore or not
        if use_merged:
            path = "vectors/merged_vectorstore.pkl"  # Path to the merged vectorstore

        else:
            path = vector_path  # Path to the specific vectorstore

        vectorstore = blob_instance.read_vector(path)  # Read the vectorstore from Blob storage
        response = openai_instance.generate_response(query=query, vectorstore=vectorstore)  # Generate response using the provided vectorstore

    else:
        response = openai_instance.generate_response(query=query)  # Generate response without using a specific vectorstore

    return response


# if __name__ == "__main__":
#     main()
