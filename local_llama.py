# Locally Downloaded Weights Version
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoLlama-Local/
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2
import os
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig
from llama_index.core import ChatPromptTemplate
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex


os.environ["CURL_CA_BUNDLE"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Model names (make sure you have access on HF)
LLAMA2_7B = "/home/llama/llama-2-7b-hf/"
selected_model = LLAMA2_7B

SYSTEM_PROMPT = """You are an AI assistant that answers questions in a friendly manner, based on the given source documents. Here are some rules you always follow:
- Generate human readable output, avoid creating output with gibberish text.
- Generate only the requested output, don't include any other language before or after the requested output.
- Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
- Generate professional language typically used in business documents in North America.
- Never generate offensive or foul language.
"""

query_wrapper_prompt = PromptTemplate(
    "[INST]<>\n" + SYSTEM_PROMPT + "<>\n\n{query_str}[/INST] "
)

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_use_double_quant=True,
)

llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.7, "top_k": 10, "top_p": 0.95, "do_sample": True, "num_return_sequences": 1},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"quantization_config": quantization_config},
)

# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.llm = llm
Settings.embed_model = embed_model

# load documents
documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine(llm=llm)
response = query_engine.query("What did the author do growing up?")
print(response)
