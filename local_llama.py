# Locally Downloaded Weights Version
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/SimpleIndexDemoLlama-Local/
# https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-2

import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import PromptTemplate

# Model names (make sure you have access on HF)
LLAMA2_7B = "/home/llama/llama-2-7b-hf/"
# LLAMA2_7B = "meta-llama/Llama-2-7b-hf"
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

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=2048,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=selected_model,
    model_name=selected_model,
    device_map="auto",
    # change these settings below depending on your GPU
    model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True},
)


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


from llama_index.core import Settings

Settings.llm = llm
Settings.embed_model = embed_model


from llama_index.core import SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader("data/").load_data()


from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)


# set Logging to DEBUG for more detailed outputs
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(f"{response}")


# Streaming
import time

query_engine = index.as_query_engine(streaming=True)
response = query_engine.query("What happened at interleaf?")

start_time = time.time()

token_count = 0
for token in response.response_gen:
    print(token, end="")
    token_count += 1

time_elapsed = time.time() - start_time
tokens_per_second = token_count / time_elapsed

print(f"\n\nStreamed output at {tokens_per_second} tokens/s")
