from llama_index import VectorStoreIndex, SimpleDirectoryReader
  

import torch
from transformers import BitsAndBytesConfig
from llama_index.prompts import PromptTemplate
from llama_index.llms import HuggingFaceLLM
from llama_index import ServiceContext
from llama_index import VectorStoreIndex
from llama_index.response.notebook_utils import display_response
import logging
import sys

#Without quantization, we will not be able to load the whole meta-llama/Llama-2-7b-chat-hf in the free tier of colab.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
documents = SimpleDirectoryReader("./data").load_data()
def messages_to_prompt(messages):
  prompt = ""
  for message in messages:
    if message.role == 'system':
      prompt += f"<|system|>\n{message.content}\n"
    elif message.role == 'user':
      prompt += f"<|user|>\n{message.content}\n"
    elif message.role == 'assistant':
      prompt += f"<|assistant|>\n{message.content}\n"

  # ensure we start with a system prompt, insert blank if needed
  if not prompt.startswith("<|system|>\n"):
    prompt = "<|system|>\n\n" + prompt

  # add final assistant prompt
  prompt = prompt + "<|assistant|>\n"

  return prompt


llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    query_wrapper_prompt=PromptTemplate("<|system|>\n\n<|user|>\n{query_str}\n<|assistant|>\n"),
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    # tokenizer_kwargs={},
    generate_kwargs={"temperature": 0.3, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    device_map="auto",

)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
vector_index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = vector_index.as_query_engine(response_mode="compact")

response = query_engine.query("Hi?")

display_response(response)
