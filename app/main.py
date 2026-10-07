import os
import time
from openai import OpenAI

try:
    from app.schemas import QuestionExtractionResponse
except ImportError:
    from schemas import QuestionExtractionResponse

# Load system prompt from file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt", "system_prompt.txt")

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

user_content = """
Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to intelligence of humans and other animals. Example tasks in which this is done include speech recognition, computer vision, translation between (natural) languages, as well as other mappings of inputs. 

Machine learning (ML) is a field of inquiry devoted to understanding and building methods that 'learn', that is, methods that leverage data to improve performance on some set of tasks. It is seen as a part of artificial intelligence.

Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Deep learning architectures such as deep neural networks, recurrent neural networks, and transformers have been applied to fields including computer vision, audio recognition, and natural language processing.
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content}
]

start_time = time.perf_counter()
ttft = None
full_response = ""
completion_tokens_count = 0
usage_info = None

stream = client.chat.completions.create(
    model="phi4-mini:3.8b",
    messages=messages,
    response_format={"type": "json_object"},
    stream=True,
    stream_options={"include_usage": True}
)

print("Response:\n")

for chunk in stream:
    if chunk.choices and len(chunk.choices) > 0:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start_time
            print(delta.content, end="", flush=True)
            full_response += delta.content
            completion_tokens_count += 1
    if hasattr(chunk, "usage") and chunk.usage:
        usage_info = chunk.usage

total_time = time.perf_counter() - start_time

# Validate output schema
print("\n\n" + "=" * 40)
try:
    validated_output = QuestionExtractionResponse.model_validate_json(full_response)
    print("Schema Validation: SUCCESS")
    print(f"Validated Questions Count: {len(validated_output.questions)}")
except Exception as e:
    print(f"Schema Validation: FAILED ({e})")

input_tokens = usage_info.prompt_tokens if usage_info else 0
output_tokens = usage_info.completion_tokens if usage_info else completion_tokens_count
total_tokens = usage_info.total_tokens if usage_info else (input_tokens + output_tokens)

ttft = ttft if ttft is not None else total_time
tps = output_tokens / total_time if total_time > 0 else 0

print("-" * 40)
print(f"Input Tokens: {input_tokens}")
print(f"Output Tokens: {output_tokens}")
print(f"Total Tokens: {total_tokens}")
print(f"Time to First Token (TTFT): {ttft:.4f}s")
print(f"Total Time: {total_time:.4f}s")
print(f"Tokens Per Second (TPS): {tps:.2f} tokens/s")
print("=" * 40)
