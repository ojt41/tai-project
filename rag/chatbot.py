import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_NEW_TOKENS = 512


def load_model(model_name = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the model and tokenizer, and run one warm-up generation."""

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # The first generate call pays one-off setup costs (CUDA kernels, memory
    # allocation), so run a short one now to keep them out of measured latency.
    generate_response(model, tokenizer, [{"role": "user", "content": "Hi"}], max_new_tokens=8)

    return model, tokenizer


def generate_response(model, tokenizer, messages, max_new_tokens=MAX_NEW_TOKENS):
    """Generate a reply to a list of chat messages.

    Returns a dict with the answer text, input tokens (everything the model
    reads: system prompt, history, question), output tokens (everything it
    generates), generation latency in seconds, and whether the answer hit
    max_new_tokens.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_tokens = model_inputs.input_ids.shape[1]

    # GPU work runs asynchronously, so synchronize before reading the clock,
    # otherwise the timer can stop before the GPU has finished.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - start

    # generate returns prompt + answer; keep only the new tokens.
    new_ids = generated_ids[0][input_tokens:]
    output_tokens = len(new_ids)
    response = tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "answer": response,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_s": latency,
        "truncated": output_tokens >= max_new_tokens,
    }


def main(model, tokenizer, system_prompt="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."):
    """Run a simple command-line chatbot using the specified model and tokenizer."""

    messages = [
    {"role": "system", "content": system_prompt}
    ]

    while True:
        prompt = input()
        if prompt.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": prompt})

        result = generate_response(model, tokenizer, messages)
        print(result["answer"])
        truncated = " | TRUNCATED at max_new_tokens" if result["truncated"] else ""
        print(f"[input {result['input_tokens']} tokens (system + history + question)"
              f" | output {result['output_tokens']} tokens"
              f" | {result['latency_s']:.2f} s{truncated}]")

        messages.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct")
    main(model, tokenizer)
