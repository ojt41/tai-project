from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Load the model and tokenizer from the specified model name."""
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="float16",
        device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # print("Model dtype:", model.dtype)
    # print("Model device:", model.device)
    # print("Model generation config:", model.generation_config)

    return model, tokenizer

def generate_response(model, tokenizer, messages):
    """Generate a response from the model given the input tokens."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.0
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_token_length = len(generated_ids[0])
    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return response, response_token_length


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

        response, response_token_length = generate_response(model, tokenizer, messages)
        print(response)
        print(f"Answered in {response_token_length} tokens.")

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    model, tokenizer = load_model("Qwen/Qwen2.5-0.5B-Instruct")
    main(model, tokenizer)