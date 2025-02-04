from config import settings
from openai import OpenAI
import tiktoken
import argparse
import json
import time

oai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
DEFAULT_MODEL = "gpt-4o-mini"
SEPARATOR = "\n\n"
TOKEN_BUFFER_FACTOR = 4

model_token_limits = {
    "gpt-4o-mini": {
        "tokens_per_minute": 200000,
        "output": 16384,
        "context_length": 128000
    },
}

def count_tokens(text, model=DEFAULT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

SEPARATOR_TOKENS = count_tokens(SEPARATOR)

def truncate_tokens_from_start(
        text: str, 
        max_tokens_per_loop: int, 
        model=DEFAULT_MODEL) -> tuple[str, str]:
    """
    Truncates the first N tokens from a string and returns both parts.
    
    Args:
        text: The input text to truncate
        max_tokens_per_loop: Number of tokens to truncate from start
        model: The model name for tokenization
        
    Returns:
        Tuple of (truncated_part, remaining_part)
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    
    if max_tokens_per_loop >= len(tokens):
        return text, ""
        
    truncated_tokens = tokens[:max_tokens_per_loop]
    remaining_tokens = tokens[max_tokens_per_loop:]
    
    truncated_text = encoding.decode(truncated_tokens)
    remaining_text = encoding.decode(remaining_tokens)
    
    return truncated_text, remaining_text


def reduce_context(context, prompt) -> list[str]:
    context_tokens = count_tokens(context)
    prompt_tokens = count_tokens(prompt)
    
    print(f"Context tokens: {context_tokens}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Total tokens: {context_tokens + prompt_tokens}")

    context_length_limit = model_token_limits[DEFAULT_MODEL]["context_length"]
    rate_limit = model_token_limits[DEFAULT_MODEL]["tokens_per_minute"]
    output_tokens = model_token_limits[DEFAULT_MODEL]["output"]
    
    max_tokens_per_loop = min(
        rate_limit - prompt_tokens - SEPARATOR_TOKENS - output_tokens,
        context_length_limit - prompt_tokens - SEPARATOR_TOKENS - output_tokens
    )

    # Cut max tokens per loop in half, if you get too close to max perf suffers
    max_tokens_per_loop = max_tokens_per_loop // TOKEN_BUFFER_FACTOR

    filtered_responses, remaining_context = [], context
    tokens_used_in_window = 0
    window_start_time = time.time()
    
    while remaining_context:
        # Check if we need to wait for the rate limit window to reset
        current_time = time.time()
        elapsed_time = current_time - window_start_time
        
        if elapsed_time >= 60:
            # Reset window if more than 60 seconds have passed
            tokens_used_in_window = 0
            window_start_time = current_time
        elif tokens_used_in_window >= rate_limit:
            # Wait for the remainder of the current minute
            sleep_time = 60 - elapsed_time
            print(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            tokens_used_in_window = 0
            window_start_time = current_time

        truncated_part, remaining_context = truncate_tokens_from_start(
            remaining_context,
            max_tokens_per_loop,
            DEFAULT_MODEL
        )
        
        # Calculate input tokens before making the API call
        input_tokens = count_tokens(prompt + SEPARATOR + truncated_part)
        
        # Check if this chunk would exceed the rate limit
        if input_tokens + tokens_used_in_window >= rate_limit:
            sleep_time = 60 - (current_time - window_start_time)
            print(f"Would exceed rate limit. Waiting {sleep_time:.2f} seconds...")
            time.sleep(max(0, sleep_time))
            tokens_used_in_window = 0
            window_start_time = time.time()
        
        response = oai_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": prompt + SEPARATOR + truncated_part}]
        )
        filtered_response = response.choices[0].message.content
        print(f"Response: {response}")
        
        # Update token usage with both input and output tokens
        output_tokens = count_tokens(filtered_response)
        tokens_used_in_window += (input_tokens + output_tokens)
        
        print(f"Used {tokens_used_in_window} tokens in current time window")
        filtered_responses.append(filtered_response)

    return filtered_responses


def load_file_content(file_path: str) -> str:
    """Load content from either txt or json file and return as string"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

def main():
    parser = argparse.ArgumentParser(description='Process context and prompt for token reduction')
    parser.add_argument('--context-file', required=True,
                      help='Path to context file (.txt or .json)')
    parser.add_argument('--prompt', required=True,
                      help='The prompt text or path to prompt file')
    parser.add_argument('--is-prompt-file', action='store_true', default=False,
                      help='Flag to indicate if prompt is a file path')
    
    
    args = parser.parse_args()
    
    # Load context file
    context = load_file_content(args.context_file)
    
    # Handle prompt
    if args.is_prompt_file:
        prompt = load_file_content(args.prompt)
    else:
        prompt = args.prompt
    
    # Reduce prompt
    reduced_context = reduce_context(context, prompt)
    print(f"\nReduced context: {reduced_context}")

    output_file = f"{args.context_file.split('.')[0]}_reduced.json"
    with open(output_file, "w") as f:
        json.dump(reduced_context, f, indent=2)  

if __name__ == "__main__":
    main()
