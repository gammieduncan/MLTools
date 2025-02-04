# MLTools

## Initial Setup

In config/ add a .env file with the following:

OPENAI_API_KEY=<your_openai_api_key>

Then create a virtual environment and install the dependencies:

pip install -r requirements.txt

## Prompt Reducer

Prompt reducer is a quick and easy substitute for RAG. If you have a large context file, you can use this tool to reduce the context file to a smaller size to fit the context window of your LLM. Right now, it only supports GPT-4o, but I welcome any contributions to support other models.

It's a CLI tool that takes a context file and a prompt, and iterates over the context file, reducing it until the prompt can be answered.

It outputs a JSON file with the reduced context.

### Possible use case

You have a giant log file, want to scan it for errors, and it won't fit in the context window of your LLM but you don't want to use RAG. You can input whatever prompt you want to query the log file, and it will scan the log file in a loop with the max number of tokens each time. Finally, it returns a JSON file with the list of all errors found. 

### Usage

To run the tool:

python prompt_reducer.py --context-file <path_to_context_file> --prompt <prompt> [--is-prompt-file <true/false>]

prompt can either be a string or a file path. It will be read as a string if it's a file path. If it's a file path, you need to pass the --is-prompt-file flag as true. I did this because I wanted a prompt, even if it's a string input, to be able to end in a file extension as well.

## Notes

The context file is read as a string, so it works well mostly with text files and JSONs. If you want to use other file types, you'll probably need to modify the code.

The default model is gpt-4o-mini, but you can change it to any other model supported by OpenAI. However, I chose 4o-mini because it's usually sufficient for simple filtering tasks, and it has a much higher token-per-minute limit than other models (200K vs 30K for gpt-4o).

## Test

Run: python prompt_reducer.py --context-file war_and_peace.txt --prompt "extract all the sentences that mention Vronsky on a horse. Only return the relevant text as it is written, not any commentary. if there are no instances, return nothing. Here is the text: " 