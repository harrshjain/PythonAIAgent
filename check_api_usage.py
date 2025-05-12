import os
import tiktoken
import configparser
import argparse
from datetime import datetime

# Define pricing per million tokens (in USD)
MODEL_PRICING = {
    "gpt-4o": {"input": 5, "output": 20}  # $5/million input, $20/million output
}

# Default and only model to use for estimations
DEFAULT_MODEL = "gpt-4o"

def count_tokens(text, model="gpt-4o"):
    """Count the number of tokens in a text string"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
            
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return len(text) // 4

def calculate_cost(input_tokens, output_tokens, model=DEFAULT_MODEL):
    """Calculate the approximate cost in USD based on token counts"""
    if model not in MODEL_PRICING:
        print(f"Warning: Unknown model '{model}'. Using {DEFAULT_MODEL} pricing.")
        model = DEFAULT_MODEL
        
    pricing = MODEL_PRICING[model]
    input_cost = (input_tokens / 1000000) * pricing["input"]
    output_cost = (output_tokens / 1000000) * pricing["output"]
    return input_cost + output_cost

def analyze_file_tokens(filepath, model=DEFAULT_MODEL):
    """Analyze token count in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        token_count = count_tokens(content, model)
        file_size = os.path.getsize(filepath)
        
        return {
            "filepath": filepath,
            "tokens": token_count,
            "size_bytes": file_size,
            "tokens_per_kb": token_count / (file_size/1024) if file_size > 0 else 0
        }
    except Exception as e:
        print(f"Error analyzing {filepath}: {e}")
        return {
            "filepath": filepath,
            "tokens": 0,
            "size_bytes": 0,
            "tokens_per_kb": 0,
            "error": str(e)
        }

def estimate_project_usage(directory=".", file_types=None, model=DEFAULT_MODEL, verbose=False):
    """Estimate token usage across the project"""
    if file_types is None:
        file_types = [".py", ".txt", ".md", ".json"]
    
    total_tokens = 0
    total_files = 0
    file_results = []
    
    print(f"\n===== Token Usage Estimation ({model}) =====")
    print(f"Analyzing files with extensions: {', '.join(file_types)}")
    
    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip certain directories
            if any(skip_dir in root for skip_dir in [".git", "__pycache__", "venv", "env", "node_modules"]):
                continue
                
            # Process only specified file types
            if any(file.endswith(ext) for ext in file_types):
                filepath = os.path.join(root, file)
                result = analyze_file_tokens(filepath, model)
                
                if "error" not in result:
                    total_tokens += result["tokens"]
                    total_files += 1
                    file_results.append(result)
                    
                    if verbose:
                        print(f"{filepath}: {result['tokens']} tokens")
    
    # Sort results by token count
    file_results.sort(key=lambda x: x["tokens"], reverse=True)
    
    # Display summary
    print(f"\nFound {total_files} relevant files")
    print(f"Total tokens: {total_tokens:,}")
    
    # Estimate costs for different scenarios
    print("\n----- Cost Estimates -----")
    
    # Scenario 1: All tokens as input
    input_cost = calculate_cost(total_tokens, 0, model)
    print(f"If all tokens are input:  ${input_cost:.4f}")
    
    # Scenario 2: All tokens as output
    output_cost = calculate_cost(0, total_tokens, model)
    print(f"If all tokens are output: ${output_cost:.4f}")
    
    # Scenario 3: 50/50 split
    mixed_cost = calculate_cost(total_tokens/2, total_tokens/2, model)
    print(f"If 50/50 input/output:    ${mixed_cost:.4f}")
    
    # List top token-heavy files
    print("\n----- Top Token-Heavy Files -----")
    for i, result in enumerate(file_results[:10]):  # Show top 10
        if i >= 10:
            break
        print(f"{i+1}. {result['filepath']}: {result['tokens']:,} tokens")
    
    return {
        "total_tokens": total_tokens,
        "total_files": total_files,
        "cost_estimates": {
            "all_input": input_cost,
            "all_output": output_cost,
            "mixed": mixed_cost
        },
        "file_results": file_results
    }

def check_config_api_limits():
    """Check if the API key exists and show instructions for checking limits"""
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        api_key = config["OPENAI_API"]["API_KEY"]
        
        if api_key:
            print("\n===== API Key Information =====")
            # Mask most of the API key
            masked_key = api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
            print(f"Found API key in config.ini: {masked_key}")
            print("\nTo check your OpenAI API limits and usage:")
            print("1. Go to https://platform.openai.com/account/usage")
            print("2. Log in with your OpenAI account")
            print("3. View your current usage and limits on that page")
        else:
            print("\nNo API key found in config.ini")
    except Exception as e:
        print(f"\nError checking config.ini: {e}")
        print("No API key information available")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate token usage in your project")
    parser.add_argument('--dir', '-d', default='.', help='Directory to analyze (default: current directory)')
    parser.add_argument('--types', '-t', default='.py,.txt,.md,.json', help='File types to analyze (comma separated)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    file_types = args.types.split(',')
    
    # Only use the default GPT-4o model
    model_to_use = DEFAULT_MODEL
    
    # Run the estimation
    print(estimate_project_usage(args.dir, file_types, model_to_use, args.verbose))
    
    # Check config for API key
    check_config_api_limits()
    
    print("\nNOTE: This is an estimation and actual API usage may vary.")
    print("The exact token count depends on the specific tokenizer used by each model.")