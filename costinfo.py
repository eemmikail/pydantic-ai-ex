
# Function to calculate cost based on token usage
def calculate_cost(model_name, usage_data):
    # Model pricing per 1M tokens (in USD) - update with current pricing
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4.1": {"input": 2, "output": 8},
        "gpt-4.1-mini": {"input": 0.4, "output": 1.6},
        "gpt-4.1-nano": {"input": 0.1, "output": 0.4},
        "o4-mini": {"input": 1.1, "output": 4.4},
        # Add more models as needed
    }
    
    if model_name not in pricing:
        return "Unknown model for pricing calculation"
    
    # Calculate cost based on Usage object attributes
    # For OpenAI models, request_tokens = input/prompt tokens
    # and response_tokens = output/completion tokens
    input_tokens = usage_data.request_tokens
    output_tokens = usage_data.response_tokens
    
    input_cost = (input_tokens / 1_000_000) * pricing[model_name]["input"]
    output_cost = (output_tokens / 1_000_000) * pricing[model_name]["output"]
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost
    }

def print_cost_info(cost_info):
    """Print cost information in a readable format with appropriate units"""
    print("\n----- COST INFORMATION -----")
    print(f"Input tokens: {cost_info['input_tokens']}")
    print(f"Output tokens: {cost_info['output_tokens']}")
    print(f"Total tokens: {cost_info['input_tokens'] + cost_info['output_tokens']}")
    
    # Convert to cents for more readable numbers
    input_cost_cents = cost_info['input_cost_usd'] * 100
    output_cost_cents = cost_info['output_cost_usd'] * 100
    total_cost_cents = cost_info['total_cost_usd'] * 100
    
    print(f"\nInput cost: {input_cost_cents:.6f} cents")
    print(f"Output cost: {output_cost_cents:.6f} cents")
    print(f"Total cost: {total_cost_cents:.6f} cents")
    
    # Also show in USD with more decimal places
    print(f"\nTotal cost (USD): ${cost_info['total_cost_usd']:.8f}")
    print("---------------------------")