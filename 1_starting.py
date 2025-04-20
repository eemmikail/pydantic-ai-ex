import nest_asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from dotenv import load_dotenv
from costinfo import calculate_cost, print_cost_info
# Load environment variables from .env file
load_dotenv()

nest_asyncio.apply()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

model_name = "gpt-4o-mini"
model = OpenAIModel(
    model_name,
    provider=OpenAIProvider(api_key=api_key)
)

basic_agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
)

response = basic_agent.run_sync("What is the capital of the moon?")
print(response.output)

#print_cost_info(calculate_cost(model_name, response.usage()))

response_2 = basic_agent.run_sync(
    user_prompt="What was my last question?",
    message_history=response.new_messages()
    )

print(response_2.output)
print(response_2.all_messages())

#print_cost_info(calculate_cost(model_name, response_2.usage()))