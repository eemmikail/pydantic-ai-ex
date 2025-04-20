import nest_asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv
from costinfo import calculate_cost, print_cost_info
from datetime import datetime
from mails import complex_mail, mail
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

class CalendarEvent(BaseModel):
    title: str = Field(description="The title of the event")
    start_date: str = Field(description="The start date of the event")
    end_date: str = Field(description="The end date of the event")
    location: str = Field(description="The location of the event")
    description: str = Field(description="The description of the event")

basic_agent = Agent(
    model=model,
    system_prompt=f"""
        You are an helpful professional assistant.
        Analyze the mail and extract the information about the event.
    """,
    output_type=CalendarEvent
)

response = basic_agent.run_sync(
    f"""
        There is an mail from Jane Doe.

        Mail:
        {mail}
    """
)
today = datetime.now().strftime("%Y-%m-%d")
response_complex = basic_agent.run_sync(
    f"""
        There is an mail from Jane Doe.
        Today is {today}.

        Mail:
        {complex_mail}
    """
)

print(response.output.model_dump_json(indent=2))
print_cost_info(calculate_cost(model_name, response.usage()))

print(response_complex.output.model_dump_json(indent=2))
print_cost_info(calculate_cost(model_name, response_complex.usage()))

print(response_complex.all_messages())
