import nest_asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from dotenv import load_dotenv
from costinfo import calculate_cost, print_cost_info
from typing import Dict
from pydantic import BaseModel, Field
from mails import complex_mail
from datetime import datetime
from pydantic_ai import RunContext, ModelRetry
from markdown import to_markdown
# Load environment variables from .env file
load_dotenv()

nest_asyncio.apply()

# Get API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

model_name = "gpt-4.1-mini"
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

class CoWorkerDetails(BaseModel):
    name: str = Field(description="The name of the co-worker")
    email: str = Field(description="The email of the co-worker")
    phone: str = Field(description="The phone number of the co-worker")

class Email(BaseModel):
    to: str = Field(description="The email address of the recipient")
    title: str = Field(description="The title of the email")
    body: str = Field(description="The body of the email")

class Emails(BaseModel):
    emails: list[Email] = Field(description="Contains multiple prepared emails. Each email has a 'to', 'title', and 'body' section.")

basic_db: Dict[str, object] = {
    "john": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1234567890",
        "details": "John is a food lover. He never misses a chance to eat."
    },
    "jane": {
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "phone": "+1234567890",
        "details": "Jane is a hardworking person. She doesnt think other than work."
    }
}


basic_agent = Agent(
    model=model,
    system_prompt=f"""
        You are an helpful professional assistant.
        Analyze the mail and extract the information about the event.
    """,
    output_type=CalendarEvent
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

event_details = response_complex.output.model_dump_json(indent=2)

email_agent = Agent(
    model=model,
    output_type=Emails,
    retries=3
)

@email_agent.tool_plain()
def get_user_details(name: str):
    """
        Get the details of the user from the basic_db.
        Args:
            name: The name of the user
        Returns:
            The details of the user
    """
    user = basic_db.get(name)

    #Sadece ModelRetry gösterimi için :)
    if user is None:
        raise ModelRetry(
            """
                User not found. Please try again.
                User name should be lowercase.
                Self correct this and try again.
            """
        )
    return user

@email_agent.system_prompt
async def add_co_worker_details(ctx: RunContext[tuple[CalendarEvent]]) -> str:
    # Dependencies tuple'dan gelir
    calendar_event = ctx.deps
    
    # Her iki dependency'i de markdown'e çevir
    event_markdown = to_markdown(calendar_event)
    
    # İstenen formatta birleştir
    return f""" 
                Event Details: {event_markdown}
            """

# Run the email agent to generate the invitation emails
emails_response = email_agent.run_sync(
    user_prompt=(
        "I am Mikail. Generate invitation emails for John and jane from me."
        "Both of them are my co-workers and my friends."
        "Use friendly tone."
        "Pay attention to the specific details about the person and specifically mention the event characteristics that fit the person."
    ),
    deps=(event_details)
    )

print(emails_response.all_messages())