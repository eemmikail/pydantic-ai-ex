import nest_asyncio
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel,Field
import os
from dotenv import load_dotenv
from costinfo import calculate_cost, print_cost_info
from datetime import datetime
from markdown import to_markdown
from mails import complex_mail
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

class CoWorkerDetails(BaseModel):
    name: str = Field(description="The name of the co-worker")
    email: str = Field(description="The email of the co-worker")
    phone: str = Field(description="The phone number of the co-worker")

class Email(BaseModel):
    subject: str = Field(description="The subject of the email")
    body: str = Field(description="The body of the email")

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
co_worker_details = CoWorkerDetails(name="Mikail Karadeniz", email="mikail.karadeniz@gmail.com", phone="+1234567890")

basic_agent_2 = Agent(
    model=model,
    system_prompt=f"""
        You are an helpful professional assistant.
        Write friendly and professional email to the co-worker.
    """,
    output_type=Email
)

@basic_agent_2.system_prompt
async def add_co_worker_details(ctx: RunContext[tuple[CalendarEvent, CoWorkerDetails]]) -> str:
    # Dependencies tuple'dan gelir
    calendar_event, co_worker_details = ctx.deps
    
    # Her iki dependency'i de markdown'e çevir
    event_markdown = to_markdown(calendar_event)
    coworker_markdown = to_markdown(co_worker_details)
    
    # İstenen formatta birleştir
    return f""" 
                Event Details: {event_markdown}
                Co-worker details: {coworker_markdown}
            """

response_email = basic_agent_2.run_sync(
        user_prompt="I am Semra. I need help about writing an email to my co-worker.", deps=(event_details, co_worker_details)
    )

print(response_email.all_messages())
