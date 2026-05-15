from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("api_key"),
    base_url="https://api.groq.com/openai/v1"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TopicRequest(BaseModel):
    topic: str


def research_agent(topic):
    prompt = f"""
    You are a research analyst.

    Find:
    - trending discussions
    - common pain points
    - viral content angles
    - startup/growth insights

    Topic: {topic}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a growth research expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def linkedin_agent(topic, research):
    prompt = f"""
    Create:
    - 3 viral LinkedIn post ideas
    - strong hooks
    - engagement CTAs

    Topic: {topic}

    Research:
    {research}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a viral LinkedIn content expert."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


def twitter_agent(topic, research):
    prompt = f"""
    Create:
    - 2 Twitter/X thread ideas
    - attention-grabbing hooks
    - concise growth-focused tweets

    Topic: {topic}

    Research:
    {research}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a Twitter growth strategist."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

@app.post("/generate")
def generate_content(data: TopicRequest):
    topic = data.topic

    research = research_agent(topic)
    linkedin_posts = linkedin_agent(topic, research)
    twitter_threads = twitter_agent(topic, research)

    return {
        "topic": topic,
        "research": research,
        "linkedin_posts": linkedin_posts,
        "twitter_threads": twitter_threads
    }

