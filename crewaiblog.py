#install the packages
!pip install crewai langchain-google-genai crewai_tools

#import libraries and gemini_api_key and only Tool for searching the web for grounding
#in Colab, Gemini_api_key may be as GOOGLE_API_KEY
import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",tools="google_search_retrieval",verbose=True, 
temperature=0.7,google_api_key="api_key")

# Warning control
import warnings
warnings.filterwarnings('ignore')

#install the required libraries of crewai for multiagents deployment
from crewai import Agent, Task, Crew

# no need to attach tools of web searching as we have already the tools="google_search_retrieval"
# create a researcher agent
researcher_agent = Agent(
        role="Expert Researcher",
        goal="Uncover the current and future technologies in {topic}",
        verbose=True,
        memory=True,
        backstory=( "As being expert, you're at the forefront of"
        "innovation, eager to explore and share knowledge that could help"
        "clarify with details."
        ),
        llm=llm,
)

# creating a write agent responsible in writing blog
writer_agent = Agent(
    role="Writer",
    goal="Narrate in formal mode the compelling tech stories about {topic}",
    verbose=True,
    memory=True,
    backstory=(
         "With a flair for simplifying complex topics, you craft"
         "engaging narratives that captivate and educate, bringing new"
         "discoveries to light in an accessible manner."
    ),
    llm=llm,
)

# creating editor for final step
editor_agent = Agent(
    role="Content Editor",
    goal="Revise the post to guarantee factual accuracy while maintaining a formal style.",
    verbose=True,
    memory=True,
    backstory=( 
         "Your task is to review and edit {topic} written by the writer." 
         "You will ensure that the final version, which should be accurate and well-written."
    ),
    llm=llm,
 )

# researcher task
research_task = Task(
    description=(
        "Identify the important big trends in {topic}."
        "Focus on identifying pros and cons and the overall narrative."
        "Your final report should clearly articulate the key points,"
        "its opportunities, and potential risks."
    ),
    expected_output="A comprehensive 4 paragraphs long report on the latest AI trends.",
    agent=researcher_agent
)

# writing task with language model configuration
write_task = Task(
    description=(
        "Compose an insightful article on {topic}."
        "Focus on the latest trends and how it's impacting the industry."
        "This article should be easy to understand, engaging, and positive in formal mode."
    ),
    expected_output="A 4 paragraph article on {topic} advancements formatted as markdown.",
    agent=writer_agent
)

# editing task final step for writing the blog
editor_task = Task(
    description=(
        "Review the blog of {topic} in question for grammatical errors."
    ),
    agent=editor_agent,
    expected_output="A great blog ready for publication. The text is formatted in paragraphs.",
    async_execution=False,
    output_file='news.md'
)

#creating the crew with agents and tasks
crew = Crew(
    agents=[researcher_agent, writer_agent,editor_agent],
    tasks=[research_task, write_task,editor_task],
    verbose=True
)

#crew output and results as blog with markdown
inputs_array = {'topic': 'Your Topic'}
crew_output = crew.kickoff(inputs=inputs_array)

from IPython.display import Markdown
Markdown(crew_output)
