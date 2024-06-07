import os
import warnings
from crewai import Agent, Task, Crew
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool
from IPython.display import Markdown, display
import openai

# save your keys to user/system environment varaibles on your local machine
def get_serper_api_key():
    """Retrieve the Serper API key from an environment variable."""
    api_key = os.getenv('SERPER_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key


def get_openai_api_key():
    """Retrieve the OpenAI API key from an environment variable."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key

# recommend using gpt-4-turbo model as the results are more human-like, softer, and in the tone of the job posting.
# Hallucinations/accuracy was a mixed bag with 4.0.  In an attempt to be more accurate, the agents completely removed
# 4 years of my experience as a senior project manager :O.  
# I will attempt to fix this with prompting or maybe...
#TO DO: add a QA agent to the team
# Be aware that you PAY FOR this improvement.  $.36 for one run compared to $.80 for 20 runs.
def setup_environment(model_version = 'gpt-3.5-turbo'):
    os.environ["OPENAI_MODEL_NAME"] = model_version
    os.environ["PYPPETEER_CHROMIUM_REVISION"] = '1263111'
    #os.environ["SERPER_API_KEY"] = get_serper_api_key()


def main():
    try:
        # Set up the environment
        setup_environment()

        # Retrieve and set the API key
        openai_api_key = get_openai_api_key()



        from crewai_tools import (
        FileReadTool,
        ScrapeWebsiteTool,
        MDXSearchTool,
        SerperDevTool
        )

        resume_name = './jon_verbose_resume.md'
        search_tool = SerperDevTool()
        scrape_tool = ScrapeWebsiteTool()
        read_resume = FileReadTool(file_path=resume_name)
        semantic_search_resume = MDXSearchTool(mdx=resume_name)

        # Create the agents
        # Agent 1: Researcher
        researcher = Agent(
            role="Tech Job Researcher",
            goal="Make sure to do amazing analysis on "
                "job posting to help job applicants",
            tools = [scrape_tool, search_tool],
            verbose=True,
            backstory=(
                "As a Job Researcher, your prowess in "
                "navigating and extracting critical "
                "information from job postings is unmatched."
                "Your skills help pinpoint the necessary "
                "qualifications and skills sought "
                "by employers, forming the foundation for "
                "effective application tailoring."
            )
        )

        # Agent 2: Profiler
        profiler = Agent(
            role="Personal Profiler for Engineers",
            goal="Do increditble research on job applicants "
                "to help them stand out in the job market",
            tools = [scrape_tool, search_tool,
                    read_resume, semantic_search_resume],
            verbose=True,
            backstory=(
                "Equipped with analytical prowess, you dissect "
                "and synthesize information "
                "from diverse sources to craft comprehensive "
                "personal and professional profiles, laying the "
                "groundwork for personalized resume enhancements."
            )
        )

        # Agent 3: Reviewer
        reviewer = Agent(
            role="Resume Reviewer",
            goal="Review a given resume and notify the Resume Strategist of any time gaps between positions on the resume. ",
            backstory="You are an expert at reviewing resumes "
                    "for the Resume Strategist. "
                    "Your goal is to review the resume "
                    "to ensure that it follows resume writing best practices.",
            tools = [read_resume, semantic_search_resume,search_tool],
            verbose=True,
            max_iter=5
        )

        # Agent 4: Resume Strategist
        resume_strategist = Agent(
            role="Resume Strategist for Engineers",
            goal="Following resume writing best practices, highlight the most "
                "relevant experiences and skills in each work experience "
                "to make a resume stand out in the job market.",

            verbose=True,
            backstory=(
                "With a strategic mind and an eye for detail, you "
                "excel at refining resumes to highlight the most "
                "relevant experiences and skills, ensuring they "
                "resonate perfectly with the job's requirements."
            )
        )

        # Creating Tasks
        # Task for Researcher Agent: Extract Job Requirements
        research_task = Task(
            description=(
                "Analyze the job posting URL provided ({job_posting_url}) "
                "to extract key experiences, skills, and qualifications "
                "required. Use the tools to gather content and identify "
                "and categorize the requirements."
            ),
            expected_output=(
                "A structured list of job requirements, including necessary "
                "experiences, skills, and qualifications."
            ),
            agent=researcher,
            async_execution=True
        )

        # Task for Profiler Agent: Compile Comprehensive Profile
        profile_task = Task(
            description=(
                "Compile a detailed personal and professional profile "
                "using the GitHub ({github_url}) URLs, and personal write-up "
                "({personal_writeup}). Utilize tools to extract and "
                "synthesize information from these sources."
            ),
            expected_output=(
                "A comprehensive profile document that includes skills, "
                "project experiences, contributions, interests, and "
                "communication style."
            ),
            agent=profiler,
            async_execution=True
        )

        edit_task = Task(
            description=("Utilize tools to identify the chronoligcal job history for the given resume and "
                        "make sure there are no gaps. You will check for grammatical errors as well."),
            expected_output="The correct chronological order of the job history and a list of of grammatical errors that should be corrected.",   
            agent=reviewer,
            async_execution=True
        )

        # Task for Resume Strategist Agent: Align Resume with Job Requirements
        resume_strategy_task = Task(
            description=(
                "Using the profile, job requirements, job history, and grammatical errors obtained from "
                "previous tasks, tailor the resume to highlight the most "
                "relevant areas. Employ tools to adjust and enhance the "
                "resume content. Make sure this is the best resume ever but "
                "don't make up any information. You must review and update every section, "
                "inlcuding the initial summary, work experience, skills, "
                "and education. Check the final resume to make sure that the experiences were not "
                "copied verbatim from the original resume.  All to better reflect how the candidates "
                "experiences and abilities match the job posting."
            ),
            expected_output=(
                "An updated resume that effectively highlights and summarizes the candidate's "
                "experiences and qualifications relevant to the job."
            ),
            output_file="tailored_resume.md",
            context=[research_task, profile_task, edit_task],
            agent=resume_strategist
        )


        job_application_crew = Crew(
            agents=[researcher
                    ,profiler
                    ,reviewer
                    ,resume_strategist],

            tasks=[research_task
                ,profile_task
                ,edit_task
                    ,resume_strategy_task],
            verbose=True
        )

        # Create the Crew
        job_application_crew = Crew(
            agents=[researcher
                    ,profiler
                    ,reviewer
                    ,resume_strategist],

            tasks=[research_task
                ,profile_task
                ,edit_task
                    ,resume_strategy_task],
            verbose=True
        )

        job_application_inputs = {
            'job_posting_url': job_posting_url,
            'github_url': github_url,
            'personal_writeup': """Jon is an accomplished Software
            Engineering Leader with 24 years of experience, specializing in
            managing remote and in-office teams, and expert in multiple
            programming languages and frameworks. He holds a MPA and a strong
            background in AI and data science. Jon has successfully led
            major tech initiatives and startups, proving his ability to drive
            innovation and growth in the tech industry. Ideal for leadership
            roles that require a strategic and innovative approach."""
        }

        ### this execution will take a few minutes to run
        result = job_application_crew.kickoff(inputs=job_application_inputs)


    except Exception as e:
        # This block catches any other exceptions that were not specifically handled above
        print(f"An unexpected error occurred: {e}")

job_posting_url = 'https://docs.google.com/document/d/e/2PACX-1vT_hJQYGSJ9lj0-aO2zJeZx4d16JdvK_HMkZihS2FH-6hCv2Z6C7w0UH4Eb3I_wFGfhuFgNFvmi9tqd/pub'
github_url = 'https://github.com/ecocarlisle/'

if __name__ == "__main__":
    main()