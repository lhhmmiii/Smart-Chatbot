import os
import io
import warnings
import chainlit as cl
from dotenv import load_dotenv
from PIL import Image
from stability_sdk import client
from langchain import hub
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from langchain.tools import StructuredTool, Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from document_summarize import *


load_dotenv()
os.environ["STABILITY_HOST"] = "grpc.stability.ai:443"
stability_api_key= os.getenv("STABILITY_KEY")
api_key = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model = 'gemini-1.5-flash', max_retries= 2, timeout= None, max_tokens = None, api_key = api_key)

stability_api = client.StabilityInference( 
    key = stability_api_key,
    verbose=True,
    engine="stable-diffusion-xl-1024-v1-0", # Set the engine to use for generation. Check out the following link for a list of available engines: https://platform.stability.ai/docs/features/api-parameters#engine
)

def create_image(prompt, init_image = None): # Nếu init_image = None thì là generate, còn không thì là edit
    start_schedule = 0.8 if init_image else 1
    # Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt = prompt,
        init_image=init_image, # init image for edit
        start_schedule=start_schedule,
        seed=4253978046, # If a seed is provided, the resulting generated image will be deterministic.
                        # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                        # Note: This isn't quite the case for Clip Guided generations, which we'll tackle in a future example notebook.
        steps=50, # Amount of inference steps performed on image generation. Defaults to 30. 
        cfg_scale=8.0, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
        width=1024, # Generation width, defaults to 512 if not included.
        height=1024, # Generation height, defaults to 512 if not included.
        samples=1, # Number of images to generate, defaults to 1 if not included.
        sampler=generation.SAMPLER_K_DPMPP_2M # Choose which sampler we want to denoise our generation with.
                                                    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
                                                    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    for resp in answers:
        for artifact in resp.artifacts:
            if artifact.finish_reason == generation.FILTER:
                warnings.warn(
                    "Your request activated the API's safety filters and could not be processed."
                    "Please modify the prompt and try again.")
            if artifact.type == generation.ARTIFACT_IMAGE:
                img = Image.open(io.BytesIO(artifact.binary))
                name = str(artifact.seed)+ ".png"
                image_save_path = f"Image/Output/{name}"
                img.save(image_save_path) # Save our generated images with their seed number as the filename.
                cl.user_session.set(name, artifact.binary)
    return name

def generate_image(prompt):
    print("00000000000000000000000000000000000")
    img_name = create_image(prompt)
    return img_name

def edit_image(prompt: str):
    print("1111111111111111111111111111111111111111111111111")
    init_image_bytes = cl.user_session.get("4253978046.png")
    if init_image_bytes is None:
        raise ValueError(f"Could not find image init_image_name.")

    init_image = Image.open(io.BytesIO(init_image_bytes))
    image_name = create_image(prompt, init_image)
    return image_name

def create_chain(llm):
    template = """
    Hello! I'm your assistant. How can I assist you today? When you're ready, feel free to upload the document you'd like help with. If you have any other questions before that, don't hesitate to ask!
    You use vietnamese to reply.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system",template),
        ("user", "{input}"),
    ])

    return (
        prompt
        | llm
        | StrOutputParser()
    )


def image_tools():
    generate_image_tool = Tool.from_function(
        func = generate_image,
        name="GenerateImage",
        description="Useful to create an image from a text prompt.",
        return_direct=True,
    )

    edit_image_tool = StructuredTool.from_function(
        func=edit_image,
        name="EditImage",
        description="Useful to edit an image with a prompt. Works well with commands such as 'replace', 'add', 'change', 'remove'.",
        return_direct=True,
    )

    tools = [generate_image_tool, edit_image_tool]

    return tools

def my_tools(llm):

    chain = create_chain(llm)
    qa_tool = chain.as_tool(
        name="qa_tool", description="Your task is that use your knowledge to answer the questions from user. If you don't know answer, don't answer."
    )
    search = TavilySearchResults(search = 2)

    generate_image_tool = Tool.from_function(
        func = generate_image,
        name="GenerateImage",
        description="Useful to create an image from a text prompt.",
        return_direct=True,
    )

    edit_image_tool = StructuredTool.from_function(
        func=edit_image,
        name="EditImage",
        description="Useful to edit an image with a prompt. Works well with commands such as 'replace', 'add', 'change', 'remove'.",
        return_direct=True,
    )

    tools = [generate_image_tool, edit_image_tool, qa_tool, search, summarize_document]

    return tools

# tools = my_tools(llm)
# prompt = hub.pull("hwchase17/openai-functions-agent")
# agent = create_tool_calling_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)
# res = agent_executor.invoke({"input": "create image include black cat and dog at the beach"})
# print(res)

