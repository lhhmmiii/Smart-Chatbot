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
# Set up our initial generation parameters.
    answers = stability_api.generate(
        prompt = prompt,
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
    return name

def generate_image(prompt):
    img_name = create_image(prompt)
    return img_name

def edit_image(init_image_name: str, prompt: str):
    init_image_bytes = cl.user_session.get(init_image_name)
    if init_image_bytes is None:
        raise ValueError(f"Could not find image `{init_image_name}`.")

    init_image = Image.open(io.BytesIO(init_image_bytes))
    image_name = create_image(prompt, init_image)

    return f"Here is {image_name} based on {init_image_name}."

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

tools = my_tools(llm)
prompt = hub.pull("hwchase17/openai-functions-agent")
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
text = '''
Suốt bao năm, để dòng tranh này không bị rơi vào quên lãng, mỗi ngày người ta đều thấy ông Đạt cặm cụi làm nên những bức tranh từ mũi dao, cán đục. Ông bảo, tranh sơn khắc ở nước ta 
ra đời sớm nhất và còn đẹp hơn cả tranh sơn khắc của Nhật. Quý giá như vậy nên ông chẳng thể để nghề mai một trong sự chông chênh của thời cuộc. Một trong những sản phẩm sơn khắc của
ông Đạt được trả 25 triệu. Theo ông Đạt, thời điểm năm 1945 đến 1995 là lúc tranh sơn khắc ở nước ta phát triển mạnh nhất. Thời điểm đó, các sản phẩm của Hạ Thái chiếm tới 70% hàng 
xuất khẩu, giải quyết được công ăn việc làm cho người dân trong làng và cả các địa phương khác, đem lại cuộc sống khấm khá cho nhiều hộ gia đình. Say mê hội họa từ nhỏ, nên chuyện 
ông Đạt đến với tranh sơn khắc như một mối duyên tiền định. Khi mới tiếp xúc với những bức tranh này, ông Đạt như bị lôi cuốn chẳng thể nào dứt ra được. Học hết cấp 3, tôi thi vào 
Đại học sư phạm nhưng sức khỏe không đảm bảo nên xin vào làm thợ vẽ trong xưởng của hợp tác xã. Năm 1979, tôi được hợp tác xã cử đi học thêm ở trường Mỹ Nghệ. Khi về lại xưởng, nhờ 
năng khiếu hội họa nên tôi được chuyển sang khâu đoạn khảm trai rồi sang tranh khắc. Tôi làm tranh khắc từ đó đến giờ ông Đạt chia sẻ. Theo lời ông Đạt, học sơn khắc khó bởi cách 
vẽ của dòng tranh này khác hẳn với sơn mài. Nếu như sơn mài người ta có thể vẽ bằng chổi hay bút lông, cũng có khi là chất liệu mềm rồi mới quét sơn lên vóc thì sơn khắc khâu đoạn 
lại làm khác hẳn. Sơn khắc là nghệ thuật của đồ họa, sự hoàn thiện của bức tranh phụ thuộc vào những nét chạm khắc và những mảng hình tinh tế, giàu cảm xúc. Cuối cùng mới là việc 
tô màu nhằm tạo sự khắc họa mạnh. Như một lẽ xoay vần tự nhiên, sự phát triển của làng nghề Hạ Thái dần chùng xuống. Làng nghề bước vào thời kỳ suy thoái, đặc biệt là trong giai 
đoạn khủng hoảng kinh tế Đông Âu từ 1984 đến 1990 đã làm hợp tác xã tan rã. Ông Đạt khi đó cũng như bao người thợ khác đều phải quay về làm ruộng. Ông Đạt giải thích, tranh sơn khắc
xuất phát từ gốc tranh sơn mài. Nếu như ở tranh sơn mài thông thường, để có một tấm vóc vẽ người ta phủ sơn ta, vải lên tấm gỗ và mài phẳng thì tranh sơn khắc độc đáo ở chỗ, phải 
sử dụng kỹ thuật thủ công để khắc lên tấm vóc sơn mài. Tranh sơn khắc từ phôi thai, phác thảo đến lúc hoàn thành có khi kéo dài cả năm trời. Chẳng hạn, riêng công khắc ở bức tranh
khổ nhỏ thường tôi làm cả ngày lẫn đêm thì mất 2 ngày, phối màu mất 3 ngày. Để người trẻ học được nghề cũng sẽ mất khoảng 6 tháng đến 1 năm - ông Trần Thành Đạt chia sẻ. Tranh 
sơn khắc đòi hỏi rất kỹ về phác thảo, bố cục, cũng như mảng màu sáng tối mà màu đen của vóc là chủ đạo. Dù trên diện tích bức tranh khổ lớn bao nhiêu nó vẫn rất cần kỹ càng và 
chính xác đến từng xen-ti-met. Nếu sai, bức tranh sẽ gần như bị hỏng, các đường nét phải khắc họa lại từ đầu. Kỳ công là vậy nên giá thành mỗi sản phẩm sơn khắc thường khá cao, 
trung bình từ 4 đến 25 triệu đồng/bức tranh. Giá thành cao lại yêu cầu khắt khe về mặt kỹ thuật, mỹ thuật nên theo Nghệ nhân Trần Thành Đạt, nhiều người trong làng đã từ bỏ, 
không làm dòng tranh này nữa. Tranh sơn khắc làm mất nhiều thời gian và công sức nhưng khó bán. Họ đều tập trung làm tranh sơn mài, với chất liệu ngoại nhập cho rẻ và ít tốn 
công sức. Hầu như cả làng đã quay lưng, bỏ rơi dòng tranh sơn khắc vào lãng quên ông Đạt buồn bã kể. Được biết, hiện xưởng sản xuất tranh của ông Đạt chủ yếu là các thành viên 
trong gia đình. Ông khoe, hai con trai và con gái đều tốt nghiệp Trường Đại học Mĩ thuật, con rể và các con dâu cũng là họa sĩ của trường. Tất cả các thành viên trong gia đình 
ông đều chung niềm say mê với sơn khắc. Đinh Luyện.
'''
res = agent_executor.invoke({"input": f"{text}\n\nTóm tắt văn bản trên"})
print(res)