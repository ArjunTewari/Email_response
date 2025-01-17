import os
from dotenv import find_dotenv, load_dotenv
_= load_dotenv(find_dotenv())
openai_api_key = os.environ["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

from langchain_core.prompts import ChatPromptTemplate
tagging_prompt = ChatPromptTemplate.from_template(
    """
    Extract the desired information from the following passage.  
    Only extract the properties mentioned in the 'Classification' class.
    
    Passage:
    {input}
    """
)

from langchain_core.pydantic_v1 import BaseModel,Field
class Classification(BaseModel):
    sentiment: str = Field(default=None, description="Sentiment of the overall passage.", enum=["happy", "neutral", "sad"])
    political_tendency: str = Field(default=None, description="Political tendency of the speaker.",enum=["conservative", "liberal", "independent"])


chain = tagging_prompt | llm.with_structured_output(Classification)

text = "The government's recent decision to increase funding for renewable energy projects has been widely praised as a step in the right direction toward combating climate change. Environmental activists and progressive political leaders have lauded the move, citing its potential to create green jobs and reduce dependency on fossil fuels. However, some conservative commentators have criticized the plan, arguing that the additional spending may strain the national budget and hurt taxpayers. Despite the divide, the initiative has garnered significant attention and sparked debates on the country's energy priorities"
print(chain.invoke({"input": text}))
