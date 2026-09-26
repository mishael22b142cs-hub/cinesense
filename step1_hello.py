from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)

response = llm.invoke("Suggest one good Malayalam thriller movie in one line.")
print(response.content)