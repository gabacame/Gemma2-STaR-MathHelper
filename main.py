from langchain_community.llms import Ollama
from langchain.agents import create_react_agent
from langchain.prompts import PromptTemplate
from calculator import Calculator
from star_model import star_iteration
from utils import load_math_notes, setup_qa_chain, setup_calculator_tools
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Cargar modelo LLM
llm = Ollama(model="gemma2")

# Configurar la calculadora
calculator = Calculator()
tools = setup_calculator_tools(calculator)

# Definir el prompt necesario para el agente REACT
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant capable of performing various mathematical calculations. 
You have access to the following tools: {tool_names}.
Use these tools to answer the user's mathematical questions.

{tools}
{agent_scratchpad}
""")

agent = create_react_agent(tools=tools, llm=llm, prompt=prompt_template)

# Cargar apuntes de matemáticas
math_notes = load_math_notes("data/math.txt")

# Configurar RAG
qa_chain = setup_qa_chain(llm, math_notes)

# Definir el conjunto de datos inicial y ejemplos
dataset = [
    {'question': 'What is the integral of x^2?', 'answer': '(1/3)*x^3 + C'},
    {'question': 'What is the derivative of x^3?', 'answer': '3*x^2'},
    {'question': 'Solve for x: 2*x + 3 = 7', 'answer': 'x = 2'},
    {'question': 'What is the derivative of sin(x)?', 'answer': 'cos(x)'},
    {'question': 'What is the integral of 1/x?', 'answer': 'ln|x| + C'},
]

examples = [
    {'question': 'What is the derivative of x^2?', 'rationale': 'The derivative of x^2 is 2*x.'},
    {'question': 'What is the integral of x^3?', 'rationale': 'The integral of x^3 is (1/4)*x^4 + C.'},
    {'question': 'Solve for x: x^2 - 4 = 0', 'rationale': 'The solutions for x^2 - 4 = 0 are x = 2 and x = -2.'},
    {'question': 'What is the derivative of cos(x)?', 'rationale': 'The derivative of cos(x) is -sin(x).'},
    {'question': 'What is the integral of e^x?', 'rationale': 'The integral of e^x is e^x + C.'},
]

# Ejecutar iteraciones de STaR
examples = star_iteration(llm, dataset, examples)

def solve_math_problem(question):
    retrieved_info = qa_chain.run(question)
    calculations = agent.run(f"Use the following information to solve the math problem: {retrieved_info}")
    final_answer = llm.invoke(f"Q: {question}\nInfo: {retrieved_info}\nCalculations: {calculations}\nA: ")
    return final_answer

# Ejemplo de uso
question = "What is the integral of x^2?"
answer = solve_math_problem(question)
print(answer)