# Gemma2-STaR-MathHelper

## Description

Gemma2-STaR-MathHelper is a project that leverages the Gemma2 language model to efficiently solve mathematical problems. The project integrates a Retrieval-Augmented Generation (RAG) system and the Self-Teaching with Rationale (STaR) method along with mathematical tools to provide precise solutions and well-structured rationales.

## Features

- **LLM Model:** Uses the `Gemma2-9b` model from Ollama.
- **Information Retrieval System:** Implemented with FAISS and `SentenceTransformer` for embeddings.
- **REACT Agent:** Configured to use mathematical tools for problem-solving.
- **Integrated Calculator:** Tools for evaluating expressions, integration, differentiation, equation solving, and matrix operations.
- **STaR Iterations:** Generation and improvement of examples through STaR iterations.

## Requirements

- Python 3.8 or higher
- Python libraries:
  - `langchain-community`
  - `sentence-transformers`
  - `faiss-cpu`
  - `dotenv`
  - `huggingface_hub`

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/Gemma2-STaR-MathHelper.git
    cd Gemma2-STaR-MathHelper
    ```

2. Create a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Configure the environment variables:

    Create a `.env` file in the root directory of the project and add your OpenAI API key:

    ```env
    OPENAI_API_KEY=your_api_key
    ```

## Usage

1. Ensure the virtual environment is activated:

    ```bash
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Run the main script:

    ```bash
    python main.py
    ```

3. Modify `data/apuntes_matematicas.txt` to add your own mathematical notes.

## Project Structure

- `main.py`: Main script that configures and runs the agent.
- `utils.py`: Utilities for loading notes, setting up the QA chain, and calculator tools.
- `calculator.py`: Implementation of the `Calculator` class with mathematical functions.
- `star_model.py`: Implementation of functions for generating and filtering rationales.
- `data/apuntes_matematicas.txt`: Text file with mathematical notes for the retrieval system.

## Paper Implementation

This project implements the concepts described in the paper "STaR: Self-Teaching with Rationales for Improved Explainability and Performance". Here's a brief overview of the implementation steps:

1. **Retrieval-Augmented Generation (RAG):**
    - Uses FAISS for indexing and retrieving relevant mathematical notes.
    - Utilizes `SentenceTransformer` to generate embeddings for the notes.

2. **Self-Teaching with Rationales (STaR):**
    - Generates initial rationales for a set of example questions and answers.
    - Filters and selects correct rationales to create a refined set of examples.
    - Iteratively improves the examples through multiple iterations of generating and filtering rationales.

3. **Integration with Tools:**
    - Configures an agent with access to various mathematical tools (calculator, integrator, differentiator, etc.) to solve complex mathematical problems.
    - Uses the REACT framework to describe the steps and tools used in solving problems.

## Example Usage

To solve a mathematical problem, you can modify `main.py` to use the `solve_math_problem` function with your desired question:

```python
question = "What is the integral of x^2?"
answer = solve_math_problem(question)
print(answer)
