from langchain_community.llms import Ollama

def generate_rationale(llm, question, examples):
    prompt = "\n".join([f"Q: {ex['question']}\nA: {ex['rationale']}" for ex in examples])
    prompt += f"\nQ: {question}\nA: "
    response = llm.invoke(prompt)
    return {'question': question, 'rationale': response.strip(), 'answer': response.split()[-1]}

def filter_rationales(dataset, generated_rationales):
    correct_rationales = []
    for data, rationale in zip(dataset, generated_rationales):
        if rationale['answer'] == data['answer']:
            correct_rationales.append({
                'question': data['question'],
                'rationale': rationale['rationale'],
                'answer': data['answer']
            })
    return correct_rationales

def rationalize(llm, question, correct_answer, examples):
    prompt = "\n".join([f"Q: {ex['question']}\nA: {ex['rationale']}" for ex in examples])
    prompt += f"\nQ: {question}\nA: {correct_answer}\nRationale: "
    response = llm.invoke(prompt)
    return {'question': question, 'rationale': response.strip(), 'answer': correct_answer}

def star_iteration(llm, dataset, examples, iterations=5):
    for i in range(iterations):
        generated_rationales = []
        for data in dataset:
            question = data['question']
            correct_answer = data['answer']
            rationale = generate_rationale(llm, question, examples)
            if rationale['answer'] == correct_answer:
                generated_rationales.append({'question': question, 'rationale': rationale['rationale'], 'answer': correct_answer, 'correct': True})
            else:
                rationalized_rationale = rationalize(llm, question, correct_answer, examples)
                generated_rationales.append({'question': question, 'rationale': rationalized_rationale['rationale'], 'answer': correct_answer, 'correct': False})

        correct_rationales = filter_rationales(dataset, generated_rationales)
        examples.extend(correct_rationales)
    return examples

