from openai import OpenAI
import re

reasoning_model = "qwen-qwq-32b"
image_model = "google/gemma-3-4b-it:free"

judge_model = "hf.co/flowaicom/Flow-Judge-v0.1-GGUF:latest"

def text_to_text(client: OpenAI , message:str, model: str, max_completion_tokens=1024,
):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message
                    },
                ]
            }
        ],
        max_completion_tokens=max_completion_tokens,
    )

    return completion.choices[0].message.content


def text_image_to_text(client:OpenAI, message:str,base_64_image, model , max_completion_tokens=1024):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": message
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base_64_image
                        }
                    }
                ]
            }
        ],
        max_completion_tokens=max_completion_tokens,
    )

    return completion.choices[0].message.content


def generate_question(client,image,thematique,qcm=False):
    prompt = """
    Using the following image : <image>

    Generate a question to help student learn the notion on the document but don't mantion the document.

    The student whant to be tested on this thematique : THEMATIQUE
    You will output the generated question in a <question> <question/> tag and also add an answer in an <answer> <answer/> tag. 
    
    You will also add in an <legal_basis> <legal_basis/> the text in the document that can serve to give the right answer
    Don't add any additional text.

    
    QTYPE

    Don't add any additionial text the output format need to strictly follow this scheme :
    <question>
    ...
    </question>
    <answer>
    ...
    </answer>
    <legal_basis>
    ...
    </legal_basis>
    """
    if qcm:
        prompt = prompt.replace("QTYPE","The question need to be a Multiple choice question with A , B , C , D.")
    else:
        prompt = prompt.replace("QTYPE","The question need to be an open question")
        
    prompt = prompt.replace("THEMATIQUE",thematique)

    pattern = re.compile(
    r"<question>\s*(.*?)\s*</question>\s*"
    r"<answer>\s*(.*?)\s*</answer>\s*"
    r"<legal_basis>\s*(.*?)\s*</legal_basis>",
    re.DOTALL  # Allows matching across multiple lines
    )
    result = text_image_to_text(client,prompt,image,image_model,max_completion_tokens=5000)
    match = pattern.search(result)
    if match:
        

        return {
            "question": match.group(1).strip(),
            "answer":match.group(2).strip(),
            "legal_basis":match.group(3).strip()
        }
    return {'message':result}


def analyse_answer(client:OpenAI,question:str,user_answer:str,real_answer:str, criteria:str,scoring:str):
    prompt ="""
    # GOAL
    You are an teacher expert in european patent , your goal is to evaluate the answer from a student to an exam question.
    
    
    # EXAMEN QUESTION
    EXAMENQUEST
    
    # USER ANSWER
    USERANS

    #REAL ANSWER
    REALANS
  
    
    # EVALUATION CRITERIA AND SCORING RUBRIC
    Here are the evaluation criteria and the rubric that you need to use for evaluating the task:
    <evaluation_criteria>
    EVALCRITERIA
    </evaluation_criteria>
    
    <scoring_rubric>
    SCORINGMEC
    </scoring_rubric>
    
    
    # INSTRUCTIONS FOR THE EVALUATION
    Understand the task and criteria: Familiarize yourself with the task to be evaluated. Review the evaluation criteria and scoring rubric to understand the different levels of performance and the descriptions for each score.
    
    Review the inputs and output: Look at the inputs provided for the task. Examine the output generated from completing the task.
    
    Compare output to score descriptions: Compare the output against the criteria and score descriptions in the scoring rubric. For each criterion,decide which description best matches the output.
    
    After comparing the output to the score descriptions, pay attention to the small details that might impact the final score that you assign. Sometimes a small difference can dictate the final score.
    
    Write verbal feedback justifying your evaluation that includes a detailed rationale, referring to specific aspects of the output and comparing them to the rubric also give advice for providing a better answer next time in your advice include the article to review.
    
    Don't use any additional knowledge to give your feedback only use the data provided in the real answer
    Assign a final score based on the scoring rubric.
    
    The student don't have access to the real answer so if he is wrong give to the student a good explanation why he is wrong
    # FORMAT FOR THE EVALUATION
    Write the verbal feedback inside <feedback> tags without any additional surrounding text.
    
    Write the numeric score inside <score> tags, without any additional surrounding text and always after the feedback.
    
    Please accurately evaluate the task. Strictly adhere to the evaluation criteria and rubric.


    """

    prompt = prompt.replace("EXAMENQUEST",question)
    prompt = prompt.replace("USERANS",user_answer)
    prompt = prompt.replace("REALANS",real_answer)
    prompt = prompt.replace("EVALCRITERIA",criteria)
    prompt = prompt.replace("SCORINGMEC",scoring)

    result = text_to_text(client,prompt,reasoning_model,9048)
    pattern = re.compile(
    r"<think>\s*(.*?)\s*</think>\s*"
    r"<feedback>\s*(.*?)\s*</feedback>\s*"
    r"<score>\s*(.*?)\s*</score>",
    re.DOTALL  # Allows matching across multiple lines
    )
    # Extract matches
    match = pattern.search(result)

    if match:
        result = {
            "think": match.group(1).strip(),
            "feedback": match.group(2).strip(),
            "score": int(match.group(3))
        }
        return result
    return {"output" : result}
