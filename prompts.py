SEQ2SEQ_PROMPT_FULL = (
    # "Below is an instruction that describes a task, paired with an input that provides further context.\n"
    "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
    # "Write a response that appropriately completes the request.\n"
    "요청을 적절히 완료하는 응답을 작성하세요.\n\n"
    "### Instruction(명령어):\n"
    "{instruction_text}\n\n"
    "### Input(입력):\n"
    "{input_text}"
)

SEQ2SEQ_PROMPT_NO_INSTRUCTION = (
    "{input_text}"
)

SEQ2SEQ_PROMPT_NO_INPUT = (
    # "Below is an instruction that describes a task.\n"
    "아래는 작업을 설명하는 명령어입니다.\n\n"
    # "Write a response that appropriately completes the request.\n"
    "명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
    "### Instruction(명령어):\n"
    "{instruction_text}"
)



def construct_seq2seq_prompt(instruction_text, input_text):
    if not instruction_text:
        prompt = SEQ2SEQ_PROMPT_NO_INSTRUCTION.format(input_text=input_text)
    
    elif not input_text:
        prompt = SEQ2SEQ_PROMPT_NO_INPUT.format(instruction_text=instruction_text)
    
    else:
        prompt = SEQ2SEQ_PROMPT_FULL.format(instruction_text=instruction_text, input_text=input_text)
    
    return prompt