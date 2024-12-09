import os, json, random, openai, re
import pandas as pd
import numpy as np

class Utils:
    def __init__(self,
                 api_key,
                 path):
        self.client = openai.OpenAI(api_key=api_key)
        self.path = path
        self.batch_id = None

    def gpt_batch_request(self,
                          filename):
        batch_input_file = self.client.files.create(
            file=open(filename, "rb"),
            purpose="batch"
        )

        completion = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        self.batch_id = completion.id
        print('batch_id : {}'.format(self.batch_id))

    def batch_status(self):
        print(self.client.batches.retrieve(self.batch_id).status)
        print(self.client.batches.retrieve(self.batch_id).request_counts)

    def gpt_result_file_save(self,
                             filename):
        if self.client.batches.retrieve(self.batch_id).status not in ['completed']:
            print('Not yet complete')
            return None
        output_file_id = self.client.batches.retrieve(self.batch_id).output_file_id
        result = self.client.files.content(output_file_id).content
        with open(filename, 'wb') as f:
            f.write(result)
        print('complete and save')

def load_from_jsonl(
    filename
):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_to_jsonl(
    filename,
    data
):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')

def prompt_json(
    id,
    inst,
    prompt,
    model_id='gpt-4o-mini',
    max_tokens=1000
):
    res_json =  {'custom_id': id,
                 'method': 'POST',
                 'url': '/v1/chat/completions',
                 'body': {'model': model_id,
                         'messages': [{'role': 'system',
                                       'content': inst},
                                       {'role': 'user',
                                       'content': prompt}],
                         'max_tokens': max_tokens}}
    return res_json

def step_2_prompt_gen(
    word_data,
    mode,
    file_name
):
    if mode == 'fin':
        inst = '이 GPT는 사용자가 제공한 주제를 기반으로 금융시장, 금융 개념, 시장 동향, 투자 전략 등을 포괄하는 내용을 자세하게 설명합니다.'
    else:
        inst = '이 GPT는 사용자가 제공한 주제를 기반으로 재무, 회계, 화폐 가치 등을 포괄하는 내용을 자세하게 설명합니다.'
    prompt = '주제 : {}'

    gpt_prompt_lst = []
    for word in word_data:
        id = '{}_{}'.format(mode, word)
        llm_prompt = prompt_json(id, inst, prompt.format(word), model_id='gpt-4o-mini', max_tokens=2500)
        gpt_prompt_lst.append(llm_prompt)
    print('prompt_size : {}'.format(gpt_prompt_lst))
    return gpt_prompt_lst

def step_3_prompt_gen(
    context_data
):
    fin_inst = '''다음 주어지는 문단에 대해서 금융시장, 금융 개념, 시장 동향, 투자 전략, 재무, 회계, 화폐 가치 등의 내용을 하나라도 포함하고 있는지 판별하시오.
먼저 간단한 1문장의 이유와 판별 결과를 True, False로 출력하시오. 출력 포맷은 다음과 같습니다.
이유 : <str>
판별 : <bool>'''
    acc_inst = '''다음 주어지는 문단에 대해서 재무, 회계, 화폐 가치 등의 내용을 하나라도 포함하고 있는지 판별하시오.
먼저 간단한 1문장의 이유와 판별 결과를 True, False로 출력하시오. 출력 포맷은 다음과 같습니다.
이유 : <str>
판별 : <bool>'''
    prompt = '{}'

    gpt_prompt_lst = []
    for data in context_data:
        id = '{}'.format(data['id'])
        inst = fin_inst if id[-3:]=='fin' else acc_inst
        llm_prompt = prompt_json(id, inst, prompt.format(data['context']), model_id='gpt-4o-mini', max_tokens=100)
        gpt_prompt_lst.append(llm_prompt)
    print('prompt_size : {}'.format(gpt_prompt_lst))
    return gpt_prompt_lst

def step_4_prompt_gen(
    context_data
):
    inst = '''이 GPT는 사용자가 제공한 주제 또는 문단을 기반으로 {} 등을 포괄하는 MCQA(Multiple Choice Question and Answer) 형태의 퀴즈를 생성합니다. 문제는 학습용으로 적합하며, {}의 보기를 랜덤하게 생성합니다. 생성된 출력은 항상 한국어로 제공되며, 정답과 간단한 해설을 포함합니다. 각 문제는 다음 규칙을 따릅니다: 1) 질문 앞에는 '문제 1. ', '문제 2. ' 등과 같은 형식이 붙습니다. 2) 선택지 앞에는 'A. ', 'B. ', 'C. ', 'D. ' 등과 같은 형식이 붙으며, 각 선택지는 반드시 '~다.'로 끝나는 하나의 문장으로 작성됩니다. 3) 선택지의 개수는 {} 사이로 랜덤하게 설정됩니다. 4) 모든 문제의 정답은 선택지 중 단 하나만 포함됩니다. 5) 해설은 문제를 풀기 위한 논리적인 설명을 3문장 이내로 작성하며, 선택지의 각 항목 번호를 언급하면서 논리적으로 설명합니다. 예: 'A, B, D, E는 조정된 화폐 단위 사용의 직접적인 이유가 아닙니다. A, B, D는 사채의 정의가 아닙니다. C만이 사채의 정의를 정확하게 설명합니다.' 6) 주어진 문단 또는 주제를 최대한 활용하여 가능한 많은 문제를 생성합니다. 7) 각 문제들의 난이도는 어렵게 생성합니다. 생성되는 모든 문제는 {} 등과 관련되어야 합니다. GPT는 문단 내용을 충실히 반영하며, 학습 목적으로 적합한 퀴즈를 생성합니다.'''
    fin_topic = '금융시장, 금융 개념, 시장 동향, 투자 전략, 금융 수식 계산'
    acc_topic = '재무, 회계, 화폐 가치'
    choice_range = '6개에서 최대 8개'

    gpt_prompt_lst = []
    for i, data in enumerate(context_data):
        for j in range(2):
            id = '{}_{}'.format(data['id'], j)
            mode = data['id'].split('_')[0]

            context = data['context']
            if mode == 'fin':
                new_inst = inst.format(fin_topic, choice_range, choice_range, fin_topic)
            else:
                new_inst = inst.format(acc_topic, choice_range, choice_range, acc_topic)
            llm_prompt = prompt_json(id, new_inst, context, model_id='gpt-4o', max_tokens=2500)
            gpt_prompt_lst.append(llm_prompt)
    print('prompt_size : {}'.format(gpt_prompt_lst))
    return gpt_prompt_lst

def step_5_prompt_gen(
    quiz_data
):
    inst = '''당신은 곧 게시될 벤치마크의 최종 민감도 판독자입니다.
벤치마크에 포함된 질문을 읽고 해당 질문에 대한 답변이 가능한지 평가합니다.
문제는 주어지는 문제 이외의 추가 정보는 없습니다.
다음과 같은 경우 답변할 수 없는 질문으로 간주됩니다:

1. 질문이 {}과 관련되지 않은 경우.
2. 질문이 오래되었거나 더 이상 관련성이 없는 정보를 기반으로 하는 경우.
3. 문제를 풀기 위해 추가 데이터가 필요한 경우.
4. 일반적으로 구할 수 없거나 벤치마크 대상의 범위를 벗어난 전문 지식이 필요한 경우.
5. 질문의 문구가 모호하거나 똑같이 유효한 해석이 여러 개 나올 수 있는 경우.
6. 편견이나 고정관념을 조장하는 특정 집단에게 불이익을 줄 수 있는 편견이나 가정이 포함된 문제인 경우.

벤치마크의 각 문제가 이러한 기준을 충족하여 답변이 가능하고 게시하기에 적합한 것으로 간주되는지 확인하시기 바랍니다.
별도의 설명 없이 질문이 각 기준을 충족하고 답변 가능한 질문에는 True를, 답변할 수 없는 질문에는 False를 표시하세요.'''
    prompt = '{}\n\n### 해설: {}'

    gpt_prompt_lst = []
    for i, data in enumerate(quiz_data):
        id = data['id']
        mode = id[:3]

        quiz = data['prompt'].split('\n\n### 정답')[0]
        description = data['target'][:-2]
        description = description + '.' if description[-1] != '.' else description

        fin_topic = '''"금융시장", "금융 개념", "시장 동향", "투자 전략", "금융 수식 계산"'''
        acc_topic = '''"재무", "회계", "화폐 가치"'''
        new_inst = inst.format(fin_topic if mode=='fin' else acc_topic)

        llm_prompt = prompt_json(id, new_inst, prompt.format(quiz, description), model_id='gpt-4o', max_tokens=10)
        gpt_prompt_lst.append(llm_prompt)
    print('prompt_size : {}'.format(gpt_prompt_lst))
    return gpt_prompt_lst

def step_6_prompt_gen(
    quiz_data
):
    inst = '''This GPT specializes in solving multiple-choice quizzes (MCQs) efficiently and accurately, focusing on providing answers in Korean. When presented with a question and its options, it analyzes the content, applies reasoning, and selects the most likely correct answer. It explains its reasoning in Korean following this strict format:

1) 문제 이해: Summarizes the question in one sentence to ensure the user understands the core of the problem.
2) 문제 풀이: Begins by prompting the user to think carefully about how to solve the problem in one guiding sentence. Then, analyzes each option step-by-step in a systematic manner. Each option is addressed individually with concise reasoning (limited to two sentences) for each step. At the end, summarizes the reasoning process in one concluding sentence. Mathematical expressions or equations, if required, are written in plain text format rather than markdown. For example:

   - 문제를 해결하려면 먼저 각 선택지가 질문과 어떤 관련이 있는지 생각해야 합니다.
   - 선택지 A는 ROI가 특정 투자에서 발생한 총 수익을 측정하는 지표라고 설명하지만, ROI는 수익률이므로 이 설명은 부정확합니다.
   - 선택지 B는 투자 성과를 평가하기 위해 투자 금액 대비 얻은 순이익을 나타내는 지표라고 설명합니다. 이는 ROI의 정의에 부합합니다.
   - 선택지 C는 미래 현금 흐름의 현재 가치를 계산하는 방법을 설명하는 것으로, 이는 현재 가치 평가 방법에 해당하며 ROI와는 관련이 없습니다.
   - 선택지 D는 투자된 자본의 내부 비용을 파악하는 지표라고 설명하지만, 이는 ROI의 정의와 맞지 않습니다.
   - 종합적으로 볼 때, 선택지 B가 질문에 가장 적합한 답변입니다.

3) 최종 정답: Outputs only the letter of the correct option (e.g., "A", "B", "C").

The GPT avoids unnecessary details, ensures clarity in its explanations, and maintains a structured response suitable for a Korean-speaking audience.'''
    prompt = '{}'

    gpt_prompt_lst = []
    for i, data in enumerate(quiz_data):
        id = data['id']
        quiz = data['prompt'].split('\n\n### 정답')[0]

        llm_prompt = prompt_json(id, inst, prompt.format(quiz), model_id='gpt-4o', max_tokens=3000)
        gpt_prompt_lst.append(llm_prompt)
    print('prompt_size : {}'.format(gpt_prompt_lst))
    return gpt_prompt_lst