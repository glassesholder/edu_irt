import numpy as np
import pandas as pd

## 1PL 모형에 대한 데이터 생성
def generate_qa_data_1PL(n_students=100, n_questions=30, random_state=42):
    # 랜덤 시드 설정
    np.random.seed(random_state)
    
    # 학생 능력 수준 생성(정규분포에 맞게 생성)
    student_abilities = np.random.normal(0, 1, n_students)
    
    # 문항난이도 생성
    question_difficulties = np.random.normal(0, 1, n_questions)
    
    # 개별 학생이 문항을 맞출 확률 계산
    probabilities = 1 / (1 + np.exp(-1.7*(student_abilities[:, None] - question_difficulties)))
    
    # 문항을 맞출 확률에 의거해서 정오답 데이터 생성
    data = np.random.binomial(1, probabilities)
    
    # 정오답 데이터, 학생 능력 수준, 문항모수 데이터 프레임 생성
    df1 = pd.DataFrame(data, columns=[f'Q{i+1}' for i in range(n_questions)])
    df2 = pd.DataFrame(student_abilities, columns=['Student_abilities'])
    df3 = pd.DataFrame(question_difficulties, columns=['Difficulty'])

    # 인덱스를 1부터 시작하도록 설정
    df1.index = range(1, n_students + 1)
    df2.index = range(1, n_students + 1)
    df3.index = range(1, n_questions + 1)
    
    return df1, df2, df3




## 2PL 모형에 대한 데이터 생성
def generate_qa_data_2PL(n_students=100, n_questions=30, random_state=42):
    # 랜덤 시드 설정
    np.random.seed(random_state)
    
    # 학생 능력 수준 생성(정규분포에 맞게 생성)
    student_abilities = np.random.normal(0, 1, n_students)
    
    # 문항난이도 생성
    question_difficulties = np.random.normal(0, 1, n_questions)
    
    # 문항변별도 생성 (양수로 제한)
    question_discrimination = np.random.uniform(0.1, 2, n_questions)  # 예: 0.5에서 2 사이의 값
    
    # 개별 학생이 문항을 맞출 확률 계산 (2PL 모델)
    probabilities = 1 / (1 + np.exp(-(1.7*question_discrimination) * (student_abilities[:, None] - question_difficulties)))
    
    # 문항을 맞출 확률에 의거해서 정오답 데이터 생성
    data = np.random.binomial(1, probabilities)
    
    # 정오답 데이터, 학생 능력 수준, 문항모수 데이터 프레임 생성
    df1 = pd.DataFrame(data, columns=[f'Q{i+1}' for i in range(n_questions)])
    df2 = pd.DataFrame(student_abilities, columns=['Student_abilities'])
    df3 = pd.DataFrame({
        'Discrimination': question_discrimination,
        'Difficulty': question_difficulties,
    })
    
    # 인덱스를 1부터 시작하도록 설정
    df1.index = range(1, n_students + 1)
    df2.index = range(1, n_students + 1)
    df3.index = range(1, n_questions + 1)
    
    return df1, df2, df3




## 3PL 모형에 대한 데이터 생성
def generate_qa_data_3PL(n_students=100, n_questions=30, random_state=42):
    # 랜덤 시드 설정
    np.random.seed(random_state)
    
    # 학생 능력 수준 생성(정규분포에 맞게 생성)
    student_abilities = np.random.normal(0, 1, n_students)
    
    # 문항난이도 생성
    question_difficulties = np.random.normal(0, 1, n_questions)
    
    # 문항변별도 생성 (양수로 제한)
    question_discrimination = np.random.uniform(0.1, 2, n_questions)  # 0.5에서 3 사이의 값
    
    # 문항추측도 생성 (0에서 0.2 사이의 값)
    guessing_parameter = np.random.uniform(0, 0.2, n_questions)
    
    # 개별 학생이 문항을 맞출 확률 계산 (3PL 모델)
    probabilities = guessing_parameter + (1 - guessing_parameter) / (1 + np.exp(-(1.7*question_discrimination) * (student_abilities[:, None] - question_difficulties)))
    
    # 문항을 맞출 확률에 의거해서 정오답 데이터 생성
    data = np.random.binomial(1, probabilities)
    
    # 정오답 데이터, 학생 능력 수준, 문항 모수 데이터 프레임 생성
    df1 = pd.DataFrame(data, columns=[f'Q{i+1}' for i in range(n_questions)])
    df2 = pd.DataFrame(student_abilities, columns=['Student_abilities'])
    df3 = pd.DataFrame({
        'Discrimination': question_discrimination,
        'Difficulty': question_difficulties,
        'Guessing': guessing_parameter
    })
    
    # 인덱스를 1부터 시작하도록 설정
    df1.index = range(1, n_students + 1)
    df2.index = range(1, n_students + 1)
    df3.index = range(1, n_questions + 1)
    
    return df1, df2, df3