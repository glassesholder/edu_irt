import numpy as np
import pandas as pd
from scipy.optimize import minimize




## 1PL 모형 문항모수 추출
# 1PL 문항 반응 함수
def irf_1pl(theta, b):
    return 1 / (1 + np.exp(-1.7*(theta - b)))

# 로그우도 함수 (Log-Likelihood)
def log_likelihood_1PL(params, responses, theta):
    b = params
    p = irf_1pl(theta, b)
    p = np.clip(p, 1e-10, 1 - 1e-10)  # 수치 안정성을 위한 클리핑
    likelihood = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
    return -likelihood  # 최소화 문제로 변환

# E-step: 능력값 추정 (여러 문항을 사용)
def estimate_theta_1PL(response_matrix, b_list):
    n_items = response_matrix.shape[1]  # 문항 수
    n_persons = response_matrix.shape[0]  # 피험자 수
    thetas = np.zeros(n_persons)
    
    for j in range(n_persons):
        responses = response_matrix[j, :]
        def theta_log_likelihood_1PL(theta):
            p = irf_1pl(theta, b_list)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # 수치 안정성을 위한 클리핑
            likelihood = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
            return -likelihood

        result = minimize(theta_log_likelihood_1PL, 0, bounds=[(-3, 3)])
        thetas[j] = result.x[0]
    
    return thetas

# M-step: 문항 모수 추정 (각 문항마다 개별적으로)
def estimate_item_parameters_1PL(response_matrix, thetas, initial_params_list):
    n_items = response_matrix.shape[1]
    estimated_params = []

    for i in range(n_items):
        responses = response_matrix[:, i]
        initial_params = initial_params_list[i]
        result = minimize(log_likelihood_1PL, initial_params, args=(responses, thetas), method='SLSQP',
                          bounds=[(-3, 3)])  # b에 대한 적절한 제한값 설정
        estimated_params.append(result.x)
    
    return np.array(estimated_params)  # 각 문항의 b(문항난이도) 반환

# EM 알고리즘
def em_1PL(df, max_iter=300, tol=1e-5):
    response_matrix = df.values
    n_items = response_matrix.shape[1]
    n_persons = response_matrix.shape[0]

    # 초기 추정값 (각 문항에 대해 b)
    initial_params_list = [[0.0]]*n_items  # 문항 6의 b 초기값

    # 초기 문항 모수
    b_list = np.array([params[0] for params in initial_params_list])

    thetas = np.zeros(n_persons)  # 초기 능력값은 0으로 설정
    
    for iteration in range(max_iter):
        # E-step: 능력값 추정
        new_thetas = estimate_theta_1PL(response_matrix, b_list)
        
        # M-step: 문항 모수 추정
        new_params = estimate_item_parameters_1PL(response_matrix, new_thetas, initial_params_list)

        # 새로운 문항 모수로 업데이트
        new_b_list = new_params[:, 0]
        
        # 수렴 조건 체크
        if np.max(np.abs(new_b_list - b_list)) < tol:
            print(f"Converged after {iteration+1} iterations")
            break
        
        # 업데이트된 문항 모수로 initial_params_list 업데이트
        initial_params_list = new_params.tolist()

        # 업데이트
        b_list = new_b_list
        thetas = new_thetas

    df_1PL_1 = pd.DataFrame({
        'Difficulty': b_list,
    })

    df_1PL_2 = pd.DataFrame(thetas, columns=['Student_abilities'])

    df_1PL_1.index = range(1, len(b_list)+1)
    df_1PL_2.index = range(1, len(thetas)+1)
    
    return df_1PL_1, df_1PL_2




## 2PL 모형 문항모수 추출
# 2PL 문항 반응 함수
def irf_2pl(theta, a, b):
    return 1 / (1 + np.exp(-1.7*a*(theta - b)))

# 로그우도 함수 (Log-Likelihood)
def log_likelihood_2PL(params, responses, theta):
    a, b = params
    p = irf_2pl(theta, a, b)
    p = np.clip(p, 1e-10, 1 - 1e-10)  # 수치 안정성을 위한 클리핑
    likelihood = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
    return -likelihood  # 최소화 문제로 변환

# E-step: 능력값 추정 (여러 문항을 사용)
def estimate_theta_2PL(response_matrix, a_list, b_list):
    n_items = response_matrix.shape[1]  # 문항 수
    n_persons = response_matrix.shape[0]  # 피험자 수
    thetas = np.zeros(n_persons)
    
    for j in range(n_persons):
        responses = response_matrix[j, :]
        def theta_log_likelihood_2PL(theta):
            p = irf_2pl(theta, a_list, b_list)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # 수치 안정성을 위한 클리핑
            likelihood = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
            return -likelihood

        result = minimize(theta_log_likelihood_2PL, 0, bounds=[(-3, 3)])
        thetas[j] = result.x[0]
    
    return thetas

# M-step: 문항 모수 추정 (각 문항마다 개별적으로)
def estimate_item_parameters_2PL(response_matrix, thetas, initial_params_list):
    n_items = response_matrix.shape[1]
    estimated_params = []

    for i in range(n_items):
        responses = response_matrix[:, i]
        initial_params = initial_params_list[i]
        result = minimize(log_likelihood_2PL, initial_params, args=(responses, thetas), method='SLSQP',
                          bounds=[(0.1, 2), (-3, 3)])  # a, b에 대한 적절한 제한값 설정
        estimated_params.append(result.x)
    
    return np.array(estimated_params)  # 각 문항의 a, b반환

# EM 알고리즘
def em_2PL(df, max_iter=300, tol=1e-5):
    response_matrix = df.values
    n_items = response_matrix.shape[1]
    n_persons = response_matrix.shape[0]

    # 초기 추정값 (각 문항에 대해 a, b)
    initial_params_list = [[1, 0.0]]*n_items  # 문항 6의 a, b초기값

    # 초기 문항 모수
    a_list = np.array([params[0] for params in initial_params_list])
    b_list = np.array([params[1] for params in initial_params_list])

    thetas = np.zeros(n_persons)  # 초기 능력값은 0으로 설정
    
    for iteration in range(max_iter):
        # E-step: 능력값 추정
        new_thetas = estimate_theta_2PL(response_matrix, a_list, b_list)
        
        # M-step: 문항 모수 추정
        new_params = estimate_item_parameters_2PL(response_matrix, new_thetas, initial_params_list)

        # 새로운 문항 모수로 업데이트
        new_a_list = new_params[:, 0]
        new_b_list = new_params[:, 1]
        
        # 수렴 조건 체크
        if np.max(np.abs(new_a_list - a_list)) < tol and \
           np.max(np.abs(new_b_list - b_list)) < tol:
            print(f"Converged after {iteration+1} iterations")
            break
        
        # 업데이트된 문항 모수로 initial_params_list 업데이트
        initial_params_list = new_params.tolist()

        # 업데이트
        a_list, b_list = new_a_list, new_b_list
        thetas = new_thetas

    df_2PL_1 = pd.DataFrame({
        'Discrimination': a_list,
        'Difficulty': b_list
    })

    df_2PL_2 = pd.DataFrame(thetas, columns=['Student_abilities'])

    df_2PL_1.index = range(1, len(a_list)+1)
    df_2PL_2.index = range(1, len(thetas)+1)
    
    return df_2PL_1, df_2PL_2




## 3PL 모형 문항모수 추출
# 3PL 문항 반응 함수
def irf_3pl(theta, a, b, c):
    return c + (1 - c) / (1 + np.exp(-1.7* a* (theta - b)))

# 로그우도 함수 (Log-Likelihood)
def log_likelihood_3PL(params, responses, theta):
    a, b, c = params
    p = irf_3pl(theta, a, b, c)
    p = np.clip(p, 1e-10, 1 - 1e-10)  # 수치 안정성을 위한 클리핑
    likelihood = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
    return -likelihood  # 최소화 문제로 변환

# E-step: 능력값 추정 (여러 문항을 사용)
def estimate_theta_3PL(response_matrix, a_list, b_list, c_list):
    n_items = response_matrix.shape[1]  # 문항 수
    n_persons = response_matrix.shape[0]  # 피험자 수
    thetas = np.zeros(n_persons)
    
    for j in range(n_persons):
        responses = response_matrix[j, :]
        def theta_log_likelihood_3PL(theta):
            p = irf_3pl(theta, a_list, b_list, c_list)
            p = np.clip(p, 1e-10, 1 - 1e-10)  # 수치 안정성을 위한 클리핑
            likelihood = np.sum(responses * np.log(p) + (1 - responses) * np.log(1 - p))
            return -likelihood

        result = minimize(theta_log_likelihood_3PL, 0, bounds=[(-3, 3)])
        thetas[j] = result.x[0]
    
    return thetas

# M-step: 문항 모수 추정 (각 문항마다 개별적으로)
def estimate_item_parameters_3PL(response_matrix, thetas, initial_params_list):
    n_items = response_matrix.shape[1]
    estimated_params = []

    for i in range(n_items):
        responses = response_matrix[:, i]
        initial_params = initial_params_list[i]
        result = minimize(log_likelihood_3PL, initial_params, args=(responses, thetas), method='SLSQP',
                          bounds=[(0.1, 2), (-3, 3), (0, 0.2)])  # a, b, c에 대한 적절한 제한값 설정
        estimated_params.append(result.x)
    
    return np.array(estimated_params)  # 각 문항의 a, b, c 반환

# EM 알고리즘
def em_3PL(df, max_iter=300, tol=1e-5):
    response_matrix = df.values
    n_items = response_matrix.shape[1]
    n_persons = response_matrix.shape[0]

    # 초기 추정값 (각 문항에 대해 a, b, c)
    initial_params_list = [[1, 0.0, 0.2]]*n_items  # 문항 6의 a, b, c 초기값

    # 초기 문항 모수
    a_list = np.array([params[0] for params in initial_params_list])
    b_list = np.array([params[1] for params in initial_params_list])
    c_list = np.array([params[2] for params in initial_params_list])

    thetas = np.zeros(n_persons)  # 초기 능력값은 0으로 설정
    
    for iteration in range(max_iter):
        # E-step: 능력값 추정
        new_thetas = estimate_theta_3PL(response_matrix, a_list, b_list, c_list)
        
        # M-step: 문항 모수 추정
        new_params = estimate_item_parameters_3PL(response_matrix, new_thetas, initial_params_list)

        # 새로운 문항 모수로 업데이트
        new_a_list = new_params[:, 0]
        new_b_list = new_params[:, 1]
        new_c_list = new_params[:, 2]
        
        # 수렴 조건 체크
        if np.max(np.abs(new_a_list - a_list)) < tol and \
           np.max(np.abs(new_b_list - b_list)) < tol and \
           np.max(np.abs(new_c_list - c_list)) < tol:
            print(f"Converged after {iteration+1} iterations")
            break
        
        # 업데이트된 문항 모수로 initial_params_list 업데이트
        initial_params_list = new_params.tolist()

        # 업데이트
        a_list, b_list, c_list = new_a_list, new_b_list, new_c_list
        thetas = new_thetas

        df_3PL_1 = pd.DataFrame({
            'Discrimination': a_list,
            'Difficulty': b_list,
            'Guessing': c_list
        })

        df_3PL_2 = pd.DataFrame(thetas, columns=['Student_abilities'])

        df_3PL_1.index = range(1, len(a_list)+1)
        df_3PL_2.index = range(1, len(thetas)+1)
    
    return df_3PL_1, df_3PL_2