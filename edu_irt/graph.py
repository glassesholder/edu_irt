from .models import irf_1pl, irf_2pl, irf_3pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## 문항정보함수
def item_information_1PL(theta, b):
    P = irf_1pl(theta, b)
    return (P * (1-P))

def item_information_2PL(theta, a, b):
    P = irf_2pl(theta, a, b)
    return (a**2 * P * (1-P))

def item_information_3PL(theta, a, b, c):
    P = irf_3pl(theta, a, b, c)
    return (a**2 * ((P-c)**2) * (1-P)) / ((1-c)**2 * P)


## 검사특성곡선(1PL)
def tcc_1PL(data, theta_values=np.linspace(-3, 3, 100)):
    """
    주어진 데이터프레임에서 검사 특성 곡선을 그리고 figure 객체를 반환하는 함수.
    그래프를 화면에 보여주고 동시에 figure 객체를 반환.
    
    Parameters:
    - data: DataFrame with 'Difficulty' column.
    - theta_values: 능력 값 범위 (default: -3 to 3)
    
    Returns:
    - fig: matplotlib의 figure 객체
    """

    # 각 문항에 대한 예상 점수 합산
    test_characteristic = np.zeros_like(theta_values)

    # 각 문항에 대한 특성 곡선 계산
    for index, row in data.iterrows():
        b= row['Difficulty']
        test_characteristic += irf_1pl(theta_values, b)

    # Figure 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 검사 특성 곡선 시각화
    ax.plot(theta_values, test_characteristic, label="Test Characteristic Curve", color="green", lw=2)
    ax.set_title("Test Characteristic Curve (TCC)", fontsize=16)
    ax.set_xlabel("Student Ability (θ)", fontsize=14)
    ax.set_ylabel("Expected Score", fontsize=14)
    ax.grid(True)
    ax.legend()

    # 그래프를 즉시 화면에 표시
    plt.show()

    # figure 객체 반환
    return fig



# 검사정보곡선(1PL)
def tic_1PL(data, theta_values=np.linspace(-3, 3, 100)):
    """
    주어진 데이터프레임에서 검사 정보 곡선을 그리고 figure 객체를 반환하는 함수.
    그래프를 화면에 보여주고 동시에 figure 객체를 반환.
    
    Parameters:
    - data: DataFrame with 'Difficulty' column.
    - theta_values: 능력 값 범위 (default: -3 to 3)
    
    Returns:
    - fig: matplotlib의 figure 객체
    """

    # 초기 검사 정보 배열 (0으로 초기화)
    test_information = np.zeros_like(theta_values)

    # 각 문항에 대한 정보 함수 계산 및 더하기
    for index, row in data.iterrows():
        b = row['Difficulty']
        test_information += item_information_1PL(theta_values, b)

    # Figure 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 검사 정보 곡선 시각화
    ax.plot(theta_values, test_information, label="Test Information Function", color="blue", lw=2)
    ax.set_title("Test Information Curve (TIF)", fontsize=16)
    ax.set_xlabel("Student Ability (θ)", fontsize=14)
    ax.set_ylabel("Information", fontsize=14)
    ax.grid(True)
    ax.legend()

    # 그래프를 즉시 화면에 표시
    plt.show()

    # figure 객체 반환
    return fig


## 검사특성곡선(2PL)
def tcc_2PL(data, theta_values=np.linspace(-3, 3, 100)):
    """
    주어진 데이터프레임에서 검사 특성 곡선을 그리고 figure 객체를 반환하는 함수.
    그래프를 화면에 보여주고 동시에 figure 객체를 반환.
    
    Parameters:
    - data: DataFrame with 'Discrimination', 'Difficulty' columns.
    - theta_values: 능력 값 범위 (default: -3 to 3)
    
    Returns:
    - fig: matplotlib의 figure 객체
    """

    # 각 문항에 대한 예상 점수 합산
    test_characteristic = np.zeros_like(theta_values)

    # 각 문항에 대한 특성 곡선 계산
    for index, row in data.iterrows():
        a, b = row['Discrimination'], row['Difficulty']
        test_characteristic += irf_2pl(theta_values, a, b)

    # Figure 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 검사 특성 곡선 시각화
    ax.plot(theta_values, test_characteristic, label="Test Characteristic Curve", color="green", lw=2)
    ax.set_title("Test Characteristic Curve (TCC)", fontsize=16)
    ax.set_xlabel("Student Ability (θ)", fontsize=14)
    ax.set_ylabel("Expected Score", fontsize=14)
    ax.grid(True)
    ax.legend()

    # 그래프를 즉시 화면에 표시
    plt.show()

    # figure 객체 반환
    return fig



# 검사정보곡선(2PL)
def tic_2PL(data, theta_values=np.linspace(-3, 3, 100)):
    """
    주어진 데이터프레임에서 검사 정보 곡선을 그리고 figure 객체를 반환하는 함수.
    그래프를 화면에 보여주고 동시에 figure 객체를 반환.
    
    Parameters:
    - data: DataFrame with 'Discrimination', 'Difficulty' columns.
    - theta_values: 능력 값 범위 (default: -3 to 3)
    
    Returns:
    - fig: matplotlib의 figure 객체
    """

    # 초기 검사 정보 배열 (0으로 초기화)
    test_information = np.zeros_like(theta_values)

    # 각 문항에 대한 정보 함수 계산 및 더하기
    for index, row in data.iterrows():
        a, b = row['Discrimination'], row['Difficulty']
        test_information += item_information_2PL(theta_values, a, b)

    # Figure 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 검사 정보 곡선 시각화
    ax.plot(theta_values, test_information, label="Test Information Function", color="blue", lw=2)
    ax.set_title("Test Information Curve (TIF)", fontsize=16)
    ax.set_xlabel("Student Ability (θ)", fontsize=14)
    ax.set_ylabel("Information", fontsize=14)
    ax.grid(True)
    ax.legend()

    # 그래프를 즉시 화면에 표시
    plt.show()

    # figure 객체 반환
    return fig


## 검사특성곡선(3PL)
def tcc_3PL(data, theta_values=np.linspace(-3, 3, 100)):
    """
    주어진 데이터프레임에서 검사 특성 곡선을 그리고 figure 객체를 반환하는 함수.
    그래프를 화면에 보여주고 동시에 figure 객체를 반환.
    
    Parameters:
    - data: DataFrame with 'Discrimination', 'Difficulty', 'Guessing' columns.
    - theta_values: 능력 값 범위 (default: -3 to 3)
    
    Returns:
    - fig: matplotlib의 figure 객체
    """

    # 각 문항에 대한 예상 점수 합산
    test_characteristic = np.zeros_like(theta_values)

    # 각 문항에 대한 특성 곡선 계산
    for index, row in data.iterrows():
        a, b, c = row['Discrimination'], row['Difficulty'], row['Guessing']
        test_characteristic += irf_3pl(theta_values, a, b, c)

    # Figure 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 검사 특성 곡선 시각화
    ax.plot(theta_values, test_characteristic, label="Test Characteristic Curve", color="green", lw=2)
    ax.set_title("Test Characteristic Curve (TCC)", fontsize=16)
    ax.set_xlabel("Student Ability (θ)", fontsize=14)
    ax.set_ylabel("Expected Score", fontsize=14)
    ax.grid(True)
    ax.legend()

    # 그래프를 즉시 화면에 표시
    plt.show()

    # figure 객체 반환
    return fig



# 검사정보곡선(3PL)
def tic_3PL(data, theta_values=np.linspace(-3, 3, 100)):
    """
    주어진 데이터프레임에서 검사 정보 곡선을 그리고 figure 객체를 반환하는 함수.
    그래프를 화면에 보여주고 동시에 figure 객체를 반환.
    
    Parameters:
    - data: DataFrame with 'Discrimination', 'Difficulty', 'Guessing' columns.
    - theta_values: 능력 값 범위 (default: -3 to 3)
    
    Returns:
    - fig: matplotlib의 figure 객체
    """

    # 초기 검사 정보 배열 (0으로 초기화)
    test_information = np.zeros_like(theta_values)

    # 각 문항에 대한 정보 함수 계산 및 더하기
    for index, row in data.iterrows():
        a, b, c = row['Discrimination'], row['Difficulty'], row['Guessing']
        test_information += item_information_3PL(theta_values, a, b, c)

    # Figure 객체 생성
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 검사 정보 곡선 시각화
    ax.plot(theta_values, test_information, label="Test Information Function", color="blue", lw=2)
    ax.set_title("Test Information Curve (TIF)", fontsize=16)
    ax.set_xlabel("Student Ability (θ)", fontsize=14)
    ax.set_ylabel("Information", fontsize=14)
    ax.grid(True)
    ax.legend()

    # 그래프를 즉시 화면에 표시
    plt.show()

    # figure 객체 반환
    return fig
