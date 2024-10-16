import pandas as pd

# 문항모수 추출 결과 검증
def mean_index_diff(df1, df2, sort_column):
    """
    두 데이터프레임의 지정된 열을 기준으로 정렬한 후, 
    원래 인덱스의 차이값들의 평균을 계산하는 함수.
    
    Parameters:
    df1 (pd.DataFrame): 첫 번째 데이터프레임
    df2 (pd.DataFrame): 두 번째 데이터프레임
    sort_column (str): 정렬 기준이 되는 열 이름

    Returns:
    float: 원래 인덱스 차이의 평균값
    """
    # 두 데이터프레임을 지정된 열 기준으로 정렬
    sorted_df_1 = df1.sort_values(by=sort_column)
    sorted_df_2 = df2.sort_values(by=sort_column)

    sorted_list1 = sorted_df_1.index.tolist()
    sorted_list2 = sorted_df_2.index.tolist()


    sum = 0

    for i in range(1, len(sorted_list1)+1):
        a = abs(sorted_list1.index(i) - sorted_list2.index(i))
        sum += a

    # 인덱스 차이의 평균값 계산
    mean_index_diff_value = sum / len(sorted_df_1)
    
    return mean_index_diff_value