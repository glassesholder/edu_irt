# edu_irt
"pip install edu_irt"<br>
created an IRT model for estimating item parameters

# 인사말(Greeting)
안녕하세요!😁<br>
가상의 학습자 정오답 데이터를 생성하고 IRT 모델을 만들어 보았습니다.<br>
저희 패키지는 크게 4가지 기능을 제공합니다.

<br>
1 데이터 생성<br>
가상의 학습자 정오답 데이터를 생성할 수 있습니다.<br>
원하는 학생 수, 문제 수를 바탕으로 학생의 능력 수준과 문항별 문항모수를 임의로 생성해서 학생의 문제별 정오답 데이터를 제공하고 있습니다.<br>
<br>
2 문항모수 추정<br>
가상의 학습자 정오답 데이터 또는 가지고 계신 정오답 데이터(행: 학생, 열 : 문제)를 바탕으로 문항별 문항모수를 추정할 수 있습니다.<br>
1PL(문항난이도), 2PL(문항변별도, 문항난이도), 3PL(문항변별도, 문항난이도, 문항추측도) 모형을 바탕으로 추정할 수 있습니다.<br>
<br>
3 검사특성곡선, 검사정보곡선 시각화<br>
확인하고자 하는 문항별 문항모수를 바탕으로 검사특성곡선, 검사정보곡선을 확인할 수 있습니다.<br>
<br>
4 모델 별 정확도 측정<br>
다른 IRT 모델을 가지고 있다면 가상 데이터를 바탕으로 모델에 대한 정확도를 측정해볼 수 있습니다. 가상데이터를 바탕으로 추출한 문항모수와 비교하여 문항별 문항난이도 순서를 비교합니다.<br>
<br>
위 패키지는 버전 0.0.1입니다. 점차 발전시켜 나가는 중에 있으며 많은 조언 부탁드립니다.<br>
<br>
<br>
Hello!😁
<br>
I’ve created an IRT model using simulated student response data. Our package offers three main features.<br>
<br>
1 Data Generation<br>
You can generate simulated student response data. Based on your desired number of students and questions, the package randomly creates student ability levels and item parameters. It provides corresponding correct/incorrect responses for each question.<br>
<br>
2 Parameter Estimation<br>
You can estimate item parameters based on the simulated student response data or your own data (where rows represent students and columns represent questions). The package supports parameter estimation using the 1PL (item difficulty), 2PL (item discrimination, item difficulty), and 3PL (item discrimination, item difficulty, item guessing) models.<br>
<br>
3 Visualization of the Test Characteristic Curve(TCC) and Test Information Curve(TIC)<br>
You can check the test characteristic curve(TCC) and test information curve(TIC) based on the item parameters for each item you want to examine.<br>
<br>
4 Model Accuracy Measurement<br>
If you have other IRT models, you can measure their accuracy using our simulated data. By comparing the estimated item parameters with the simulated data, you can check the ordering of item difficulties.<br>


# 사용법(usage)
패키지 설치(package install) : pip install edu_irt

1. 데이터 생성(Data Generation)

```python
from edu_irt import datas
## 1PL 기반 데이터 생성(1PL data generation)
df1, df2, df3 = datas.generate_qa_data_1PL(n_students=100, n_questions=30, random_state=42) # default 100, 30, 42
## 2PL 기반 데이터 생성(1PL data generation)
df4, df5, df6 = datas.generate_qa_data_2PL(n_students=100, n_questions=30, random_state=42) # default 100, 30, 42
## 3PL 기반 데이터 생성(1PL data generation)
df7, df8, df9 = datas.generate_qa_data_3PL(n_students=100, n_questions=30, random_state=42) # default 100, 30, 42

## result
# df1,4,7은 정오답 데이터(response), df2,5,8는 학생 능력 수준(student ability levels), df3,6,9는 문항모수(item parameters)
```

2. 문항모수 추정(Parameter Estimation)

```python
from edu_irt import models
## 1PL 기반 문항모수 추정(1PL estimation)
df_1PL_1, df_1PL_2 = models.em_1PL(df1)
## 2PL 기반 문항모수 추정(2PL estimation)
df_2PL_1, df_2PL_2 = models.em_2PL(df4)
## 3PL 기반 문항모수 추정(3PL estimation)
df_3PL_1, df_3PL_2 = models.em_3PL(df7)

## result
# df_1PL_1, df_2PL_1, df_3PL_1은 문항모수(item parameters), df_1PL_2, df_2PL_2, df_3PL_3는 학생 능력 수준(student ability levels)
```

3. 검사특성곡선, 검사정보곡선 시각화(TCC, TIC Visualization)

```python
from edu_irt import graph
## 1PL 기반 검사특성곡선, 검사정보곡선 시각화(1PL TCC, TIC)
a = graph.tcc_1PL(df3)
b = graph.tic_1PL(df3)
## 2PL 기반 검사특성곡선, 검사정보곡선 시각화(2PL TCC, TIC)
c = graph.tcc_2PL(df6)
d = graph.tic_2PL(df6)
## 3PL 기반 검사특성곡선, 검사정보곡선 시각화(3PL TCC, TIC)
e = graph.tcc_1PL(df9)
f = graph.tic_1PL(df9)

## result
# graph of TCC, TIC
```

4. 모델 정확도 측정(Model Accuracy Measurement)

```python
from edu_irt import test
## 가상 데이터의 문항모수와 가상 정오답 데이터를 IRT 모델로 분석한 문항모수를 바탕으로 문항난이도 순서 비교
## Comparison of Item Parameter order based on Item Parameters from IRT Model Analysis of simulated response data and simulated item parameters
accuracy = test.mean_index_diff(df3, df_1PL_1, 'Difficulty')

## result
# ex) 0.6
```

# 마무리(End)
위 패키지는 버전 0.1.1입니다. 점차 발전시켜 나가는 중에 있으며 많은 조언 부탁드립니다!

This package is currently at version 0.1.1. We are continuously working to improve it, and I would love to hear any feedback or suggestions you may have!