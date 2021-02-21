# Deep reinforcement learning for guidewire navigation in coronary artery phantom

강화학습을 이용한 심혈관 모형에서의 가이드와이어 네비게이션 데모 코드.
본 연구의 실제 학습 및 테스트는 심혈관 중재술 로봇과 2D, 3D 혈관 모형을 이용한 세팅이 필요하기 때문에 강화학습 모델과 학습 환경 코드를 공유하여도 재현이 어렵다.
이러한 제약사항을 해결하기 위해 심혈관 모형 이미지를 전처리한 이미지를 공유한다. 그리고 공유한 이미지로 구성한 더미 환경과 강화학습 알고리즘을 상호작용하는 코드를 구현하였다.
강화학습 모델은 [rl_algorithms](https://github.com/medipixel/rl_algorithms)에 구현된 모델을 사용하였다.

## Prerequisites
```
python > 3.6
torch > 1.6
```

## Installation
아래 명령어로 실행에 필요한 요소들을 설치한다:
```
make setup
```

## How to use


```
python run.py
```
