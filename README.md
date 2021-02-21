# Deep reinforcement learning for guidewire navigation in coronary artery phantom

강화학습을 이용한 심혈관 모형에서의 가이드와이어 네비게이션 데모 코드.
본 연구의 실제 학습 및 테스트는 심혈관 중재술 로봇과 2D, 3D 혈관 모형을 이용한 세팅이 필요하기 때문에 강화학습 모델과 학습 환경 코드를 공유하여도 재현이 어렵다.
이러한 제약사항을 해결하기 위해 심혈관 모형 이미지를 전처리한 이미지를 공유한다. 그리고 공유한 이미지로 구성한 더미 환경과 강화학습 알고리즘을 상호작용하는 코드를 구현하였다.
강화학습 모델은 [rl_algorithms](https://github.com/medipixel/rl_algorithms)에 구현된 모델을 사용하였다.

## Prerequisites
이 코드를 테스트하기 위해서는 python 3.6+ 이 필요하다.
아래 명령어로 Anaconda virtual environment를 사용하는 것을 추천한다:
```
$ conda create -n rl-pci-demo python=3.6.9
$ conda activate rl-pci-demo
```


## Installation
[rl_algorithms](https://github.com/medipixel/rl_algorithms) 패키지와 dependancy를 설치합니다.
아래 명령어를 사용합니다:
```
make setup
```

## How to use
미리 학습된 강화학습 모델과 더미 환경을 생성한 후 상호작용합니다. 특정 state에서 학습된 강화학습 모델이 어떤 action을 만들어내는지 확인합니다.

```
python run.py
```
