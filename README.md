# Continual Learning Implementation
### paper iCaRL: Incremental Classifier and Representation Learning

## Experimental Setup

### Model Architecture

- **ResNet-32**: 
  - 기본 특징 추출기로 수정된 ResNet-32를 사용합니다.
  - `layer1`, `layer2`, `layer3`의 세 개의 합성곱 층을 포함하고 있으며, 각 층은 배치 정규화(Batch Normalization)와 ReLU 활성화 함수를 갖춘 `BasicBlock`으로 구성됩니다.

- **IncrementalNet**: 
  - 이 네트워크는 학습된 전체 클래스 수에 따라 동적으로 조정되는 완전 연결 계층을 포함합니다.
  - 새로운 클래스가 추가될 때마다 출력 계층이 자동으로 업데이트됩니다.

### Continual Learning Framework

- **iCaRL (Incremental Classifier and Representation Learning)**:
  - iCaRL은 리허설 메모리(Exemplar Set)와 지식 증류(Knowledge Distillation)를 통해 새로운 클래스가 추가될 때 발생할 수 있는 망각 문제를 완화합니다.
  - 이전에 학습한 클래스를 대표하는 샘플을 메모리에 저장하여 새 클래스 학습 중에도 이전 클래스를 반복 학습합니다.

### Datasets

- **CIFAR-10** 데이터셋을 주로 사용하였으며, 코드 구조는 **CIFAR-100**, **MNIST**, **ImageNet** 등 다양한 데이터셋을 지원합니다.
- 각 데이터셋별 전처리는 모델의 입력 크기에 맞춰 조정되었습니다. 예를 들어, **MNIST**는 32x32로 리사이즈되고, **CIFAR-10**과 **CIFAR-100**은 RGB 표준화가 적용되었습니다.

## Training Setup

### Hyperparameters

- **초기 학습 (init_epoch)**: 20 에포크 동안 진행됩니다.
- **초기 학습률 (init_lr)**: 0.1로 시작하며, 이후 학습률(lrate)도 0.1로 설정되었습니다.
- **학습률 감소**: 일정 에포크 이후 단계적으로 적용되며, `init_milestones`와 `milestones`는 각각 `[60, 120, 170]`, `[80, 120]`으로 설정되었습니다.
- **Batch size**: 실험마다 다양하게 설정되었습니다 (32, 64, 128, 256).
- **num_workers**: 데이터 로딩 속도를 최적화하기 위해 `num_workers=4`로 설정되었습니다.

### Knowledge Distillation

- **지식 증류 (Knowledge Distillation)**: 현재 모델과 이전 모델의 출력을 비교하여 이전 클래스에 대한 지식을 유지하도록 돕습니다.
- **온도 매개변수 (T)**: 지식 증류의 온도는 2, 5, 10으로 실험되었습니다. 온도가 높을수록 이전 클래스에 대한 정보가 부드럽게 유지되지만, 너무 높으면 모델 성능이 떨어질 수 있습니다.

### Rehearsal Memory

- **메모리 예산 (Memory Budget)**: 클래스당 특정 수의 샘플을 저장하며, 메모리 예산은 500, 1000, 1500, 2000으로 설정하여 실험하였습니다.
- 각 클래스에 대해 저장할 샘플 수는 메모리 예산과 총 클래스 수에 따라 계산됩니다. 이를 통해 새 클래스를 학습할 때도 이전 클래스 샘플을 사용할 수 있어 망각을 줄이는 데 도움을 줍니다.

## Evaluation

- 각 증분 학습 단계 후 테스트 데이터셋에서 **평균 정확도 (average accuracy)**를 측정하여 성능을 평가합니다.
- 모든 실험 결과는 메모리 예산, 예제 크기, 온도, 배치 크기 등 다양한 매개변수에 따른 테스트 정확도와 평균 정확도로 기록하여 성능 변화를 분석합니다.
