[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/FVjNDCrt)
# Title (Please modify the title)
## Team

| ![전은지](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이수형](https://avatars.githubusercontent.com/u/156163982?v=4) | ![서정민](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이지윤](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이승미](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [전은지](https://github.com/UpstageAILab)             |            [이수형](https://github.com/UpstageAILab)             |            [서정민](https://github.com/UpstageAILab)             |            [이지윤](https://github.com/UpstageAILab)             |            [이승미](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

* **전은지(팀장)**: 팀원 서포터, 초기EDA 및 모델선정, 평가지표 설정 및 결과분석
* **이승미(팀원)**: 데이터 분석 및 전처리, resnext50\_32x4d 모델 등을 실험  
* **서정민(팀원)**: efficientnet, resnext 등 여러 모델 튜닝 및 실험, 모델 앙상블
* **이지윤(팀원)**:  densenet 모델 튜닝 및 실험
* **이수형(팀원)**:  모델 설계 및 실험,  모델 튜닝(Wandb)

## 0. Overview
### Environment
* *NVIDIA GeForce RTX 3090 24GB*  
* *AMD Ryzen Threadripper 3960X 24-Core Processor*  
* *용량: 2 TB*

### Requirements
* *wandb*  
* *augraphy*  
* *albumentations*  
* *imgaug*  
* *torchvision*  
* *Augmentor*

### 활용 장비 및 재료(개발 환경, 협업 tool 등)
* **(컴퓨팅 환경)**   
  * *NVIDIA GeForce RTX 3090 24GB*  
  * *AMD Ryzen Threadripper 3960X 24-Core Processor*  
  * *용량: 2 TB*  
* **(협업 환경)** *Github, Notion. Google Drive*
* **(의사 소통)** *Slack, Zoom*

## 1. Competiton Info

### Overview

* **경진대회 주제:**  
  * 이번 대회는 computer vision domain에서 가장 중요한 태스크인 이미지 분류 대회로 주어진 이미지를 여러 클래스 중 하나로 분류하는 프로젝트입니다.
* **경진대회 구현 내용, 컨셉, 교육 내용과의 관련성 등**  
  * 그 중, 이번 대회는 문서 타입 분류를 위한 이미지 분류 대회입니다. 문서 데이터는 금융, 의료, 보험, 물류 등 산업 전반에 가장 많은 데이터이며, 많은 대기업에서 디지털 혁신을 위해 문서 유형을 분류하고자 합니다. 이러한 문서 타입 분류는 의료, 금융 등 여러 비즈니스 분야에서 대량의 문서 이미지를 식별하고 자동화 처리를 가능케 할 수 있습니다.

### Timeline

* 대회 시작 : 7/30 (화)  
* 데이터  분석 : 7/30-7/31(수)  
* 데이터 전터리 : 7/30\~8/2  
  * 어그멘테이션 (공부/코드완성)  
    * 1차(test와 유사한 결과물이 나오도록) : 7/31-8/1(목)  
    * 2차 : 8/2  
    * 3차 : 8/3  
    * 4차 : 8/5  
    * 5차 : 8/6  
    * 6차 : 8/9  
* 결과 분석 사이트 배포 : 8/3  
* 모델 선택 : 8/3-8/6  
  * 전체모델 약하게 성능 확인 : 7/31  
  * 모델 성능비교 : 8/3\~8/6  
* 성능 개선 : 8/7\~8/11

## 2. Components

### Directory

```
├──code  
│   ├── model\_train.ipynb  
│   ├── augmentation\_v1.ipynb  
│   ├── augmentation\_v2.ipynb  
│   ├── augmentation\_v3.ipynb  
│   ├── augmentation\_v4.ipynb  
│   ├── augmentation\_v5.ipynb  
│   ├── augmentation\_v6.ipynb  
│   ├── model\_Wandb.ipynb  
│   ├── model\_Analysis.ipynb  
│   ├──model\_OCR.ipynb  
│   └── model\_Score.ipynb  
└──Readme.md

```

## 3. Data descrption

### Dataset overview

* *train : 1570장의 이미지*  
* *train.csv : 1570개의 행. train 폴더에 존재하는 1570개의 이미지에 대한 정답 클래스 제공*  
  * *ID: 학습 샘플의 파일명*  
  * *target: 학습 샘플의 정답 클래스 번호*  
* *meta: 17개의 행*  
  * *target 17개의 클래스 번호에 대응하는 클래스 이름*  
* *test*  
  * *3140장의 이미지*  
* *sample\_submission.csv : 3140개의 행*  
  * *ID 평가 샘플의 파일명*  
  * *target 예측 결과가 입력될 컬럼 (기존에는 전부 0\)*

### EDA

###### Train EDA:
1. *이미지 사이즈에 대해서 이상치 이미지를 확인*  
2. *CV2 라이브러리를 사용한 노이즈 체크를 시도하였으나 실제 이미지와 분석 내용이 일치하지 않았음.*  
3. *Feature 확인 결과, 글씨를 잘 인지하고 있는 것을 확인.*  

###### Test EDA :
1. *모든 클래스에 대해서 이미지의 크기가 평균적으로 보았을 때 일관적이고 이상치 이미지가 없음.*  
2. *Feature 확인 결과, 글씨를 잘 인지하고 있는 것을 확인.*  

### Data Processing

* *1\. 오류 데이터 Searching*  
  * *총 7장의 Error Image Data 발견 및 수정 조치.*  
* *2\. Test Image 에 따른 Augmentation ( 순서 및 변형 종류 )*

#### Train Augmentation

- Ver1: 직접 함수 생성해서 변형 추가  
- Ver2: 직접 생성한 함수 \+ albumentation, agraphy 라이브러리를 사용해 무작위 확률로 순서대로 변형 적용  
- Ver3: ver2에서 노이즈. 오버레이, scale 함수 수정해 속도 최적화  

- Ver4: 10개까지의 증강에서는 증강 횟수를 단계별로 변형 적용

   (e.g. 0번째 증강 이미지는 변형 0개 적용 \-  원본, 1번째 증강 이미지는 변형 1개 적용 …)



- Ver5:  타겟하는 이미지의 사이즈로 변환하는 과정(baseline의 transform-Resize)에서 비율에 변형이 발생함. ver4에서 LongestMaxSize와 PadIfNeeded   
  (albumentation 라이브러리)를 사용한 이미지 크기 조정 추가  

- Ver6:  ver 5에서 변형 중 이미지가 잘리는 현상 해결

## 4. Modeling

### Model descrition

* *성능평가한 모델 :*   
  * Mobilenet (v2\_100, v3\_100)  
  * Efficientnet (b0, b2, b4)  
  * Resnext50\_32x4d  
  * Resnet (32, 50, 101, 152\)  
  * Swin Transformer tiny   
  * Convnext\_tiny  
  * Densenet (201)

* *Upsupervised learning으로 각 모델들의 17 클래스 분류로 특성차는 능력 확인 (timm의 모델 중 몇만 캡쳐)*  
    * *Resnext\_50\_30x4d*  
    * *Efficientnet\_b0,b2,b3*  
    * *Mobilev2\_100 (혼동이 많았던 class 7 의 정답률 준수)*  

    * *Timm 으로 모델들을 로드하여 unsupervised learning 으로 특징을 추출하고*  
    * *Kmeans 로 clustering 한 결과:*  
    * *Resnext50\_32x4d클러스터링알고리즘 kmeansNMI점수=0.5242*  
    * *convnext\_base클러스터링알고리즘kmeansNMI점수=0.5154*  
    * *convnext\_large클러스터링알고리즘kmeansNMI점수=0.5381*  
    * *swin\_tiny\_patch4\_window7\_224클러스터링알고리즘kmeansNMI점수=05710*  

    *결론: convnext,swin\_tiny 모델들의 경우 실제 학습 테스트에서 저조한 성적을 보임.*  
    *단일 모델에 대한 1 epoch 정답률 막대 그래프*  
    *Resnext50\_32x4d, efficientnet\_b0,b2,b3 모델들이 상위권에 위치함.*

### Modeling Process

* *Ensemble \- Soft Voting*  
  * *단일로 학습된 모델들 중에서 정답률이 상위권에 일치하며 오답률이 높은 특정 클래스에 대한 정답률이 높은 모델들을 모아 Ensemble 하여 Soft Voting 함.*

* *TTA(Test-Time Augmentation)*  
  * *모델의 에측 성능을 높이기 위한 방법으로 모델이 예측할 때 단일 이미지의 특정한 변형이나 상태에 과도하게 의존하는 것을 방지하고, 예측의 안정성과 일반화를 도모하는 앙상블 기법*

## 5. Tries

  1. *Denoising*  
     1. *변형 확인이 잘 안되어서 패스함*  
  2. *배경 삭제*  
     1. *rembg*  
  3. *Segmentation \+ masking*  
  4. *OCR*  
     1. *이미지 내 글자를 feature 삼아 class 를 특정할 수 있지 않을까하는 아이디어에서 착안. 적어도 가중치를 줄 수 있지 않을까 고려한 부분.*  

  5. *Attention*  
     1. *각 피처의 중요도를 독립적으로 조정하는 AttentionModule 생성*  
     2. *CBAM AttentionModule 생성*  
     3. *ECA AttentionModule 생성*

  *\- AttentionModule : F1-score \= 0.9102*

   *\- CBAM : CUDA Out of Memory 로 인한 진행 불가.*

   *\- ECA AttentionModule : F1-score= 0.9313*

  *\[ECA AttentionModule예시 이미지\]*

  6. *Wandb Sweeps*  
     1. *단일 모델의 하이퍼 파라미터를 탐색하기 위한 Wandb 기능*  
        *\[ECA Attention Module\]*  
        *\[CBAM Attention Module\]*  
  7. *Weight Hard voting*  
     1. *학습된 모델 중 3과 7 클래스에 좋은 성능을 보이는 모델들과 전반적으로 좋은 성능을 보인 모델을 로드*  
     2. *각각의 모델 예측 결과에서 모델별 특성에 따라 예측 결과에 가중치 적용 (egefficientnet\_b0 의경우 3 클래스에 대한 정답률이 높아 3 클래스에 대해 가중치 3.0적용)*

## 6. Result

### Leader Board

- 리더보드 [중간순위]

- 리더보드 [최종순위]
  - Rank 5
  - Score 0.9357

## 7. 자체 평가 의견

### *발전 가능성*

*1. Voting 에서 최적의 가중치 조합을 찾아낸다면 더욱 좋은 예측 성능을 기대할 수 있을 것입니다.*

*2. LayoutLMv3나 VILA같은 복잡한 모델을 사용하여 집중적으로 오류가 많았던 데이터를 맞추게 한다면 앙상블 했을 때 더 좋은 결과가 있을 것입니다.*

*3. 아직 충분히 좋은 모델을 찾지못해 의미없는 아이디어였지만 만약 특정레이블, 특히 3,7의 구분을 잘하는 모델이 있다면 모델별로 weight를 주는것 뿐만 아니라 모델의 클래스 예측에 따라 weight를 주어 판단했다면 좋았을 것입니다.*

*4. OCR이 가능했다면 몇몇 파일들을 추측하기 더 좋았을 것입니다.*

*5. Deskew로 평행이미지로 조정하여 ocr하면 더 잘 되었을지도 모릅니다.*

### *잘한점*

*1. Data augmentation 과정에서 test 데이터 분석 후 세밀하게 변형을 적용했습니다.*

*2. 저번 프로젝트 과정에서 아쉬웠던 점인 과정 중간의 구체적인 내용 기록이 잘 진행되었습니다.*

*3. 여러 시도를 해보았습니다.*

### *아쉬운점*

*1. custom model을 구현해보지 못하여서 아쉬웠습니디.*

*2. 틀린 데이터에 대한 구체적인 분석이 부족했던 것 같습니다.*

*3. attention 부분을 cnn을 결합하는 것을 일찍 시도해봤으면 좋은 결과를 얻을 수 있었을 것 같습니다.*

*4. wandb를 통해 적절한 하이퍼파라미터를 찾을 수 있었을 것 같습니다.*

*5. 혼동 레이블에 개선을 하지 못한게 너무 아쉽습니다.*

*6. 학습에 마이너스인 불량 데이터를 확실하게 걸러내지 못한게 아쉽습니다.*

### *배운점*

*1. 문서라는 이미지에 집중해서 augmentation하는 라이브러리를 써볼 수 있었습니다.*

*2. 코드를 작성할 때 OOM이 안나게 관리하는 법을 많이 생각해 보았습니다.*

*3. 더 이상 방법이 없나 싶을 때 진행한 멘토링에서 Wandb sweeps, Attention 등의 방법을 배울 수 있었습니다.*

*4. OCR을 간단하게나마 사용해볼 수 있었습니다.*

*5. Timm 으로 로드한 모델을 커스텀 할 수 있게 되었습니다.*

### Presentation

* *https://www.canva.com/design/DAGNixyTFx4/FjSJIpYt9qc57eUvg1anJg/edit?utm\_content=DAGNixyTFx4\&utm\_campaign=designshare\&utm\_medium=link2\&utm\_source=sharebutton*

## etc

### Meeting Log

* *Insert your meeting log link like Notion or Google Docs*  
* *Mon\~Fri 10:00\~13:00, 14:00\~19:00*  
* *2024-08-10 21:00\~24:00*  
* *2024-08-11 13:00\~14:00 & 18:00\~24:00*

### Reference

* *Insert related reference*  
* *Squeeze-and-Excitation Networks (SENet)*  
  * *채널 어텐션 메커니즘에 집중한 논문*  
  * *https://arxiv.org/abs/1709.01507*  
* *ECA*  
  * [*https://arxiv.org/abs/1910.03151*](https://arxiv.org/abs/1910.03151)  
* *CBAM*  
  * [*https://arxiv.org/abs/1807.06521*](https://arxiv.org/abs/1807.06521)  
* *Ref Projects*  
  * [*https://break210b.tistory.com/69*](https://break210b.tistory.com/69)  
  * [*https://sixth-drum-9ac.notion.site/Document-Type-Classification-b01886bae17c4dd9b2d3244429f56fee*](https://sixth-drum-9ac.notion.site/Document-Type-Classification-b01886bae17c4dd9b2d3244429f56fee)  
  * [*https://velog.io/@sohyeonos248/Upstage-AILab-1기-CV-경진대회-회고*](https://velog.io/@sohyeonos248/Upstage-AILab-1%EA%B8%B0-CV-%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C-%ED%9A%8C%EA%B3%A0)  
  * [*https://github.com/kangggggggg/Document-Type-Classification-upstage-cv/tree/main*](https://github.com/kangggggggg/Document-Type-Classification-upstage-cv/tree/main)  
  * [*https://www.notion.so/4-3cf0b6ee2e7b4d5aaf6c0dc76d9c65a5*](https://www.notion.so/4-3cf0b6ee2e7b4d5aaf6c0dc76d9c65a5)  
