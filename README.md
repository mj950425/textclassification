# textclassification
---
### 프로젝트 목적
![image](https://user-images.githubusercontent.com/52944973/112805465-91d17800-90b0-11eb-9049-8a1d46d9e791.png)

인공지능(AI) 챗봇 ‘이루다’가 동성애를 혐오하게 되었다.

원인은 비 윤리적인 데이터들을 필터링하지 못한채로 학습했기 때문이다.

따라서 이러한 데이터들을 필터링 할 수 있는 모델을 설계하고자 했다.

👀 나아가 완벽한 페르소나를 만들기 위해서 원하는 챗봇의 성격에 맞는 데이터들만 가져올 수 있도록 하는것이 목표이다. 

욕설, 성적, 공격적인 텍스트 데이터를 분류하기 위해서 multi binaray classification이 필요하다.

간단하게 아래 3가지 방법이 존재한다.

*   Binary Relevance : 각 특징들끼리 관계가 없다고 가정하고 독립적으로 binary classification

*   Classifier Chains : 하나씩 이어서 classification

*   Label Powerset : 2^feature로 예측하기


Binary Relevance 와 Classifier Chains를 통해서 

toxic
severe_toxic
obscene
threat
insult
identity_hate

6가지 multiclass classification을 수행

https://github.com/mj950425/kaggle_transcription/blob/main/toxic_%ED%95%84%EC%82%AC.ipynb

word2vec으로 임베딩을 뽑은 뒤 classification 진행 -> 의미있는 성능 get

하지만 종속인 칼럼들이 아니라 같은 카테고리에 속하는 칼럼이라 생각이 들어 굳이 multiclass classification을 할 필요가 있을까?

Toxic or non Toxic으로 나누는 다른 kaggle competition을 찾아봄

여기선 버트를 활용
https://github.com/mj950425/kaggle_transcription/blob/main/toxic_classification_Bert.ipynb

한국어 기반으로 된 Kcbert에 Koco에서 fine tune한 버트 모델이 존재
https://github.com/mj950425/textclassification/blob/main/koco.ipynb
(성능이 아주 좋다)

하지만 슈ㅣ발을 욕설로 잡아 주지 못하는 단점이 존재했는데, 중요한것은 데이터셋에서 욕설을 통과시키지 않는것이니 다소 손해가 있더라도 rule based로 한번 걸러주고 모델을 돌려주는것을 구상 중

### 데이터셋 만들기

---

![image](https://user-images.githubusercontent.com/52944973/113390906-98703000-93cd-11eb-8c79-aaa0ce89c194.png)
(출처 koco)

(출처 aihub)

---



---

---
### 관련 자료
https://github.com/mj950425/kaggle_transcription/blob/main/toxic_%ED%95%84%EC%82%AC.ipynb
https://github.com/inmoonlight/koco


---
