### 평가 산식

[
\text{Score}
============

\sum_{s} w_s \cdot
\left(
\frac{1}{|I_s|}
\sum_{i \in I_s}
\left(
\frac{1}{T_i}
\sum_{t=1}^{T_i}
\frac{2 \lvert A_{t,i} - P_{t,i} \rvert}
{\lvert A_{t,i} \rvert + \lvert P_{t,i} \rvert}
\right)
\right)
]

---

### 기호 설명

* ( s ) : 식음업장명
* ( w_s ) : 식음업장 ( s )의 가중치 (비공개)
* ( I_s ) : 식음업장 ( s )에 속한 품목 컬럼 집합
* ( T_i ) : 품목 ( i )에서 유효한 날짜 수
  [
  (A_{t,i} \neq 0)
  ]
* ( A_{t,i} ) : 날짜 ( t ), 품목 ( i )의 실제값
* ( P_{t,i} ) : 날짜 ( t ), 품목 ( i )의 예측값

---

### 추가 조건

* **실제 매출 수량이 0인 경우**
  → 해당 항목은 평가 산식 계산에서 **제외**됨

---
