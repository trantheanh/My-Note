# Enriching Word Vectors with Subword Information

## 1. Morphological word representation: (Biểu diễn hình thái học của từ)

English: love, lover, loving

Vietnamese:  mảnh_mai, mảnh_dẻ, mảnh_khảnh

=> Nếu không có biểu diễn hình thái học của từ vựng, những từ dù có chung hình thái cũng sẽ không liên quan gì tới nhau

=> Giải pháp: 

- Character Level features
- Subword level features

## 2. Model:

- General model:

  - **Skip-gram**: (Given word, predict context)

    - Corpus = $[w_1, \ldots, w_T] $
    - context = $C_t$ (set of indices of words surrounding word $w_t$)
    - objective: 
      - maximum  
        $$
      \sum^T_{t=1}\sum_{c\in C_t}log \space p(w_c \mid w_t)
        $$
      
    - trong đó: 
        $$
        p(w_c \mid w_t) = \frac{e^{s(w_t, w_c)}}{\sum^W_{j=1}e^{s(w_t, j)}}
        $$
        (s : **score function** maps **pair** (word, context))
      
        
    
    Tuy nhiên model của tác giả chỉ dự đoán 1 context word $w_c$ .
    
    
    
  - **Negative Sampling**: 

  Vì bài toán dự đoán context word (tất cả context word) có thể được thay thế bằng một tập các bài toán phân loại nhị phân (dự đoán 1 context word). Khi đó mục tiêu của bài toán trở thành dự đoán sự có măt hay không của context word. 

  - các cặp ($w_t, w_c$)  trong đó $w_c \in C_t$ , $w_t$ là word thứ $t$ trong corpus
    - các cặp ($w_t, n$) trong đó $n \in N_{t,c}$, với $N_{t,c}$  là tập những word không thuộc context của $w_t$ được sample từ từ điển (vocab)

  

  Cross-entropy giữa phân phối $p$ và phân phối $q$: $plog(q)$

  => Binary logistic loss (negative cross-entropy): $-y\space log\space p(x) - (1-y)\space log \space (1-p(x))$ 

  => Tương ứng với dữ liệu: 

  - Positive pair $(y=1)$ : 
      $$
      -1 \space log \frac{1}{1 + e^{-s(w_t, w_c)}} = -(log(1) - log(1 + e^{-s(w_t, w_c)})) = log(1 + e^{-s(w_t, w_c)})
      $$
      
  - Negative pairs $(y = 0)$ : 
    $$
      -1 \space log(1 - \frac{1}{1 + e^{-s(w_t, n)}}) = log(1 + e^{s(w_t, n)})
    $$
    
    
    Nhưng vì 1 cặp positive có tới $N_{t, c}$ cặp negative nên loss trên negative pair trở thành: 

    
    $$
      \sum_{n \in N_{t,c}} log(1 + e^{s(w_t, n)})
    $$
    

  => **negative log-likelihood**:    
  $$
    log(1 + e^{-s(w_t, w_c)}) + \sum_{n \in N_{t,c}} log(1 + e^{s(w_t, n)})
  $$
    Ký hiệu: $ℓ: x\to log(1+e^{-x})$ 

  => **Objective function**: 
  $$
    \sum^{T}_{t=1} \left[ \sum_{c \in C_t}ℓ(s(w_t, w_c)) + \sum_{n \in N_{t,c}} ℓ(-s(w_t, n))   \right]
  $$

  - Hàm tính score $s$ ở đây là tích vô hướng. Trong đó word tại vị trí $t$ trong corpus $w_t$ được biểu diễn bằng vector input (embedding của original word) còn $w_c$ (context của $w_t$) được biểu diễn bởi vector output (embedding của context word). Để hiểu hơn về embedding của original word (input embedding) và embedding của context word (output embedding) ta nên xem paper của 

    [word2vec]: https://arxiv.org/pdf/1301.3781.pdf

    

  - ádasd

-  Subword model:

## 3. Experiment setup:

- Baseline:
- Optimization:
- Implementation details:
- Datasets:

## 4. Results:

- Human similarity judgement:
- Word analogy task:
- Comparison with morphological representation:
- Effect of the size of the training data:
- Effect of the size of n-gram:
- Language modeling:

## 5. Qualitative Analysis:

- Nearest Neighbors
- Character n-grams and morphemes
- Word similarity for OOV words