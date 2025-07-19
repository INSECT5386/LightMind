import numpy as np
import pickle

with open("my_model_vocab.pkl", "rb") as f:
    token2idx, idx2token = pickle.load(f)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=axis, keepdims=True)

def glu_forward_np(x, weights):
    # x: (batch_size, seq_len, embed_dim) - 넘파이 배열
    # weights: dict, 저장된 npz 가중치들

    # Q_proj = x @ Wq_proj + bq_proj
    h_proj = np.matmul(x, weights["model_Wq_proj"]) + weights["model_bq_proj"]  # (B, T, hidden_dim)

    Q = np.matmul(h_proj, weights["model_Wq"])  # (B, T, hidden_dim)
    K = np.matmul(x, weights["model_Wk"])       # (B, T, hidden_dim)
    V = np.matmul(x, weights["model_Wv"])       # (B, T, hidden_dim)

    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0,2,1)) / np.sqrt(d_k)  # (B, T, T)
    weights_attn = softmax(scores, axis=-1)                   # (B, T, T)

    attn_out = np.matmul(weights_attn, V)                     # (B, T, hidden_dim)

    h_in = np.matmul(attn_out, weights["model_W1"]) + weights["model_b1"]  # (B, T, hidden_dim*2)
    x1, x2 = np.split(h_in, 2, axis=-1)

    h = x1 * gelu(x2)  # (B, T, hidden_dim)

    # 여기서 LayerNorm은 TensorFlow Layer라 넘파이로 완벽히 동일 구현 어려운데,
    # 간단히 평균0, 분산1 정규화 정도로 대체 가능
    mean = h.mean(axis=-1, keepdims=True)
    std = h.std(axis=-1, keepdims=True) + 1e-5
    h_norm = (h - mean) / std

    combined = attn_out + h_norm  # (B, T, hidden_dim)

    out_seq = np.matmul(combined, weights["model_W2"]) + weights["model_b2"]  # (B, T, output_dim)

    # 마지막 시퀀스 타임스텝만 출력
    out = out_seq[:, -1, :]  # (B, output_dim)
    return out

# 예시

npz_path = "my_model_weights.npz"
weights = dict(np.load(npz_path))

# 임베딩 가중치 로드
embedding_matrix = weights["embedding"]  # (vocab_size, embed_dim)

# 입력 예시 : "안녕"이라는 단어 인덱스  [예: 1234] 1개 배치, 시퀀스 길이 1
input_indices = np.array([[1234]])

# vocab 불러온 상태 가정
idx = token2idx.get("안녕", 0)
input_indices = np.array([[idx]])

x_embedded = embedding_matrix[input_indices]  # shape: (1,1,embed_dim)


def generate_sequence_np(weights, token2idx, idx2token, prompt, max_tokens=50, top_p=0.9, temperature=1.0, min_length=5):
    if isinstance(prompt, str):
        prompt_tokens = prompt.split()
    else:
        prompt_tokens = prompt
    
    generated = []
    token_counts = {}
    
    for step in range(max_tokens):
        context = prompt_tokens + generated
        
        # 임베딩 인덱스 변환
        try:
            input_indices = np.array([[token2idx[tok] for tok in context]])
        except KeyError as e:
            print(f"[Warning] Unknown token: {e}")
            break
        
        # 임베딩 lookup
        x_embedded = weights["embedding"][input_indices]  # (1, seq_len, embed_dim)
        
        # forward numpy
        logits = glu_forward_np(x_embedded, weights)  # (1, vocab_size)
        logits = logits[0]  # (vocab_size,)
        
        # 온도 조절
        logits = logits / temperature
        
        # softmax
        probs = softmax(logits)
        
        # 토큰 중복 페널티 (optional)
        for tok, count in token_counts.items():
            idx = token2idx[tok]
            probs[idx] /= (count + 1) ** 2
        
        probs = probs / probs.sum()
        
        # top-p 샘플링 (간단 구현)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cumulative_probs, top_p) + 1
        top_indices = sorted_indices[:cutoff]
        top_probs = sorted_probs[:cutoff]
        top_probs = top_probs / top_probs.sum()
        
        next_token_idx = np.random.choice(top_indices, p=top_probs)
        next_token = idx2token[next_token_idx]
        
        if next_token == "<EOS>" and step < min_length:
            probs[next_token_idx] = 0
            probs = probs / probs.sum()
            continue
        
        if next_token == "<EOS>":
            break
        
        generated.append(next_token)
        token_counts[next_token] = token_counts.get(next_token, 0) + 1
        
    return generated

generated = generate_sequence_np(weights, token2idx, idx2token, "안녕", max_tokens=50, top_p=0.9, temperature=0.95)
print("생성:", " ".join(generated))
