import numpy as np
import csv
import tensorflow as tf
import tensorflow.experimental.numpy as tfnp
from tqdm import tqdm
import numpy as np
import pickle

tfnp.experimental_enable_numpy_behavior()

def softmax(x):
    x = x - tf.reduce_max(x, axis=-1, keepdims=True)
    e_x = tf.exp(x)
    return e_x / tf.reduce_sum(e_x, axis=-1, keepdims=True)

def top_p_sample_tf(probs, p=0.9):
    sorted_indices = tfnp.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    cumulative = tfnp.cumsum(sorted_probs)

    cumulative_np = cumulative.numpy()  # 텐서 → numpy 배열 변환
    cutoff = np.searchsorted(cumulative_np, p) + 1  # 여기 numpy 함수 사용

    top_indices = sorted_indices[:cutoff]
    top_probs = probs[top_indices]
    top_probs /= tf.reduce_sum(top_probs)

    top_indices_np = top_indices.numpy()
    top_probs_np = top_probs.numpy()
    sampled = np.random.choice(top_indices_np, p=top_probs_np)

    return tfnp.array(sampled)

def scaled_dot_product_attention(Q, K, V, causal_mask=True):
    d_k = tf.cast(tf.shape(Q)[-1], tf.float32)
    scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)

    if causal_mask:
        T = tf.shape(scores)[1]
        causal = tf.linalg.band_part(tf.ones((T, T)), -1, 0)  # 하삼각 causal mask
        causal = tf.reshape(causal, (1, T, T))  # [1, T, T] 브로드캐스트용
        scores = tf.where(causal == 0, tf.constant(-1e9, dtype=scores.dtype), scores)

    scores = scores - tf.reduce_max(scores, axis=-1, keepdims=True)
    weights = tf.exp(scores)
    weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
    output = tf.matmul(weights, V)
    return output, weights

def scaled_dot_product_attention_backward(Q, K, V, weights, grad_output, causal_mask=True):
    d_k = tf.cast(tf.shape(Q)[-1], tf.float32)

    grad_weights = tf.matmul(grad_output, V, transpose_b=True)
    grad_V = tf.matmul(weights, grad_output, transpose_a=True)

    s = weights
    grad_scores = s * (grad_weights - tf.reduce_sum(grad_weights * s, axis=-1, keepdims=True))

    if causal_mask:
        T = tf.shape(grad_scores)[1]
        causal = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        causal = tf.reshape(causal, (1, T, T))
        grad_scores = tf.where(causal == 0, 0.0, grad_scores)

    grad_Q = tf.matmul(grad_scores, K) / tf.sqrt(d_k)
    grad_K = tf.matmul(grad_scores, Q, transpose_a=True) / tf.sqrt(d_k)

    return grad_Q, grad_K, grad_V


class AdamOptimizerTF:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [tfnp.zeros_like(p) for p in params]
        self.v = [tfnp.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * (tfnp.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))
        for i in range(len(self.params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            update = lr_t * m_hat / (tfnp.sqrt(v_hat) + self.eps)

            # params[i]가 tf.Variable이라면 assign_sub 사용
            if isinstance(self.params[i], tf.Variable):
                self.params[i].assign_sub(update)
            else:
                # 그냥 ndarray라면 직접 빼기
                self.params[i] = self.params[i] - update

# --- Xavier 초기화 ---
def xavier_init(fan_in, fan_out):
    limit = tfnp.sqrt(6.0 / (fan_in + fan_out))
    return tfnp.random.uniform(-limit, limit, size=(fan_in, fan_out))

class GLULayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, hidden_dim, output_dim, seed=0):
        super().__init__()
        tf.random.set_seed(seed)
        
        self.W1 = self.add_weight(shape=(hidden_dim, hidden_dim * 2), initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.b1 = self.add_weight(shape=(hidden_dim * 2,), initializer="zeros", trainable=True)
        
        self.W2 = self.add_weight(shape=(hidden_dim, output_dim), initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.b2 = self.add_weight(shape=(output_dim,), initializer="zeros", trainable=True)
        
        self.Wq_proj = self.add_weight(shape=(input_dim, hidden_dim), initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.bq_proj = self.add_weight(shape=(hidden_dim,), initializer="zeros", trainable=True)
        
        self.Wq = self.add_weight(shape=(hidden_dim, hidden_dim), initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Wk = self.add_weight(shape=(input_dim, hidden_dim), initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        self.Wv = self.add_weight(shape=(input_dim, hidden_dim), initializer=tf.keras.initializers.GlorotUniform(), trainable=True)
        
        self.layernorm = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

    def forward(self, x):
        h_proj = tf.matmul(x, self.Wq_proj) + self.bq_proj
        Q = tf.matmul(h_proj, self.Wq)
        K = tf.matmul(x, self.Wk)
        V = tf.matmul(x, self.Wv)

        dk = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(dk)
        weights = tf.nn.softmax(scores, axis=-1)
        attn_out = tf.matmul(weights, V)

        h_in = tf.matmul(attn_out, self.W1) + self.b1
        x1, x2 = tf.split(h_in, 2, axis=-1)

        h = x1 * tf.nn.gelu(x2)
        h = self.layernorm(h)
        combined = attn_out + h

        out_seq = tf.matmul(combined, self.W2) + self.b2
        out = out_seq[:, -1, :]
        return out

    def call(self, x):
        return self.forward(x)


def sparse_cross_entropy_loss(probs, true_indices):
    batch = probs.shape[0]
    clipped_probs = tfnp.clip(probs, 1e-9, 1.0)
    log_probs = tfnp.log(clipped_probs)
    loss = -tfnp.sum(log_probs[tfnp.arange(batch), true_indices]) / batch
    return loss

def grad_sparse_cross_entropy(probs, true_indices):
    grad = tfnp.array(probs)  # 복사
    grad = tfnp.asarray(grad)  # 확실히 tfnp 배열로
    batch_indices = tfnp.arange(probs.shape[0])
    # 인덱스별 true token 위치에 대해 1 빼기
    grad = tfnp.array(grad)
    grad = grad.at[batch_indices, true_indices].add(-1)
    grad /= probs.shape[0]
    return grad


# 2. build_vocab 내부 idx2token 뒤집기 수정
def build_vocab(pairs):
    tokens = set()
    for inp, out in pairs:
        tokens.update(inp.split())
        tokens.update(out.split())
    tokens.add('<EOS>')
    token2idx = {tok: idx for idx, tok in enumerate(sorted(tokens))}
    idx2token = {v: k for k, v in token2idx.items()}  # 이 부분 꼭 이렇게 바꾸기
    return token2idx, idx2token

def save_vocab(token2idx, idx2token, filepath):
    """어휘사전 저장"""
    with open(filepath, "wb") as f:
        pickle.dump((token2idx, idx2token), f)


def load_vocab(path):
    """pickle로 저장된 어휘사전 로드"""
    with open(path, 'rb') as f:
        vocab_data = pickle.load(f)
    return vocab_data['token2idx'], vocab_data['idx2token']



def tokenize_and_encode(pairs, token2idx):
    contexts_idx = []
    next_tokens_idx = []

    for inp_text, out_text in pairs:
        inp_ids = [token2idx[tok] for tok in inp_text.split()]
        out_ids = [token2idx[tok] for tok in out_text.split()] + [token2idx['<EOS>']]

        seq = inp_ids.copy()
        for out_id in out_ids:
            contexts_idx.append(seq.copy())
            next_tokens_idx.append(out_id)
            seq.append(out_id)

    max_len = max(len(seq) for seq in contexts_idx)
    padded = np.zeros((len(contexts_idx), max_len), dtype=np.int32)
    for i, seq in enumerate(contexts_idx):
        padded[i, :len(seq)] = seq

    return tfnp.array(padded), tfnp.array(next_tokens_idx, dtype=tfnp.int32)


def generate_sequence(model, token2idx, idx2token, prompt, max_tokens=100, top_p=0.75, min_length=5, temperature=0.7):
    if isinstance(prompt, str):
        prompt_tokens = prompt.split()
    else:
        prompt_tokens = prompt

    generated = []
    token_counts = {}

    for step in range(max_tokens):
        context = prompt_tokens + generated
        try:
            X_indices = tfnp.array([[token2idx[tok] for tok in context]])
        except KeyError:
            print(f"[Warning] Unknown token in prompt/context: {context}")
            break

        probs = model.predict(X_indices)  # probs는 tf.Tensor 혹은 tfnp array 예상
        probs = tfnp.clip(probs, 1e-9, None)

        probs = probs ** (1.0 / temperature)
        probs /= tf.reduce_sum(probs)


        for token, count in token_counts.items():
            idx = token2idx[token]
            probs = probs.numpy()  # Tensor → numpy
            probs[idx] /= (count + 1) ** 2
            probs = tfnp.array(probs)  # 다시 tfnp array 변환

        probs /= tf.reduce_sum(probs)


        next_idx = top_p_sample_tf(probs, p=top_p)
        next_token = idx2token[int(next_idx.numpy())]

        if next_token == '<EOS>' and step < min_length:
            probs = probs.numpy()
            probs[int(next_idx.numpy())] = 0
            probs = tfnp.array(probs)
            probs /= tf.reduce_sum(probs)

            next_idx = top_p_sample_tf(probs, p=top_p)
            next_token = idx2token[int(next_idx.numpy())]

        if next_token == '<EOS>':
            break

        generated.append(next_token)
        token_counts[next_token] = token_counts.get(next_token, 0) + 1

    return generated


class Model(tf.keras.Model):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.model = GLULayer(embed_dim, hidden_dim, output_dim)

    def call(self, inputs):
        x = self.embedding(inputs)
        logits = self.model(x)
        return logits


import csv

# --- 데이터 관련 함수 ---
def load_data(csv_path):
    pairs = []
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            inp = row['input_text'].strip()
            out = row['output_text'].strip()
            pairs.append((inp, out))
    return pairs


def save_model_and_vocab_npz(model, token2idx, idx2token, path_prefix):
    """모델 전체를 npz로 저장 + vocab pickle로 저장"""
    weights = {}
    # TensorFlow Embedding layer weight는 get_weights()로 가져오기
    weights["embedding"] = model.embedding.get_weights()[0]

    # GLULayer 가중치 (CuPy -> numpy 변환이 아닌 tf.Variable이니까 numpy로 변환)
    layer = model.model
    weights[f"model_W1"] = layer.W1.numpy()
    weights[f"model_b1"] = layer.b1.numpy()
    weights[f"model_W2"] = layer.W2.numpy()
    weights[f"model_b2"] = layer.b2.numpy()
    weights[f"model_Wq_proj"] = layer.Wq_proj.numpy()
    weights[f"model_bq_proj"] = layer.bq_proj.numpy()
    weights[f"model_Wq"] = layer.Wq.numpy()
    weights[f"model_Wk"] = layer.Wk.numpy()
    weights[f"model_Wv"] = layer.Wv.numpy()

    np.savez(f"{path_prefix}_weights.npz", **weights)
    save_vocab(token2idx, idx2token, f"{path_prefix}_vocab.pkl")
    print(f"✅ 모델(.npz)과 어휘사전(.pkl)이 저장되었습니다!")

import tensorflow as tf

MAX_SEQ_LEN = 128  # 시퀀스 최대 길이

# 상삼각 행렬(대각선 제외) 1, 나머지 0
causal_mask = 1 - tf.linalg.band_part(tf.ones((1, MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=tf.bool), -1, 0)
def main():
    csv_path = "MLdata.csv"
    pairs = load_data(csv_path)
    
    token2idx, idx2token = build_vocab(pairs)
    X, Y = tokenize_and_encode(pairs, token2idx)

    vocab_size = len(token2idx)
    embed_dim = 128
    hidden_dim = 128
    output_dim = vocab_size

    model = Model(vocab_size, embed_dim, hidden_dim, output_dim)
    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

    model.fit(X, Y, epochs=3, batch_size=32)

    save_model_and_vocab_npz(model, token2idx, idx2token, "my_model")

    prompt = "이"
    generated = generate_sequence(model, token2idx, idx2token, prompt)
    print("생성:", " ".join(generated))


if __name__ == "__main__":
    main()
