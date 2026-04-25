E = 2.718281828459045

word_to_idx = {"무논리": 0, "논리": 1}
idx_to_word = {0: "무논리", 1: "논리"}

def exp(x):
    if x > 40:
        x = 40
    if x < -40:
        x = -40
    return E ** x

def silu(x):
    return x / (1.0 + exp(-x))

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def add(a, b):
    return [x + y for x, y in zip(a, b)]

def matmul_vec(v, w):
    return [sum(v[i] * w[i][j] for i in range(len(v))) for j in range(len(w[0]))]

def linear_seq(x, w):
    return [matmul_vec(v, w) for v in x]

def softmax(v):
    m = max(v)
    e = [exp(x - m) for x in v]
    s = sum(e)
    return [x / s for x in e]

def layer_norm(v, eps=1e-6):
    m = sum(v) / len(v)
    var = sum((x - m) * (x - m) for x in v) / len(v)
    inv = (var + eps) ** -0.5
    return [(x - m) * inv for x in v]

def layer_norm_seq(x):
    return [layer_norm(v) for v in x]

def identity(n, scale=1.0):
    return [[scale if i == j else 0.0 for j in range(n)] for i in range(n)]

def attention(x, wq, wk, wv, wo, heads):
    q = linear_seq(x, wq)
    k = linear_seq(x, wk)
    v = linear_seq(x, wv)
    n = len(x)
    d = len(x[0])
    hd = d // heads
    out = []

    for i in range(n):
        merged = []
        for h in range(heads):
            s = h * hd
            e = s + hd
            scores = [dot(q[i][s:e], k[j][s:e]) * (hd ** -0.5) for j in range(n)]
            weights = softmax(scores)
            head = [sum(weights[j] * v[j][s + t] for j in range(n)) for t in range(hd)]
            merged.extend(head)
        out.append(matmul_vec(merged, wo))

    return out

def ffn(x, w1, w2):
    return [matmul_vec([silu(a) for a in matmul_vec(v, w1)], w2) for v in x]

def block(x, wq, wk, wv, wo, w1, w2, heads):
    a = attention(x, wq, wk, wv, wo, heads)
    x = layer_norm_seq([add(x[i], a[i]) for i in range(len(x))])
    f = ffn(x, w1, w2)
    x = layer_norm_seq([add(x[i], f[i]) for i in range(len(x))])
    return x

def positional(i, d):
    return [((i + 1) * (j + 3) % 11 - 5) / 200.0 for j in range(d)]

def embed(token, d):
    base = [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.0625, -0.0625, 0.03125, -0.03125]
    if word_to_idx[token] == 0:
        base = [-x for x in base]
    return base[:d]

def predict(text):
    d_model = 16
    heads = 4
    depth = 512

    wq = identity(d_model, 1.0)
    wk = identity(d_model, 1.0)
    wv = identity(d_model, 0.05)
    wo = identity(d_model, 1.0)
    w1 = identity(d_model, 1.0)
    w2 = identity(d_model, 0.02)

    tokens = text.split()
    x = [add(embed(t, d_model), positional(i, d_model)) for i, t in enumerate(tokens)]

    for _ in range(depth):
        x = block(x, wq, wk, wv, wo, w1, w2, heads)

    pooled = [sum(row[j] for row in x) / len(x) for j in range(d_model)]
    logits = [pooled[0], -pooled[0]]
    return idx_to_word[max(range(len(logits)), key=lambda i: logits[i])]

print(f"ResShibal: {predict('논리')}")
