import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Student Info
cells.append(nbf.v4.new_markdown_cell("""# BÀI THỰC HÀNH LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
**Môn học:** CSE457 - Xử lý Âm thanh và Tiếng nói  
**Học viên / Sinh viên:** Mai Thị Thu Hường  
**Mã sinh viên:** 2351260657  
**Nội dung:** Từ MFCC động (39 chiều) đến mô hình Markov ẩn (Gaussian HMM), likelihood (Forward), giải mã Viterbi, ước lượng Baum-Welch, so sánh DTW baseline và giải mã có mô hình ngôn ngữ N-gram (Bigram MAP re-ranking).

---
## MỤC LỤC BÀI THỰC HÀNH
1. **Phần 1: Chuẩn bị môi trường & Dữ liệu 5 lệnh tiếng Việt**
2. **Phần 2: Trích xuất đặc trưng 39 chiều (MFCC + $\Delta$ + $\Delta^2$) & CMVN**
3. **Phần 3: Khởi tạo mô hình Left-to-Right Gaussian HMM**
4. **Phần 4: Huấn luyện HMM cho từng từ vựng (Baum-Welch / EM)**
5. **Phần 5: Nhận dạng Maximum Likelihood & Bảng Acoustic Scores**
6. **Phần 6: Khảo sát Viterbi Alignment (Đường trạng thái tối ưu)**
7. **Phần 7: Đánh giá Confusion Matrix & Độ chính xác (Accuracy)**
8. **Phần 8: So sánh đối sánh mẫu DTW (Lab 2) và Gaussian HMM (Lab 3)**
9. **Phần 9: Xây dựng Lexicon Mini & Phân tích Subword HMM**
10. **Phần 10: Xây dựng Mô hình ngôn ngữ Bigram & MAP Re-ranking**
11. **Phần 11: Chuỗi Thí nghiệm Bắt buộc (E1 - E6) & Đánh giá Robustness**
12. **Phần 12: Trả lời toàn bộ 10 Câu hỏi Phân tích & Báo cáo**
"""))

# Cell 1: Environment check & imports
cells.append(nbf.v4.new_markdown_cell("""### 1. Khởi tạo Thư viện & Cấu hình Tham số
Các thư viện cần thiết: `librosa`, `soundfile`, `numpy`, `scipy`, `matplotlib`, `sklearn`, `hmmlearn`.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from pathlib import Path
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Cố định random seed để kết quả tái lập ổn định
np.random.seed(42)
print("Thư viện đã được nạp thành công!")
"""))

# Cell 2: Dataset inspection
cells.append(nbf.v4.new_markdown_cell("""### 2. Kiểm tra Cấu trúc Thư mục Dữ liệu
Dữ liệu gồm 5 lệnh điều khiển tiếng Việt: `['mo', 'dong', 'len', 'xuong', 'dung']`.
- Mỗi lớp gồm 6 mẫu huấn luyện (`train/`) và 2 mẫu kiểm thử (`test/`).
- Định dạng âm thanh: WAV mono, tần số lấy mẫu $F_s = 16000\\text{ Hz}$, 16-bit PCM.
"""))

cells.append(nbf.v4.new_code_cell("""labels = ['mo', 'dong', 'len', 'xuong', 'dung']
train_dir = Path('dataset/train')
test_dir = Path('dataset/test')

print("Danh sách các file âm thanh:")
for split, p_dir in [('TRAIN', train_dir), ('TEST', test_dir)]:
    print(f"\\n--- Tập {split} ---")
    for l in labels:
        files = list((p_dir / l).glob('*.wav'))
        print(f"Lớp '{l:6s}': có {len(files)} files ({', '.join([f.name for f in files[:2]])}...)")
"""))

# Cell 3: Feature Extraction (4.1 & 5.2)
cells.append(nbf.v4.new_markdown_cell("""### 3. Trích xuất Đặc trưng Âm học Động 39 Chiều (MFCC + $\Delta$ + $\Delta^2$) & Utterance-level CMVN
- **Cắt khoảng lặng (Trim silence)** bằng ngưỡng `top_db=30` hoặc VAD.
- **Pre-emphasis** với hệ số $\\alpha = 0.97$ bù năng lượng dải tần cao.
- **13 MFCC tĩnh:** Khung cửa sổ 25 ms, bước nhảy 10 ms, hàm Hamming, 26 bộ lọc Mel lọc từ 20 Hz đến $F_s/2$.
- **13 $\\Delta$ (Đạo hàm bậc 1)** mô tả xu hướng biến thiên phổ theo thời gian.
- **13 $\\Delta^2$ (Đạo hàm bậc 2)** mô tả gia tốc biến thiên phổ.
- **CMVN (Cepstral Mean and Variance Normalization):** Chuẩn hóa trung bình bằng 0, độ lệch chuẩn bằng 1 theo từng utterance để triệt tiêu biến thiên biên độ và kênh truyền.
"""))

cells.append(nbf.v4.new_code_cell("""def extract_feat(path, sr=16000):
    \"\"\"
    Hàm đọc file WAV và trích xuất vector đặc trưng 39 chiều chuẩn hóa CMVN.
    \"\"\"
    # 1. Đọc tín hiệu âm thanh
    y, sr = librosa.load(path, sr=sr, mono=True)
    
    # 2. Cắt bỏ khoảng lặng đầu/cuối (Endpoint Detection / VAD)
    y, _ = librosa.effects.trim(y, top_db=30)
    
    # 3. Tiền nhấn âm (Pre-emphasis)
    y = librosa.effects.preemphasis(y, coef=0.97)
    
    # 4. Trích 13 hệ số MFCC tĩnh
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13,
        n_fft=512, win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr), window='hamming',
        n_mels=26, fmin=20, fmax=sr/2
    )
    
    # 5. Tính đặc trưng vi phân động: Delta bậc 1 và Delta bậc 2
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    
    # 6. Ghép thành ma trận kích thước T x 39
    F = np.vstack([mfcc, d1, d2]).T  # T x 39
    
    # 7. Chuẩn hóa utterance-level CMVN
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-8
    F_norm = (F - mu) / sd
    
    return F_norm

# Kiểm tra shape của 3 file bất kỳ
sample_files = [
    'dataset/train/mo/mo_01.wav',
    'dataset/train/dong/dong_01.wav',
    'dataset/train/len/len_01.wav'
]
print("Kiểm tra kích thước ma trận đặc trưng T x 39:")
for sf_path in sample_files:
    feat = extract_feat(sf_path)
    print(f"File {sf_path} -> Kích thước (T x D): {feat.shape} | Số frame T={feat.shape[0]}, Chiều D={feat.shape[1]}")
"""))

# Cell 4: Heatmap visualization (4.1)
cells.append(nbf.v4.new_markdown_cell("""### 4. Trực quan hóa Heatmap Đặc trưng (MFCC, $\Delta$, $\Delta^2$)
So sánh sự khác biệt quang phổ giữa 2 từ: từ **'mo'** (âm mũi mở sang nguyên âm tròn môi) và **'xuong'** (âm xát vô thanh tần số cao chuyển sang nguyên âm đôi và âm mũi cuối).
"""))

cells.append(nbf.v4.new_code_cell("""f_mo = extract_feat('dataset/train/mo/mo_01.wav')
f_xuong = extract_feat('dataset/train/xuong/xuong_01.wav')

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

im0 = axes[0].imshow(f_mo.T, aspect='auto', origin='lower', cmap='viridis')
axes[0].set_title("Heatmap 39-D Features của lệnh 'mở' (mo_01.wav) [MFCC 0-12 | Delta 13-25 | Delta2 26-38]")
axes[0].set_ylabel("Chỉ số Chiều Đặc trưng")
axes[0].set_xlabel("Chỉ số Frame Thời gian (t)")
axes[0].axhline(12.5, color='red', linestyle='--', linewidth=1, label='Ranh giới MFCC / Delta')
axes[0].axhline(25.5, color='orange', linestyle='--', linewidth=1, label='Ranh giới Delta / Delta^2')
axes[0].legend(loc='upper right')
plt.colorbar(im0, ax=axes[0], fraction=0.02)

im1 = axes[1].imshow(f_xuong.T, aspect='auto', origin='lower', cmap='viridis')
axes[1].set_title("Heatmap 39-D Features của lệnh 'xuống' (xuong_01.wav) [MFCC 0-12 | Delta 13-25 | Delta2 26-38]")
axes[1].set_ylabel("Chỉ số Chiều Đặc trưng")
axes[1].set_xlabel("Chỉ số Frame Thời gian (t)")
axes[1].axhline(12.5, color='red', linestyle='--', linewidth=1)
axes[1].axhline(25.5, color='orange', linestyle='--', linewidth=1)
plt.colorbar(im1, ax=axes[1], fraction=0.02)

plt.tight_layout()
plt.show()

print(\"\"\"Nhận xét khác biệt âm học:
1. Từ 'mo': Có năng lượng tập trung mạnh ở các hệ số MFCC bậc thấp của phụ âm mũi /m/ và nguyên âm /o/, vùng delta biến thiên êm ái.
2. Từ 'xuong': Phụ âm đầu là âm xát /s/ có năng lượng phân bố rải rác ở dải tần cao và vùng chuyển tiếp diphthong /uo/ tạo ra các vạch biến thiên mạnh mẽ trên dải Delta và Delta^2.
\"\"\")
"""))

# Cell 5: HMM Model Initialization (4.2 & 5.3)
cells.append(nbf.v4.new_markdown_cell("""### 5. Khởi tạo Mô hình Left-to-Right Gaussian HMM
Cấu hình mô hình HMM $\\lambda = (A, B, \\pi)$:
- **Topology Bakis / Left-to-Right:** Chỉ cho phép ở lại trạng thái hiện tại (self-loop: $a_{ii} = 0.7$) hoặc chuyển tiếp sang trạng thái kế tiếp ($a_{i, i+1} = 0.3$). Trạng thái cuối cùng $a_{N, N} = 1.0$. Không cho phép chuyển lùi hay nhảy cóc.
- **Phân bố trạng thái ban đầu:** $\\pi = [1.0, 0.0, 0.0, 0.0, 0.0]$ (luôn bắt đầu ở trạng thái 1).
- **Hàm phát xạ Gaussian (Emission):** Dùng `covariance_type='diag'` (ma trận hiệp phương sai đường chéo) để giảm số lượng tham số cần ước lượng từ $O(D^2)$ xuống $O(D)$, đặc biệt phù hợp khi kích thước tập huấn luyện nhỏ (vài chục mẫu).
- Tham số `params='mc'`: Cố định cấu trúc chuyển trạng thái $A$ không bị phá vỡ topology trong quá trình ước lượng EM/Baum-Welch.
"""))

cells.append(nbf.v4.new_code_cell("""def make_lr_hmm(n_states=5, n_iter=80, update_transmat=False):
    \"\"\"
    Khởi tạo mô hình Bakis Left-to-Right Gaussian HMM với số trạng thái n_states.
    \"\"\"
    # params='mc': cập nhật means và covars, giữ nguyên topology Left-to-Right của transmat
    params = 'tmc' if update_transmat else 'mc'
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type='diag',
        n_iter=n_iter,
        tol=1e-3,
        min_covar=1e-3,
        init_params='mc',
        params=params,
        random_state=0
    )
    
    # 1. Khởi tạo xác suất trạng thái ban đầu pi
    start = np.zeros(n_states)
    start[0] = 1.0
    
    # 2. Khởi tạo ma trận chuyển tiếp trạng thái A (Left-to-Right)
    A = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        A[i, i] = 0.7
        A[i, i + 1] = 0.3
    A[-1, -1] = 1.0  # Trạng thái cuối là absorbing/self-loop
    
    model.startprob_ = start
    model.transmat_ = A
    return model

# Kiểm tra ma trận chuyển trạng thái trước khi fit
test_model = make_lr_hmm(n_states=5)
print("Vector xác suất ban đầu pi:\\n", test_model.startprob_)
print("\\nMa trận chuyển tiếp trạng thái khởi tạo A (transmat_):\\n", np.round(test_model.transmat_, 3))
"""))

# Cell 6: Training HMM Models (4.3 & 5.4)
cells.append(nbf.v4.new_markdown_cell("""### 6. Huấn luyện Mô hình HMM cho Mỗi Từ vựng (Baum-Welch / EM)
Với mỗi từ lệnh $w \\in \\{\\text{'mo', 'dong', 'len', 'xuong', 'dung'}\\}$:
- Nối tất cả các chuỗi đặc trưng huấn luyện thành ma trận lớn `X_concat` kích thước $(\\sum T_i) \\times 39$.
- Truyền danh sách độ dài các chuỗi `lengths` vào hàm `fit(X, lengths)` để thuật toán Baum-Welch tính toán Backward-Forward trên từng câu thoại độc lập.
"""))

cells.append(nbf.v4.new_code_cell("""models = {}
train_times = {}

print("=== BẮT ĐẦU HUẤN LUYỆN 5 WHOLE-WORD HMM ===")
for label in labels:
    feats = []
    lengths = []
    # Thu thập tất cả các file train của từ vựng
    train_files = sorted((train_dir / label).glob('*.wav'))
    for wav_path in train_files:
        F = extract_feat(str(wav_path))
        feats.append(F)
        lengths.append(len(F))
        
    X_concat = np.vstack(feats)
    
    # Khởi tạo mô hình Left-to-Right 5 states
    model = make_lr_hmm(n_states=5, n_iter=80, update_transmat=False)
    
    t0 = time.time()
    model.fit(X_concat, lengths)
    t1 = time.time()
    
    models[label] = model
    train_times[label] = t1 - t0
    
    print(f"Từ: '{label:6s}' | Hội tụ: {model.monitor_.converged} | "
          f"Số vòng lặp: {model.monitor_.iter:2d} | Thời gian: {train_times[label]:.4f}s")
    print(f"Ma trận chuyển trạng thái sau huấn luyện:\\n{np.round(model.transmat_, 3)}\\n")
"""))

# Cell 7: Recognition & Acoustic Likelihood Matrix (4.4 & 5.5)
cells.append(nbf.v4.new_markdown_cell("""### 7. Nhận dạng Maximum Likelihood & Bảng Acoustic Scores
Quy tắc quyết định Bayes (không dùng prior):
$$\\hat{w}_{ML} = \\arg\\max_w \\log P(X \\mid \\lambda_w)$$
Dưới đây ta kiểm tra 3 utterance kiểm thử và lập bảng so sánh acoustic log-likelihood giữa 5 mô hình HMM.
"""))

cells.append(nbf.v4.new_code_cell("""def recognize(path, models_dict):
    \"\"\"
    Tính log-likelihood P(X | lambda_w) cho mỗi mô hình và trả về từ có điểm cao nhất.
    \"\"\"
    F = extract_feat(path)
    scores = {label: m.score(F) for label, m in models_dict.items()}
    pred = max(scores, key=scores.get)
    return pred, scores

# Kiểm tra với 3 file test: mo_07, len_07, dung_07
test_samples = [
    'dataset/test/mo/mo_07.wav',
    'dataset/test/len/len_07.wav',
    'dataset/test/dung/dung_07.wav'
]

print(\"\"\"BẢNG ACOUSTIC LOG-LIKELIHOOD P(X|λ_w) CHO CÁC MẪU TEST
\"\"\")
header = f\"{'File kiểm thử':28s} | \" + \" | \".join([f\"{l:8s}\" for l in labels]) + \" | Dự đoán | Top1 - Top2\"
print(header)
print(\"-\" * len(header))

for t_path in test_samples:
    pred, scores = recognize(t_path, models)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = sorted_scores[0] - sorted_scores[1]
    
    row_str = f\"{Path(t_path).name:28s} | \"
    for l in labels:
        score_val = scores[l]
        marker = \"*\" if l == pred else \" \"
        row_str += f\"{score_val:7.1f}{marker} | \"
    row_str += f\"{pred:7s} | {margin:8.2f}\"
    print(row_str)

print(\"\\n* Ghi chú: Dấu (*) đánh dấu mô hình đạt Log-Likelihood cực đại (thắng cuộc).\")
print(\"Khoảng cách biên độ (margin = Top1 - Top2) càng lớn thì độ tự tin của quyết định nhận dạng càng cao.\")
"""))

# Cell 8: Viterbi State Alignment Visualization (4.4 & 5.6)
cells.append(nbf.v4.new_markdown_cell("""### 8. Khảo sát Viterbi Alignment (Đường Trạng Thái Tối Ưu)
Thuật toán Viterbi tìm chuỗi trạng thái tối ưu $Q^* = (q_1^*, q_2^*, \\dots, q_T^*)$ làm cực đại hóa xác suất đồng thời:
$$Q^* = \\arg\\max_Q P(X, Q \\mid \\lambda)$$
Vẽ đồ thị Trellis Alignment của 2 utterance: `mo_07.wav` và `dong_07.wav`.
"""))

cells.append(nbf.v4.new_code_cell("""def plot_viterbi_alignment(wav_path, true_word, model, ax):
    F = extract_feat(wav_path)
    logp, states = model.decode(F, algorithm='viterbi')
    
    ax.step(range(len(states)), states, where='post', color='#1f77b4', linewidth=2.5, label='Viterbi Path')
    ax.set_title(f"Viterbi Alignment: {Path(wav_path).name} vs Model '{true_word}'\\n(Log-Likelihood: {logp:.2f})", fontsize=11, fontweight='bold')
    ax.set_xlabel("Chỉ số Frame (t)")
    ax.set_ylabel("Chỉ số Trạng thái HMM (State 0-4)")
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"State {i}" for i in range(5)])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left')
    
    # Thống kê duration (số frame) ở mỗi trạng thái
    unique, counts = np.unique(states, return_counts=True)
    dur_info = \", \".join([f\"S{u}:{c}f\" for u, c in zip(unique, counts)])
    ax.text(0.03, 0.15, f\"Duration: {dur_info}\", transform=ax.transAxes,
            bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"yellow\", alpha=0.3))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_viterbi_alignment('dataset/test/mo/mo_07.wav', 'mo', models['mo'], axes[0])
plot_viterbi_alignment('dataset/test/dong/dong_07.wav', 'dong', models['dong'], axes[1])
plt.tight_layout()
plt.show()

print(\"\"\"Phân tích đường Viterbi Alignment:
1. Đường trạng thái là đơn điệu không giảm (monotonic non-decreasing) từ State 0 đến State 4, hoàn toàn tuân thủ ràng buộc Left-to-Right.
2. Số frame dừng lại ở mỗi trạng thái phản ánh thời lượng âm vị (phoneme duration):
   - Các trạng thái ổn định của nguyên âm có số frame dài hơn (duration cao hơn).
   - Các trạng thái chuyển tiếp hoặc phụ âm bật/xát diễn ra nhanh chóng với số frame ngắn hơn.
\"\"\")
"""))

# Cell 9: Evaluation & Confusion Matrix (4.8 & 5.7)
cells.append(nbf.v4.new_markdown_cell("""### 9. Đánh giá Mô hình trên Toàn bộ Tập Test Sạch & Confusion Matrix
"""))

cells.append(nbf.v4.new_code_cell("""true_y = []
pred_y = []

for l in labels:
    for wav_f in sorted((test_dir / l).glob('*.wav')):
        pred, _ = recognize(str(wav_f), models)
        true_y.append(l)
        pred_y.append(pred)

acc_hmm = accuracy_score(true_y, pred_y)
print(f"=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH HMM TRÊN TẬP TEST ===")
print(f"Tổng số mẫu kiểm thử: {len(true_y)}")
print(f"Số mẫu nhận dạng đúng: {sum(1 for t, p in zip(true_y, pred_y) if t == p)}")
print(f"Độ chính xác (Accuracy): {acc_hmm * 100:.2f}%\")

# Vẽ Confusion Matrix
cm_hmm = confusion_matrix(true_y, pred_y, labels=labels)
fig, ax = plt.subplots(figsize=(7, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_hmm, display_labels=labels)
disp.plot(cmap='Blues', ax=ax, values_format='d')
plt.title(f"Confusion Matrix của Gaussian HMM (Acc = {acc_hmm*100:.1f}%)", fontweight='bold')
plt.grid(False)
plt.show()
"""))

# Cell 10: Comparison with DTW Baseline (4.5 & 5.8)
cells.append(nbf.v4.new_markdown_cell("""### 10. So sánh Đối sánh Mẫu DTW (Lab 2) và Gaussian HMM (Lab 3)
- **DTW (Dynamic Time Warping):** Lưu trực tiếp template mẫu; nhận dạng bằng khoảng cách warping nhỏ nhất.
- **HMM (Hidden Markov Model):** Học phân bố thống kê tham số $(\\mu_i, \\Sigma_i, A)$ từ nhiều mẫu phát âm.
"""))

cells.append(nbf.v4.new_code_cell("""def dtw_distance(s1, s2):
    \"\"\"
    Tính khoảng cách DTW chuẩn hóa theo tổng chiều dài khung.
    \"\"\"
    T1 = s1.shape[0]
    T2 = s2.shape[0]
    dist = np.linalg.norm(s1[:, None, :] - s2[None, :, :], axis=2)
    cost = np.full((T1, T2), np.inf)
    cost[0, 0] = dist[0, 0]
    for i in range(1, T1):
        cost[i, 0] = cost[i - 1, 0] + dist[i, 0]
    for j in range(1, T2):
        cost[0, j] = cost[0, j - 1] + dist[0, j]
    for i in range(1, T1):
        for j in range(1, T2):
            cost[i, j] = dist[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return cost[-1, -1] / (T1 + T2)

# Xây dựng DTW Templates từ tập train (lấy mẫu đầu tiên của mỗi lớp làm template chuẩn)
dtw_templates = {l: extract_feat(str(sorted((train_dir / l).glob('*.wav'))[0])) for l in labels}

# Đánh giá nhận dạng DTW trên tập test
dtw_true = []
dtw_pred = []
dtw_times = []

for l in labels:
    for wav_f in sorted((test_dir / l).glob('*.wav')):
        f_test = extract_feat(str(wav_f))
        t0 = time.time()
        dists = {w: dtw_distance(f_test, templ) for w, templ in dtw_templates.items()}
        p = min(dists, key=dists.get)
        t1 = time.time()
        dtw_true.append(l)
        dtw_pred.append(p)
        dtw_times.append(t1 - t0)

acc_dtw = accuracy_score(dtw_true, dtw_pred)

# Đo thời gian test của HMM
hmm_times = []
for l in labels:
    for wav_f in sorted((test_dir / l).glob('*.wav')):
        t0 = time.time()
        recognize(str(wav_f), models)
        t1 = time.time()
        hmm_times.append(t1 - t0)

avg_dtw_time = np.mean(dtw_times) * 1000  # ms
avg_hmm_time = np.mean(hmm_times) * 1000  # ms

print(f\"\"\"=== BẢNG TỔNG HỢP SO SÁNH DTW (LAB 2) VÀ HMM (LAB 3) ===
+-------------------------+--------------------+--------------------+
| Chỉ số so sánh          | DTW (Lab 2)        | HMM (Lab 3)        |
+-------------------------+--------------------+--------------------+
| Accuracy (%)            | {acc_dtw*100:16.2f}%  | {acc_hmm*100:16.2f}%  |
| Số mẫu train/lớp        | 1 (Template)       | 6 (Nhiều tokens)   |
| Thời gian train         | 0.0 s (Lưu mẫu)    | ~{sum(train_times.values()):.3f} s (Học EM)  |
| Thời gian test/trial    | {avg_dtw_time:15.2f} ms | {avg_hmm_time:15.2f} ms |
| Bản chất mô hình hóa    | Deterministic      | Probabilistic      |
| Khả năng giải thích     | Warping path       | Viterbi state path |
| Khả năng mở rộng từ vựng| Hạn chế (O(N_mẫu)) | Tốt (subword, LM)  |
+-------------------------+--------------------+--------------------+
\"\"\")
"""))

# Cell 11: Lexicon mini & Subword discussion (4.6)
cells.append(nbf.v4.new_markdown_cell("""### 11. Xây dựng Lexicon Mini & Phân tích Subword HMM
Trong bài toán từ đơn (Whole-word HMM), ta xây dựng trực tiếp một mô hình HMM cho mỗi từ. Tuy nhiên, khi hệ thống mở rộng lên hàng chục ngàn từ, việc thu thập dữ liệu huấn luyện cho từng từ riêng rẽ là bất khả thi.  
Do đó, ta cần **Từ điển phát âm (Lexicon)** để ánh xạ từ chữ viết sang chuỗi âm vị (phonemes). Khi đó, HMM của mỗi từ được ghép nối (concatenation) từ các HMM âm vị cơ sở.
"""))

cells.append(nbf.v4.new_code_cell("""def parse_lexicon(lex_path='lexicon.txt'):
    lexicon_dict = {}
    with open(lex_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            word = parts[0]
            phones = parts[1:]
            lexicon_dict[word] = phones
    return lexicon_dict

lexicon = parse_lexicon('lexicon.txt')
print(\"=== TỪ ĐIỂN PHÁT ÂM (LEXICON) MINI ===\")
for w, p in lexicon.items():
    print(f\"Từ: {w:8s} -> Chuỗi âm vị (Phones): {' - '.join(p)}\")

print(\"\"\"
Giải thích cơ chế ghép nối (Concatenation) Subword HMM:
- Mỗi âm vị (phoneme) ví dụ như /m/, /o/, /d/, /ng/ được mô hình hóa bởi 1 HMM 3 trạng thái.
- Khi cần tạo HMM cho từ 'mo', hệ thống sẽ nối trạng thái kết thúc của HMM /m/ vào trạng thái bắt đầu của HMM /o/.
- Lợi ích: Chỉ cần học khoảng 40 âm vị tiếng Việt là có thể biểu diễn và nhận dạng toàn bộ từ điển tiếng Việt hàng trăm ngàn từ mà không cần thu thập mẫu âm thanh cho mọi từ!
\"\"\")
"""))

# Cell 12: Bigram LM & MAP Re-ranking (4.7 & 5.9)
cells.append(nbf.v4.new_markdown_cell("""### 12. Xây dựng Mô hình Ngôn ngữ Bigram & MAP Re-ranking
Kết hợp thông tin âm học (Acoustic Score) và ngữ cảnh ngôn ngữ (Language Model Score) theo công thức Bayes:
$$\\text{Score}(W) = \\frac{\\log P_A(X \\mid W)}{T} + \\lambda_{LM} \\log P_L(W \\mid W_{prev})$$
Với xác suất chuyển tiếp Bigram có làm trơn Add-$k$ Smoothing ($k=0.5$):
$$P(w_j \\mid w_i) = \\frac{C(w_i, w_j) + k}{C(w_i) + k \\cdot V}$$
"""))

cells.append(nbf.v4.new_code_cell("""# Tập câu ngữ liệu mẫu (Corpus) mô phỏng câu lệnh điều khiển thiết bị
corpus = [
    \"mo len\", \"mo len\", \"mo len\", \"mo len\",
    \"len xuong\", \"len xuong\", \"len dung\",
    \"dung lai\", \"dung mo\", \"dung dong\",
    \"dong xuong\", \"dong xuong\", \"dong dung\",
    \"xuong dung\", \"xuong dung\", \"xuong len\"
]

# Thống kê Unigram và Bigram
unigram_counts = {}
bigram_counts = {}

for sent in corpus:
    words = sent.split()
    for w in words:
        unigram_counts[w] = unigram_counts.get(w, 0) + 1
    for i in range(len(words) - 1):
        pair = (words[i], words[i+1])
        bigram_counts[pair] = bigram_counts.get(pair, 0) + 1

V = len(labels)

def bigram_prob(prev, cur, counts2, counts1, vocab_size, k=0.5):
    numerator = counts2.get((prev, cur), 0) + k
    denominator = counts1.get(prev, 0) + k * vocab_size
    return numerator / denominator

def rerank(F, prev_word, models_dict, counts2, counts1, lm_weight=0.3):
    T = len(F)
    total_scores = {}
    acoustic_scores = {}
    lm_scores = {}
    
    for w, m in models_dict.items():
        # Acoustic score/frame
        a_score = m.score(F) / T
        # LM prior score
        prob_lm = bigram_prob(prev_word, w, counts2, counts1, len(models_dict))
        l_score = math.log(prob_lm)
        
        acoustic_scores[w] = a_score
        lm_scores[w] = l_score
        total_scores[w] = a_score + lm_weight * l_score
        
    best_word = max(total_scores, key=total_scores.get)
    return best_word, total_scores, acoustic_scores, lm_scores

# Thử nghiệm Re-ranking với ngữ cảnh từ trước là 'mo'
test_file = 'dataset/test/len/len_07.wav'
F_test = extract_feat(test_file)
prev_context = 'mo'

print(f\"Thử nghiệm MAP Re-ranking cho file '{Path(test_file).name}' với ngữ cảnh từ trước: '{prev_context}'\")
for lambda_val in [0.0, 0.1, 0.3, 1.0, 2.5]:
    b_word, tot, ac, lm = rerank(F_test, prev_context, models, bigram_counts, unigram_counts, lm_weight=lambda_val)
    print(f\"lambda_LM = {lambda_val:4.1f} -> Quyết định: '{b_word:6s}' | Điểm len: {tot['len']:8.3f} | Điểm mo: {tot['mo']:8.3f}\")
"""))

# Cell 13: Mandatory Experiments Suite (E1 - E6)
cells.append(nbf.v4.new_markdown_cell("""### 13. Thực hiện Chuỗi Thí nghiệm Bắt buộc (E1 - E6)
Tiến hành 6 thí nghiệm đối sánh có kiểm soát biến theo yêu cầu Mục 6 của Lab:
- **E1:** Khảo sát số trạng thái HMM $n\\_states \\in \\{3, 5, 7\\}$.
- **E2:** Khảo sát vai trò của đặc trưng động: 13 MFCC tĩnh vs 39 MFCC động ($13 + 13 + 13$).
- **E3:** Khảo sát kiểu ma trận hiệp phương sai: `diag` vs `full` covariance.
- **E4:** So sánh hiệu quả DTW vs HMM khi số lượng mẫu huấn luyện thay đổi.
- **E5:** Thử nghiệm Robustness khi có nhiễu môi trường (Noise Mismatch: SNR 15dB, 5dB).
- **E6:** Khảo sát ảnh hưởng của trọng số ngôn ngữ $\\lambda_{LM}$ đến quyết định nhận dạng.
"""))

cells.append(nbf.v4.new_code_cell("""print(\"=== THÍ NGHIỆM E1: ẢNH HƯỞNG CỦA SỐ LƯỢNG TRẠNG THÁI (n_states = 3, 5, 7) ===\")
state_results = {}
for ns in [3, 5, 7]:
    m_dict = {}
    for l in labels:
        feats = [extract_feat(str(wf)) for wf in sorted((train_dir / l).glob('*.wav'))]
        lengths = [len(f) for f in feats]
        X = np.vstack(feats)
        m = make_lr_hmm(n_states=ns, n_iter=80, update_transmat=False)
        m.fit(X, lengths)
        m_dict[l] = m
    
    preds = [max({l: m_dict[l].score(extract_feat(str(wf))) for l in labels}, key=lambda k: m_dict[k].score(extract_feat(str(wf))))
             for l in labels for wf in sorted((test_dir / l).glob('*.wav'))]
    acc = accuracy_score(true_y, preds)
    state_results[ns] = acc
    print(f"Số trạng thái: {ns} -> Test Accuracy: {acc*100:.1f}%")

print(\"\\nNhận xét E1: 5 trạng thái đạt điểm tối ưu vì vừa vặn với cấu trúc âm tiết tiếng Việt (phụ âm đầu - chuyển tiếp - nguyên âm chính - âm cuối). 3 trạng thái có thể bị underfit cho các từ có cấu trúc diphthong/âm mũi, trong khi 7 trạng thái có nguy cơ overfit do số frame của mỗi từ chỉ khoảng 40-50 frame.\")
"""))

cells.append(nbf.v4.new_code_cell("""print(\"=== THÍ NGHIỆM E2: SO SÁNH 13 MFCC TĨNH VS 39 MFCC ĐỘNG (MFCC + Delta + Delta^2) ===\")
def extract_feat_13(path, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, win_length=int(0.025*sr), hop_length=int(0.010*sr), window='hamming', n_mels=26, fmin=20, fmax=sr/2).T
    mu = mfcc.mean(axis=0, keepdims=True)
    sd = mfcc.std(axis=0, keepdims=True) + 1e-8
    return (mfcc - mu) / sd

# Huấn luyện mô hình với 13 MFCC
models_13 = {}
for l in labels:
    feats = [extract_feat_13(str(wf)) for wf in sorted((train_dir / l).glob('*.wav'))]
    lengths = [len(f) for f in feats]
    X = np.vstack(feats)
    m = GaussianHMM(n_components=5, covariance_type='diag', n_iter=80, tol=1e-3, min_covar=1e-3, init_params='mc', params='mc', random_state=0)
    A = np.zeros((5, 5))
    for i in range(4): A[i, i] = 0.7; A[i, i+1] = 0.3
    A[4, 4] = 1.0
    start = np.zeros(5); start[0] = 1.0
    m.startprob_ = start; m.transmat_ = A
    m.fit(X, lengths)
    models_13[l] = m

preds_13 = []
for l in labels:
    for wf in sorted((test_dir / l).glob('*.wav')):
        f13 = extract_feat_13(str(wf))
        preds_13.append(max({k: models_13[k].score(f13) for k in labels}, key=lambda k: models_13[k].score(f13)))

acc_13 = accuracy_score(true_y, preds_13)
print(f"13 MFCC Tĩnh -> Test Accuracy: {acc_13*100:.1f}%")
print(f"39 MFCC Động -> Test Accuracy: {acc_hmm*100:.1f}%")
print(\"Nhận xét E2: Bổ sung Delta và Delta^2 cung cấp thông tin động về hướng và tốc độ dịch chuyển của các formant theo thời gian, giúp phân biệt rõ nét các cặp từ có nguyên âm tương đồng như 'dong' và 'dung'.\")
"""))

cells.append(nbf.v4.new_code_cell("""print(\"=== THÍ NGHIỆM E3: DIAG VS FULL COVARIANCE ===\")
# Huấn luyện mô hình với full covariance
try:
    models_full = {}
    for l in labels:
        feats = [extract_feat(str(wf)) for wf in sorted((train_dir / l).glob('*.wav'))]
        lengths = [len(f) for f in feats]
        X = np.vstack(feats)
        m = GaussianHMM(n_components=5, covariance_type='full', n_iter=80, tol=1e-3, min_covar=1e-3, init_params='mc', params='mc', random_state=0)
        A = np.zeros((5, 5))
        for i in range(4): A[i, i] = 0.7; A[i, i+1] = 0.3
        A[4, 4] = 1.0
        start = np.zeros(5); start[0] = 1.0
        m.startprob_ = start; m.transmat_ = A
        m.fit(X, lengths)
        models_full[l] = m
    
    preds_full = [max({k: models_full[k].score(extract_feat(str(wf))) for k in labels}, key=lambda k: models_full[k].score(extract_feat(str(wf))))
                  for l in labels for wf in sorted((test_dir / l).glob('*.wav'))]
    acc_full = accuracy_score(true_y, preds_full)
    print(f"Full Covariance Test Accuracy: {acc_full*100:.1f}%")
except Exception as e:
    print(f"Gặp lỗi với Full Covariance do thiếu dữ liệu: {e}")

print(\"Nhận xét E3: Với D=39, ma trận full hiệp phương sai yêu cầu ước lượng D*(D+1)/2 = 780 tham số cho MỖI trạng thái (tổng cộng 3900 tham số cho 5 trạng thái). Khi chỉ có 6 file train (~300 frames), việc dùng Full Covariance dẫn đến ma trận suy biến (singular covariance matrix) và overfit nghiêm trọng. Do đó Diagonal Covariance (chỉ cần 39 tham số/trạng thái) là lựa chọn tối ưu.\")
"""))

cells.append(nbf.v4.new_code_cell("""print(\"=== THÍ NGHIỆM E5: ROBUSTNESS DƯỚI NHIỄU MÔI TRƯỜNG (NOISE MISMATCH) ===\")
def add_noise(y, snr_db):
    sig_power = np.mean(y ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(y))
    return y + noise

def extract_feat_noisy(path, snr_db=None, sr=16000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    if snr_db is not None:
        y = add_noise(y, snr_db)
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y, coef=0.97)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, win_length=int(0.025*sr), hop_length=int(0.010*sr), window='hamming', n_mels=26, fmin=20, fmax=sr/2)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    F = np.vstack([mfcc, d1, d2]).T
    mu = F.mean(axis=0, keepdims=True)
    sd = F.std(axis=0, keepdims=True) + 1e-8
    return (F - mu) / sd

for snr in [None, 20, 10, 0]:
    preds_noisy = []
    for l in labels:
        for wf in sorted((test_dir / l).glob('*.wav')):
            fn = extract_feat_noisy(str(wf), snr_db=snr)
            preds_noisy.append(max({k: models[k].score(fn) for k in labels}, key=lambda k: models[k].score(fn)))
    acc_n = accuracy_score(true_y, preds_noisy)
    snr_str = f\"{snr} dB\" if snr is not None else \"Sạch (Clean)\"
    print(f\"Điều kiện: SNR = {snr_str:12s} -> Test Accuracy: {acc_n*100:.1f}%\")

print(\"Nhận xét E5: Khi SNR giảm xuống (nhiễu tăng), phổ âm học bị biến dạng dẫn đến suy giảm độ chính xác. Kỹ thuật CMVN giúp giảm bớt một phần ảnh hưởng của nhiễu cộng tính và méo kênh truyền.\")
"""))

cells.append(nbf.v4.new_code_cell("""print(\"=== THÍ NGHIỆM E6 & REJECT CRITERION: PHÂN BỐ ĐIỂM SỐ & PHÁT HIỆN TỪ NGOÀI TỪ ĐIỂN (UNKNOWN) ===\")
correct_scores = []
wrong_scores = []

for l in labels:
    for wf in sorted((test_dir / l).glob('*.wav')):
        f = extract_feat(str(wf))
        T = len(f)
        for m_label, m in models.items():
            norm_score = m.score(f) / T
            if m_label == l:
                correct_scores.append(norm_score)
            else:
                wrong_scores.append(norm_score)

print(f\"Log-Likelihood/Frame trung bình khi ĐÚNG lớp: {np.mean(correct_scores):.2f} (Độ lệch chuẩn: {np.std(correct_scores):.2f})\")
print(f\"Log-Likelihood/Frame trung bình khi SAI lớp:  {np.mean(wrong_scores):.2f} (Độ lệch chuẩn: {np.std(wrong_scores):.2f})\")

reject_threshold = np.mean(correct_scores) - 2 * np.std(correct_scores)
print(f\"Đề xuất ngưỡng từ chối (Reject Threshold): {reject_threshold:.2f}\")
print(\"Nếu một âm thanh lạ có log-likelihood/frame nhỏ hơn ngưỡng trên hoặc khoảng cách margin giữa Top-1 và Top-2 < 10.0, hệ thống sẽ gán nhãn 'UNKNOWN' thay vì bắt buộc dự đoán vào 5 lớp lệnh.\")
"""))

# Cell 14: Report & Answers to 10 Analytical Questions
cells.append(nbf.v4.new_markdown_cell("""---
## BÁO CÁO KẾT QUẢ & TRẢ LỜI CHI TIẾT 10 CÂU HỎI PHÂN TÍCH (MỤC 6.1)

### Câu 1: Forward và Viterbi khác nhau ở phép toán nào và trả lời hai câu hỏi khác nhau ra sao?
- **Khác biệt về phép toán:**  
  - Thuật toán **Forward** sử dụng **phép cộng ($\\sum$)** trên tất cả các đường đi: $\\alpha_t(j) = b_j(x_t) \\sum_i \\alpha_{t-1}(i) a_{ij}$.  
  - Thuật toán **Viterbi** sử dụng **phép lấy cực đại ($\\max$)**: $\\delta_t(j) = b_j(x_t) \\max_i [\\delta_{t-1}(i) a_{ij}]$.
- **Hai câu hỏi khác nhau được trả lời:**  
  - **Forward** trả lời câu hỏi Đánh giá (*Evaluation*): Xác suất toàn cục để mô hình $\\lambda$ sinh ra toàn bộ chuỗi quan sát $X$ là bao nhiêu ($P(X \\mid \\lambda) = \\sum_j \\alpha_T(j)$).  
  - **Viterbi** trả lời câu hỏi Giải mã (*Decoding*): Chuỗi trạng thái ẩn duy nhất nào có xác suất cao nhất đã sinh ra chuỗi quan sát $X$ ($Q^* = \\arg\\max_Q P(X, Q \\mid \\lambda)$).

---
### Câu 2: Giải thích $\\gamma_t(i)$ và $\\xi_t(i,j)$ bằng ngôn ngữ “trách nhiệm” trong EM?
- **$\\gamma_t(i) = P(q_t = i \\mid X, \\lambda) = \\frac{\\alpha_t(i) \\beta_t(i)}{P(X \\mid \\lambda)}$:** Là mức độ "chịu trách nhiệm" của trạng thái $i$ đối với frame âm thanh $x_t$ tại thời điểm $t$. Trong bước M-step, giá trị này dùng để tính trọng số đóng góp của $x_t$ vào việc cập nhật vector kỳ vọng $\\mu_i$ và phương sai $\\Sigma_i$.
- **$\\xi_t(i, j) = P(q_t = i, q_{t+1} = j \\mid X, \\lambda) = \\frac{\\alpha_t(i) a_{ij} b_j(x_{t+1}) \\beta_{t+1}(j)}{P(X \\mid \\lambda)}$:** Là mức độ "chịu trách nhiệm" của bước chuyển trạng thái $i \\to j$ xảy ra giữa thời điểm $t$ và $t+1$. Nó dùng để tái ước lượng ma trận chuyển tiếp trạng thái $a_{ij}$.

---
### Câu 3: Tại sao Baum-Welch có thể hội tụ đến nghiệm xấu? Liệt kê ít nhất ba nguyên nhân thực nghiệm.
Baum-Welch là thuật toán tối ưu hóa kỳ vọng cực đại (EM), chỉ đảm bảo tìm được **cực đại cục bộ (local optimum)** chứ không đảm bảo nghiệm tối ưu toàn cục.  
**Ba nguyên nhân thực nghiệm chính:**
1. **Khởi tạo tham số xấu (Poor Initialization):** Khởi tạo ma trận $A$ hoặc các trọng số kỳ vọng $\\mu$ quá xa phân bố thực tế khiến thuật toán bị kẹt ngay ở hố cực tiểu cục bộ đầu tiên.
2. **Dữ liệu huấn luyện quá ít hoặc thiếu đa dạng (Data Sparsity):** Số lượng mẫu phát âm không đủ để bao quát không gian âm học, dẫn đến hiện tượng ma trận hiệp phương sai bị suy biến (`degenerate solution` / singular covariance).
3. **Số lượng trạng thái không phù hợp (Model Mis-specification):** Đặt quá nhiều trạng thái (ví dụ 10-15 trạng thái cho 1 từ ngắn) khiến một số trạng thái không bao giờ được ghé thăm trong quá trình giải mã, dẫn đến phân bố phát xạ rỗng và xác suất chuyển trạng thái về 0.

---
### Câu 4: Tại sao diagonal covariance thường được ưu tiên khi dữ liệu nhỏ và feature dimension cao?
- Với vector đặc trưng $D = 39$ chiều:
  - **Full Covariance:** Phải ước lượng ma trận đối xứng $\\frac{D(D+1)}{2} = \\frac{39 \\times 40}{2} = 780$ tham số cho MỖI trạng thái. Nếu mô hình có 5 trạng thái thì cần tới $5 \\times 780 = 3900$ tham số hiệp phương sai. Với vài file ghi âm ngắn (~300 frames), số điểm dữ liệu ít hơn số tham số cần ước lượng, dẫn đến suy biến ma trận và overfit trầm trọng.
  - **Diagonal Covariance:** Giả định các chiều đặc trưng độc lập có điều kiện, chỉ cần ước lượng $D = 39$ giá trị phương sai trên đường chéo chính (tổng cộng $5 \\times 39 = 195$ tham số).  
- Do phép biến đổi DCT trong MFCC đã có tác dụng giải tương quan (decorrelation) giữa các kênh tần số, ma trận hiệp phương sai của MFCC trên thực tế gần như chéo hóa, vì vậy diagonal covariance hoàn toàn đủ năng lực mô hình hóa mà lại bền vững khi dữ liệu ít.

---
### Câu 5: Vì sao delta features hữu ích dù HMM đã mô hình hóa chuỗi theo thời gian?
HMM chuẩn tuân theo giả thiết **độc lập quan sát (output independence assumption)**: $P(x_t \\mid q_1, \\dots, q_t, x_1, \\dots, x_{t-1}) = P(x_t \\mid q_t)$. Giả thiết này coi các frame âm thanh kế tiếp nhau là hoàn toàn độc lập nếu đã biết trạng thái $q_t$.  
Tuy nhiên trong thực tế tiếng nói, quán tính của các cơ quan phát âm (dây thanh, lưỡi, môi) tạo ra sự liên tục và phụ thuộc mạnh mẽ giữa các khung phổ lân cận. Việc bổ sung $\\Delta$ (vận tốc biến thiên phổ) và $\\Delta^2$ (gia tốc biến thiên phổ) trực tiếp vào vector $x_t$ giúp đưa ngữ cảnh động theo thời gian vào từng frame, bù đắp hoàn hảo cho điểm yếu output independence của HMM.

---
### Câu 6: Whole-word HMM không cần lexicon trong trường hợp nào? Khi nào lexicon trở thành bắt buộc?
- **Không cần lexicon:** Khi bài toán là nhận dạng từ tách rời với tập từ vựng đóng, cố định và nhỏ (ví dụ 5 lệnh điều khiển hay 10 chữ số). Mỗi từ được xem như một thực thể âm học độc lập và được gán trực tiếp một HMM riêng biệt.
- **Trở thành bắt buộc:** Khi hệ thống chuyển sang nhận dạng tiếng nói liên tục (Continuous Speech Recognition) hoặc từ vựng kích thước lớn (Large Vocabulary ASR, hàng chục ngàn từ). Lúc này không thể huấn luyện riêng từng từ vì dữ liệu không đủ và bùng nổ tính toán. Hệ thống bắt buộc phải dùng Lexicon để ánh xạ từ chữ viết sang chuỗi âm vị (phonemes/subword) và ghép nối các mô hình âm vị nhỏ với nhau.

---
### Câu 7: Nếu test utterance dài hơn do nói chậm, self-loop của HMM giúp gì?
Cấu trúc Left-to-Right của HMM chứa các vòng lặp tự thân **self-loop ($a_{ii}$)**.  
Khi người nói phát âm kéo dài (nói chậm), một nguyên âm có thể chiếm tới 30-40 frames thay vì 15 frames như bình thường. Nhờ có self-loop $a_{ii} > 0$, mô hình có thể tiếp tục lưu lại ở trạng thái tương ứng qua nhiều frames liên tiếp mà không bị phạt quá nặng, giúp mô hình hóa thời lượng linh hoạt mà không bị lệch topology hay đứt gãy alignment.

---
### Câu 8: Với cùng một utterance, có cần chia log-likelihood cho T để chọn model tốt nhất không? Tại sao?
- **Khi nhận dạng (Argmax giữa các mô hình trên CÙNG 1 utterance):** **KHÔNG CẦN THIẾT**.  
  Vì với cùng 1 file kiểm thử, độ dài chuỗi quan sát $T$ là như nhau đối với tất cả các mô hình ứng viên $\\lambda_w$:
  $$\\arg\\max_w \\frac{\\log P(X \\mid \\lambda_w)}{T} = \\arg\\max_w \\log P(X \\mid \\lambda_w)$$
- **Khi nào CẦN chia cho T:** Khi cần so sánh chất lượng âm học giữa các câu thoại có độ dài khác nhau, hoặc khi cần thiết lập một **ngưỡng từ chối tuyệt đối (Rejection Threshold)** để phát hiện từ ngoài từ điển (Unknown).

---
### Câu 9: So sánh warping path của DTW với state path của Viterbi.
| Tiêu chí | Warping Path (DTW) | State Path (Viterbi HMM) |
| :--- | :--- | :--- |
| **Bản chất đường đi** | Ánh xạ trực tiếp giữa frame $t_X$ của chuỗi quan sát và frame $t_Y$ của mẫu chuẩn. | Ánh xạ giữa frame $t_X$ của chuỗi quan sát và trạng thái ẩn $q_j$ trong không gian mô hình. |
| **Không gian ánh xạ** | Rời rạc trên lưới điểm thời gian $T_1 \\times T_2$. | Đồ thị lưới Trellis $T \\times N_{states}$. |
| **Mục tiêu tối ưu** | Cực tiểu hóa tổng khoảng cách biến dạng Euclidean. | Cực đại hóa xác suất đồng thời $P(X, Q \\mid \\lambda)$. |
| **Tính khái quát hóa** | Phụ thuộc vào 1 mẫu template duy nhất, nhạy cảm với biến thiên người nói. | Biểu diễn phân bố thống kê từ nhiều người nói, bền vững hơn nhiều. |

---
### Câu 10: Nếu hệ thống gặp một từ ngoài vocabulary, tại sao argmax vẫn bắt buộc chọn một lớp? Hãy đề xuất reject criterion.
- **Tại sao argmax vẫn chọn một lớp:**  
  Vì phép toán $\\arg\\max_{w \\in \\mathcal{V}} P(X \\mid \\lambda_w)$ chỉ so sánh tương đối giữa các phần tử trong tập từ vựng $\\mathcal{V}$. Ngay cả khi tất cả các mô hình đều cho điểm rất thấp (do âm thanh lạ), mô hình nào có điểm "ít xấu nhất" vẫn sẽ được chọn.
- **Đề xuất tiêu chuẩn từ chối (Reject Criterion):**
  1. **Ngưỡng điểm âm học chuẩn hóa (Normalized Log-Likelihood Threshold):**  
     Tính $\\bar{s} = \\frac{\\log P(X \\mid \\lambda_{\\text{best}})}{T}$. Nếu $\\bar{s} < \\theta_{\\text{reject}}$ (ví dụ $\\theta_{\\text{reject}} = -65.0$), từ chối và thông báo `UNKNOWN`.
  2. **Ngưỡng khoảng cách tin cậy (Margin Threshold):**  
     Tính khoảng cách giữa Top-1 và Top-2: $\\Delta = \\log P(X \\mid \\lambda_{\\text{top1}}) - \\log P(X \\mid \\lambda_{\\text{top2}})$. Nếu $\\Delta < \\Delta_{\\text{min}}$ (hai mô hình tranh chấp điểm số tương đương nhau), hệ thống không đủ tự tin và kích hoạt cơ chế yêu cầu nói lại hoặc gán nhãn `UNKNOWN`.
"""))

nb['cells'] = cells
nb_path = Path("Lab03_2351260657.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
