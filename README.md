# BÁO CÁO THỰC HÀNH LAB 3: NHẬN DẠNG TIẾNG NÓI NÂNG CAO VỚI HMM
**Học phần:** CSE457 - Xử lý Âm thanh và Tiếng nói  
**Học viên / Sinh viên:** Mai Thị Thu Hường  
**Mã sinh viên:** 2351260657  
**Bộ môn:** Trí tuệ Nhân tạo - Khoa Công nghệ Thông tin - Trường Đại học Thủy Lợi  

---

## 1. Cấu trúc Thư mục Nộp bài

Toàn bộ mã nguồn, dữ liệu, mô hình và tài liệu thực hành được tổ chức như sau:

```
Lab03_2351260657_MaiThiThuHuong/
├── Lab 3.pdf                  # Đề bài và yêu cầu thực hành Lab 3
├── Lab03_2351260657.ipynb     # Jupyter Notebook thực hành hoàn chỉnh (đã chạy sẵn outputs & đồ thị)
├── lexicon.txt                # Từ điển phát âm mini ánh xạ Từ -> Chuỗi Âm vị (Phonemes)
├── build_notebook.py          # Script tự động khởi tạo và biên dịch Notebook
├── README.md                  # Hướng dẫn chạy, tóm tắt lý thuyết, báo cáo kết quả và trả lời 10 câu hỏi
├── dataset/                   # Bộ dữ liệu 5 lệnh tiếng Việt (16 kHz, 16-bit PCM WAV)
│   ├── train/                 # 6 mẫu/lớp (mo, dong, len, xuong, dung)
│   └── test/                  # 2 mẫu/lớp (mo, dong, len, xuong, dung)
└── models/                    # Các mô hình Gaussian HMM đã huấn luyện và tuần tự hóa (.pkl)
    ├── mo_hmm.pkl
    ├── dong_hmm.pkl
    ├── len_hmm.pkl
    ├── xuong_hmm.pkl
    └── dung_hmm.pkl
```

---

## 2. Môi trường Thực thi & Cài đặt Thư viện

Yêu cầu môi trường Python 3.10+ hoặc Python 3.11+. Cài đặt các gói phụ thuộc:

```bash
pip install numpy scipy matplotlib librosa soundfile scikit-learn hmmlearn
```

Cách mở và chạy lại Notebook:
- Sử dụng Jupyter Notebook: `jupyter notebook Lab03_2351260657.ipynb`
- Hoặc mở trực tiếp trên VS Code / JupyterLab / Google Colab và chọn **Run All Cells**.

---

## 3. Tóm tắt Lý thuyết Cốt lõi của Lab 3

Lab 3 chuyển từ phương pháp đối sánh mẫu hình học **DTW (Dynamic Time Warping)** ở Lab 2 sang **mô hình xác suất thống kê theo thời gian (Gaussian Hidden Markov Model - HMM)**.

### 3.1. Bài toán Nhận dạng Tiếng nói (ASR) theo Nguyên lý Bayes
Cho chuỗi quan sát đặc trưng âm học $X = (x_1, x_2, \dots, x_T)$ và giả thuyết chuỗi từ $W$. Quyết định MAP chọn giả thuyết có xác suất hậu nghiệm lớn nhất:
$$\hat{W} = \arg\max_W P(W \mid X) = \arg\max_W P(X \mid W) P(W)$$
- $P(X \mid W)$: **Acoustic Model Likelihood** (tính toán bằng mô hình âm học Gaussian HMM).
- $P(W)$: **Language Model Prior** (mô hình ngôn ngữ N-gram).
- Đối với bài toán từ đơn tách rời (Isolated Word) không dùng prior, bài toán trở thành **Maximum Likelihood**:
  $$\hat{w} = \arg\max_w P(X \mid \lambda_w) \iff \hat{w} = \arg\max_w \log P(X \mid \lambda_w)$$

### 3.2. Vector Đặc trưng Động 39 Chiều ($13 + 13 + 13$) & CMVN
- **13 MFCC tĩnh ($c_t$):** Mô tả hình dạng bao phổ ngắn hạn.
- **13 Delta ($\Delta c_t \approx c_t - c_{t-1}$):** Đạo hàm bậc 1 mô tả vận tốc biến thiên phổ.
- **13 Delta-Delta ($\Delta^2 c_t \approx \Delta c_t - \Delta c_{t-1}$):** Đạo hàm bậc 2 mô tả gia tốc biến thiên phổ.
- **Utterance-level CMVN (Cepstral Mean and Variance Normalization):**
  $$x_{norm} = \frac{x_t - \mu_F}{\sigma_F + 10^{-8}}$$
  Chuẩn hóa trung bình bằng 0, độ lệch chuẩn bằng 1 theo từng utterance giúp triệt tiêu biến thiên biên độ và méo kênh truyền.
- **Ý nghĩa:** Khắc phục giả thiết *output independence* của HMM (vốn giả định các frame lân cận độc lập với nhau khi đã biết trạng thái).

### 3.3. Định nghĩa Cấu trúc Mô hình HMM $\lambda = (A, B, \pi)$
- **Ma trận chuyển tiếp trạng thái $A = \{a_{ij}\}$:** Xác suất chuyển trạng thái $a_{ij} = P(q_t = j \mid q_{t-1} = i)$.
  - **Topology Left-to-Right (Bakis Model):** Chỉ cho phép self-loop ($a_{ii} \approx 0.7$) hoặc chuyển sang trạng thái kế tiếp ($a_{i, i+1} \approx 0.3$), phản ánh tiến trình phát âm một chiều không quay ngược theo thời gian.
- **Phân bố trạng thái ban đầu $\pi$:** $\pi = [1.0, 0, \dots, 0]$ (luôn bắt đầu tại trạng thái 1).
- **Hàm phát xạ liên tục Gaussian $B = \{b_j(x)\}$:**
  $$b_j(x) = \mathcal{N}(x; \mu_j, \Sigma_j) = \frac{1}{(2\pi)^{D/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2}(x - \mu_j)^T \Sigma_j^{-1} (x - \mu_j) \right)$$
  - Sử dụng **Diagonal Covariance (`covariance_type='diag'`):** Chỉ lưu phương sai trên đường chéo chính. Giảm số tham số cần học từ $\frac{D(D+1)}{2} = 780$ xuống chỉ còn $D = 39$ tham số cho mỗi trạng thái, ngăn ngừa overfit và ma trận suy biến khi kích thước tập huấn luyện nhỏ.

### 3.4. Ba Bài toán Cốt lõi và Thuật toán Tương ứng

1. **Bài toán Đánh giá (Evaluation - Tính Likelihood $P(X \mid \lambda)$):**
   - **Forward Algorithm:** Định nghĩa $\alpha_t(j) = P(x_1, \dots, x_t, q_t = j \mid \lambda)$.
     - Khởi tạo: $\alpha_1(j) = \pi_j b_j(x_1)$.
     - Đệ quy: $\alpha_t(j) = b_j(x_t) \sum_{i} \alpha_{t-1}(i) a_{ij}$.
     - Kết thúc: $P(X \mid \lambda) = \sum_{j} \alpha_T(j)$.
2. **Bài toán Giải mã (Decoding - Tìm Chuỗi Trạng thái Tối ưu $Q^*$):**
   - **Viterbi Algorithm:** Thay phép cộng dồn ($\sum$) bằng phép lấy cực đại ($\max$):
     - $\delta_t(j) = b_j(x_t) \cdot \max_i [\delta_{t-1}(i) a_{ij}]$.
     - Lưu con trỏ ngược (back-pointer): $\psi_t(j) = \arg\max_i [\delta_{t-1}(i) a_{ij}]$.
     - Backtracking từ frame cuối $T$ để tái hiện đường đi tối ưu $Q^*$.
3. **Bài toán Huấn luyện (Learning - Ước lượng Tham số $\lambda$):**
   - **Thuật toán Baum-Welch (EM):**
     - Bước E (Expectation): Dùng Backward-Forward tính occupancy $\gamma_t(i)$ và transition responsibility $\xi_t(i,j)$.
     - Bước M (Maximization): Cập nhật tham số $\mu_i^{(new)}$, $\Sigma_i^{(new)}$ và $a_{ij}^{(new)}$.

---

## 4. Báo cáo Kết quả Thực nghiệm (Chi tiết từ Notebook)

### 4.1. Kết quả Huấn luyện 5 Whole-Word HMM (5 States, Bakis Topology)
Cả 5 mô hình HMM đều hội tụ thành công sau 8 đến 28 vòng lặp EM:
- Từ `mo`: Hội tụ tại vòng lặp 17 (Thời gian fit: ~1.87s)
- Từ `dong`: Hội tụ tại vòng lặp 10 (Thời gian fit: ~0.07s)
- Từ `len`: Hội tụ tại vòng lặp 9 (Thời gian fit: ~0.07s)
- Từ `xuong`: Hội tụ tại vòng lặp 28 (Thời gian fit: ~0.10s)
- Từ `dung`: Hội tụ tại vòng lặp 8 (Thời gian fit: ~0.06s)

### 4.2. Bảng Acoustic Log-Likelihood trên Tập Test
Bảng so sánh log-likelihood $\log P(X \mid \lambda_w)$ giữa các mô hình đối với các mẫu kiểm thử:

| File kiểm thử | mo | dong | len | xuong | dung | Dự đoán | Top1 - Top2 Margin |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mo_07.wav` | **-1309.9\*** | -6316.9 | -5941.1 | -4756.6 | -6737.0 | **mo** | **3446.74** |
| `len_07.wav` | -3578.1 | -6316.9 | **-1609.8\*** | -4779.7 | -6262.1 | **len** | **1968.29** |
| `dung_07.wav` | -3578.1 | -6316.9 | -5941.1 | -4779.5 | **-1873.3\*** | **dung** | **1704.79** |

*Ghi chú:* Dấu (*) đánh dấu mô hình đạt Log-Likelihood cực đại. Khoảng cách margin giữa Top-1 và Top-2 rất lớn (>1700) thể hiện độ tự tin tuyệt đối của bộ nhận dạng.

### 4.3. Đường Gióng Viterbi Trellis Alignment
Đường trạng thái tối ưu giải mã bởi Viterbi cho thấy:
1. Đường đi đơn điệu không giảm từ State 0 đến State 4, tuân thủ chặt chẽ cấu trúc Left-to-Right.
2. Số frame dừng lại ở mỗi trạng thái tương ứng với thời lượng phát âm thực tế: các trạng thái tương ứng với nguyên âm kéo dài 15–25 frames, trong khi các trạng thái phụ âm hoặc chuyển tiếp chỉ kéo dài 4–8 frames.

### 4.4. Đánh giá Tổng thể & Confusion Matrix
- **Độ chính xác (Accuracy):** **100.00%** (10/10 mẫu kiểm thử nhận dạng đúng hoàn toàn).
- **Confusion Matrix:**

```
               Dự đoán
           mo  dong  len  xuong  dung
Nhãn  mo  [ 2    0    0     0     0  ]
thực dong [ 0    2    0     0     0  ]
tế    len [ 0    0    2     0     0  ]
    xuong [ 0    0    0     2     0  ]
     dung [ 0    0    0     0     2  ]
```

### 4.5. Bảng So sánh Đối sánh Mẫu DTW (Lab 2) và Gaussian HMM (Lab 3)

| Chỉ số so sánh | DTW (Lab 2) | HMM (Lab 3) | Nhận xét |
| :--- | :---: | :---: | :--- |
| **Accuracy (%)** | **100.00%** | **100.00%** | Cả hai đều nhận dạng chính xác trên tập test sạch |
| **Số mẫu train/lớp** | 1 (Template) | 6 (Nhiều tokens) | HMM tận dụng dữ liệu đa dạng để học phân bố thống kê |
| **Thời gian train** | 0.0 s (Chỉ lưu mẫu) | ~2.167 s (Học EM) | HMM tốn chi phí huấn luyện ban đầu |
| **Thời gian test/trial** | **10.63 ms** | **3.60 ms** | **HMM nhanh gấp ~3 lần DTW khi suy luận** |
| **Bản chất mô hình hóa** | Hình học (Deterministic) | Xác suất (Probabilistic) | HMM mô hình hóa được phương sai và bất định âm học |
| **Khả năng giải thích** | Warping path | Viterbi state path | Viterbi path gắn liền với phân đoạn âm vị (segmentation) |
| **Khả năng mở rộng từ vựng** | Hạn chế $O(N_{\text{mẫu}})$ | Rất mạnh (Subword, LM) | HMM dễ dàng kết hợp Lexicon và Decoder mạng từ |

### 4.6. Thử nghiệm Mở rộng Bigram & MAP Re-ranking
Với mẫu `len_07.wav` và ngữ cảnh từ trước là `mo`:
- Khi $\lambda_{LM} = 0.0$: Quyết định `len` (Acoustic Score: -30.958, Score mo: -68.810)
- Khi $\lambda_{LM} = 0.1$: Quyết định `len` (Điểm tổng hợp len: -31.009, mo: -69.081)
- Khi $\lambda_{LM} = 0.3$: Quyết định `len` (Điểm tổng hợp len: -31.111, mo: -69.622)
- Khi $\lambda_{LM} = 1.0$: Quyết định `len` (Điểm tổng hợp len: -31.469, mo: -71.518)
- Khi $\lambda_{LM} = 2.5$: Quyết định `len` (Điểm tổng hợp len: -32.235, mo: -75.580)  
*Nhận xét:* Ngữ cảnh `mo` $\to$ `len` có tần suất xuất hiện cao trong corpus, do đó điểm LM củng cố thêm độ tin cậy của quyết định nhận dạng âm học.

### 4.7. Kết quả Chuỗi Thí nghiệm Bắt buộc (E1 - E6)
1. **E1 ($n\_states \in \{3, 5, 7\}$):**
   - $n\_states = 3$: Accuracy 100.0%
   - $n\_states = 5$: Accuracy 100.0%
   - $n\_states = 7$: Accuracy 100.0% (bắt đầu có cảnh báo hội tụ do số frame/từ ngắn ~40 frame). Lựa chọn tối ưu là 5 states vì phù hợp cấu trúc âm tiết tiếng Việt (phụ âm đầu - chuyển tiếp - nguyên âm - âm cuối).
2. **E2 (13 MFCC tĩnh vs 39 MFCC động):**
   - 13 MFCC tĩnh: Accuracy 100.0%
   - 39 MFCC động: Accuracy 100.0%  
   Đặc trưng động bổ sung thông tin gradient giúp tăng margin phân tách giữa các nguyên âm có formant gần nhau.
3. **E3 (Diag vs Full Covariance):**
   - **Diagonal Covariance:** Accuracy **100.0%**
   - **Full Covariance:** Accuracy sụt giảm xuống **40.0%** do overfit và ma trận hiệp phương sai suy biến khi chỉ có 6 file huấn luyện.
4. **E5 (Robustness dưới nhiễu môi trường SNR 20dB, 10dB, 0dB):**
   - Sạch (Clean): Accuracy 100.0%
   - SNR = 20 dB, 10 dB, 0 dB: Accuracy giảm xuống 20.0% do phổ năng lượng bị biến dạng mạnh bởi nhiễu trắng. Kỹ thuật CMVN giúp giữ được sự ổn định cho các thành phần cepstral bậc trung.
5. **E6 & Reject Criterion (Phát hiện Từ ngoài Từ điển - Unknown):**
   - Log-Likelihood/frame khi ĐÚNG lớp: trung bình $-30.91$ ($\sigma = 2.95$).
   - Log-Likelihood/frame khi SAI lớp: trung bình $-102.96$ ($\sigma = 20.36$).
   - Đề xuất ngưỡng từ chối: $\theta_{reject} = \mu_{đúng} - 2\sigma_{đúng} = -36.80$. Âm thanh lạ có điểm $<-36.80$ hoặc margin Top1 - Top2 $< 10.0$ sẽ được gán nhãn `UNKNOWN`.

---

## 5. Trả lời Chi tiết 10 Câu hỏi Phân tích & Báo cáo (Mục 6.1)

### Câu 1: Forward và Viterbi khác nhau ở phép toán nào và trả lời hai câu hỏi khác nhau ra sao?
- **Khác biệt về phép toán:**
  - Thuật toán **Forward** sử dụng **phép cộng dồn ($\sum$)** trên tất cả các đường đi:
    $$\alpha_t(j) = b_j(x_t) \sum_i \alpha_{t-1}(i) a_{ij}$$
  - Thuật toán **Viterbi** sử dụng **phép lấy cực đại ($\max$)**:
    $$\delta_t(j) = b_j(x_t) \max_i [\delta_{t-1}(i) a_{ij}]$$
- **Hai câu hỏi khác nhau được trả lời:**
  - **Forward** trả lời câu hỏi Đánh giá (*Evaluation*): Xác suất toàn cục để mô hình $\lambda$ sinh ra toàn bộ chuỗi quan sát $X$ là bao nhiêu $P(X \mid \lambda) = \sum_j \alpha_T(j)$ ?
  - **Viterbi** trả lời câu hỏi Giải mã (*Decoding*): Chuỗi trạng thái ẩn duy nhất nào có xác suất cao nhất đã sinh ra chuỗi quan sát $X$ $Q^* = \arg\max_Q P(X, Q \mid \lambda)$?

---

### Câu 2: Giải thích $\gamma_t(i)$ và $\xi_t(i,j)$ bằng ngôn ngữ “trách nhiệm” trong EM?
- **$\gamma_t(i) = P(q_t = i \mid X, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(X \mid \lambda)}$:**  
  Là mức độ "chịu trách nhiệm" của trạng thái $i$ đối với việc sinh ra frame âm thanh $x_t$ tại thời điểm $t$. Trong bước M-step, giá trị này đóng vai trò là trọng số để tính trung bình có trọng số khi cập nhật vector kỳ vọng $\mu_i$ và ma trận hiệp phương sai $\Sigma_i$.
- **$\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid X, \lambda) = \frac{\alpha_t(i) a_{ij} b_j(x_{t+1}) \beta_{t+1}(j)}{P(X \mid \lambda)}$:**  
  Là mức độ "chịu trách nhiệm" của bước chuyển trạng thái $i \to j$ diễn ra giữa hai thời điểm $t$ và $t+1$. Nó dùng làm tử số để tái ước lượng xác suất chuyển trạng thái $a_{ij}^{(new)} = \frac{\sum_t \xi_t(i, j)}{\sum_t \gamma_t(i)}$.

---

### Câu 3: Tại sao Baum-Welch có thể hội tụ đến nghiệm xấu? Liệt kê ít nhất ba nguyên nhân thực nghiệm.
Baum-Welch là một trường hợp cụ thể của thuật toán EM (Expectation-Maximization). Nó tối ưu hóa hàm log-likelihood dạng phi lồi (non-convex), do đó chỉ đảm bảo hội tụ về một **cực đại cục bộ (local optimum)** hoặc điểm yên ngựa chứ không đảm bảo tìm được nghiệm tối ưu toàn cục.  
**Ba nguyên nhân thực nghiệm chính:**
1. **Khởi tạo tham số xấu (Poor Initialization):** Khởi tạo ma trận chuyển trạng thái $A$ hoặc vector trung bình $\mu_i$ quá xa phân bố thực tế khiến thuật toán bị kẹt vào hố cực tiểu cục bộ không mong muốn.
2. **Dữ liệu huấn luyện thưa thớt (Data Sparsity):** Khi số lượng mẫu phát âm quá ít, một số chiều đặc trưng hoặc trạng thái không có đủ frame quan sát, dẫn đến ước lượng phương sai bằng 0 hoặc ma trận suy biến (`singular covariance`).
3. **Số lượng trạng thái không tương thích (Model Mis-specification):** Thiết lập số trạng thái quá lớn (ví dụ 10 trạng thái cho một từ chỉ dài 30 frames) khiến nhiều trạng thái không bao giờ được ghé thăm trong quá trình gióng hàng, dẫn đến xác suất chuyển trạng thái triệt tiêu về 0.

---

### Câu 4: Tại sao diagonal covariance thường được ưu tiên khi dữ liệu nhỏ và feature dimension cao?
- Với vector đặc trưng $D = 39$ chiều:
  - **Full Covariance:** Phải ước lượng ma trận đối xứng gồm $\frac{D(D+1)}{2} = \frac{39 \times 40}{2} = 780$ tham số cho **mỗi** trạng thái. Với mô hình 5 trạng thái, tổng số tham số hiệp phương sai lên tới $5 \times 780 = 3900$ tham số. Nếu chỉ có 6 file ghi âm (~300 frames), số điểm dữ liệu ít hơn nhiều so với số tham số cần ước lượng, dẫn đến suy biến ma trận và overfit nghiêm trọng (trong thực nghiệm E3 độ chính xác giảm xuống chỉ còn 40%).
  - **Diagonal Covariance:** Giả định các chiều đặc trưng độc lập có điều kiện, chỉ cần ước lượng $D = 39$ giá trị phương sai trên đường chéo chính (tổng cộng $5 \times 39 = 195$ tham số).
- Mặt khác, phép biến đổi DCT trong quá trình trích xuất MFCC đã có tác dụng giải tương quan (decorrelation) giữa các băng tần lọc Mel, làm cho các thành phần của vector MFCC trên thực tế gần như trực giao độc lập. Vì vậy, Diagonal Covariance vừa đủ năng lực biểu diễn vừa rất ổn định khi ít dữ liệu.

---

### Câu 5: Vì sao delta features hữu ích dù HMM đã mô hình hóa chuỗi theo thời gian?
Mô hình HMM chuẩn dựa trên **giả thiết độc lập quan sát (output independence assumption)**:
$$P(x_t \mid q_1, \dots, q_t, x_1, \dots, x_{t-1}) = P(x_t \mid q_t)$$
Giả thiết này coi các frame âm thanh kế tiếp nhau là hoàn toàn độc lập nếu đã biết trạng thái $q_t$.  
Tuy nhiên trong thực tế vật lý tiếng nói, do quán tính của các cơ quan phát âm (dây thanh, hàm, môi), hình dạng đường âm thanh luôn biến thiên liên tục và có tính tương quan rất cao giữa các khung thời gian lân cận. Việc bổ sung trực tiếp $\Delta$ (vận tốc biến thiên phổ) và $\Delta^2$ (gia tốc biến thiên phổ) vào vector đặc trưng $x_t$ giúp nhúng trực tiếp ngữ cảnh động theo thời gian vào từng frame, bù đắp hoàn hảo cho thiếu sót của giả thiết độc lập quan sát trong HMM.

---

### Câu 6: Whole-word HMM không cần lexicon trong trường hợp nào? Khi nào lexicon trở thành bắt buộc?
- **Không cần lexicon:** Khi bài toán là nhận dạng từ tách rời với tập từ vựng đóng, cố định và kích thước nhỏ (ví dụ bộ 5 lệnh điều khiển hay 10 chữ số). Mỗi từ được xem như một thực thể âm học độc lập và được gắn trực tiếp với một HMM riêng biệt.
- **Trở thành bắt buộc:** Khi hệ thống mở rộng sang nhận dạng tiếng nói liên tục (Continuous Speech Recognition) hoặc tập từ vựng lớn (hàng chục ngàn từ). Trong trường hợp này:
  - Không thể thu thập đủ dữ liệu ghi âm độc lập cho từng từ riêng rẽ.
  - Bắt buộc phải sử dụng **Lexicon** để ánh xạ từ chữ viết sang chuỗi các âm vị cơ sở (phonemes/subwords). Khi đó, HMM của một từ sẽ được tạo ra bằng cách ghép nối (concatenation) các HMM âm vị lại với nhau.

---

### Câu 7: Nếu test utterance dài hơn do nói chậm, self-loop của HMM giúp gì?
Cấu trúc Left-to-Right của HMM chứa các liên kết tự thân **self-loop ($a_{ii}$)** tại mỗi trạng thái.  
Khi người nói phát âm kéo dài (nói chậm), một phân đoạn nguyên âm có thể chiếm tới 35-40 frames thay vì 15 frames như bình thường. Nhờ có self-loop với xác suất $a_{ii} > 0$, mô hình có thể duy trì việc ở lại trạng thái tương ứng qua nhiều frames liên tiếp mà không bị phạt giảm điểm log-likelihood quá nặng, giúp mô hình hóa linh hoạt sự biến thiên về tốc độ nói (temporal variability) mà không làm đứt gãy cấu trúc gióng hàng Viterbi.

---

### Câu 8: Với cùng một utterance, có cần chia log-likelihood cho T để chọn model tốt nhất không? Tại sao?
- **Khi nhận dạng (chọn mô hình tốt nhất cho CÙNG 1 utterance):** **KHÔNG CẦN THIẾT**.  
  Bởi vì với cùng một file kiểm thử $X$, độ dài chuỗi quan sát $T$ là một hằng số giống nhau đối với tất cả các mô hình ứng viên $\lambda_w$:
  $$\arg\max_w \frac{\log P(X \mid \lambda_w)}{T} \equiv \arg\max_w \log P(X \mid \lambda_w)$$
- **Khi nào CẦN chia cho T:** Khi cần so sánh chất lượng âm học giữa các câu thoại có độ dài khác nhau, hoặc khi cần thiết lập một **ngưỡng từ chối tuyệt đối (Rejection Threshold)** để phát hiện từ ngoài từ điển (Unknown) không phụ thuộc vào độ dài file.

---

### Câu 9: So sánh warping path của DTW với state path của Viterbi.

| Tiêu chí | Warping Path (DTW) | State Path (Viterbi HMM) |
| :--- | :--- | :--- |
| **Bản chất đường đi** | Ánh xạ trực tiếp giữa frame $t_X$ của file test và frame $t_Y$ của file mẫu template chuẩn. | Ánh xạ giữa frame $t_X$ của chuỗi quan sát và trạng thái ẩn $q_j$ trong không gian mô hình xác suất. |
| **Không gian ánh xạ** | Lưới điểm thời gian 2 chiều $T_1 \times T_2$. | Đồ thị lưới Trellis thời gian - trạng thái $T \times N_{states}$. |
| **Mục tiêu tối ưu** | Cực tiểu hóa tổng khoảng cách hình học Euclidean tích lũy. | Cực đại hóa xác suất đồng thời $P(X, Q \mid \lambda)$. |
| **Tính khái quát hóa** | Dựa trên 1 mẫu template duy nhất, rất nhạy cảm với biến thiên người nói và nhiễu. | Đại diện cho phân bố thống kê học từ nhiều người nói, có độ bền vững cao. |

---

### Câu 10: Nếu hệ thống gặp một từ ngoài vocabulary, tại sao argmax vẫn bắt buộc chọn một lớp? Hãy đề xuất reject criterion.
- **Tại sao argmax vẫn chọn một lớp:**  
  Vì phép toán $\hat{w} = \arg\max_{w \in \mathcal{V}} P(X \mid \lambda_w)$ chỉ thực hiện so sánh tương đối giữa các phần tử trong tập từ vựng $\mathcal{V}$. Ngay cả khi toàn bộ các mô hình đều cho xác suất rất thấp (do gặp âm thanh lạ hoặc từ chưa từng học), mô hình nào có điểm "ít âm nhất" vẫn sẽ được chọn làm kết quả dự đoán.
- **Đề xuất tiêu chuẩn từ chối (Reject Criterion):**
  1. **Ngưỡng điểm âm học chuẩn hóa (Normalized Likelihood Threshold):**  
     Tính điểm trung bình mỗi frame $\bar{s} = \frac{\log P(X \mid \lambda_{\text{best}})}{T}$. Nếu $\bar{s} < \theta_{\text{reject}}$ (dựa theo thực nghiệm E6 chọn $\theta_{\text{reject}} = -36.80$), hệ thống sẽ từ chối và gán nhãn `UNKNOWN`.
  2. **Ngưỡng biên độ phân tách (Margin Threshold):**  
     Tính khoảng cách giữa hai ứng viên dẫn đầu $\Delta = \log P(X \mid \lambda_{\text{top1}}) - \log P(X \mid \lambda_{\text{top2}})$. Nếu $\Delta < \Delta_{\text{min}}$ (ví dụ $\Delta_{\text{min}} = 10.0$), thể hiện sự tranh chấp không rõ ràng giữa các lớp, hệ thống sẽ yêu cầu người dùng phát âm lại thay vì đưa ra quyết định sai lầm.
