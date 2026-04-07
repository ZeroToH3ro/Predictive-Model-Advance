# Đánh Giá Toàn Diện Các Thuật Toán — Predictive_model_3seq.ipynb

## 1. Tổng Quan

Tài liệu này đánh giá toàn diện các thuật toán Machine Learning được sử dụng trong notebook **Predictive_model_3seq.ipynb**, sử dụng đặc trưng trình tự (sequence features) từ ba vùng gene: NS3, NS5A, NS5B.

**Mục tiêu dự đoán**: Phân loại phản ứng điều trị — SVR (nhãn 0) vs Non-SVR (nhãn 1).

**Phân bố lớp**: `{0: 101, 1: 61}` — Mất cân bằng lớp (~62% vs ~38%).

**Đặc điểm riêng**: Notebook 3seq xử lý **hai loại đặc trưng trình tự** riêng biệt:
- **Amino Acid (Aa)**: Mã hóa trình tự amino acid từ 3 vùng gene → số nguyên 1–20
- **Nucleotide (Nu)**: Mã hóa trình tự nucleotide từ 3 vùng gene → số nguyên 1–4

---

## 2. Kiến Trúc Pipeline

### 2.1. Pipeline tiền xử lý (giống RAS)

```
VarianceThreshold → StandardScaler → SMOTE → FiniteClipper → Model
```

### 2.2. Feature Engineering đặc thù 3seq

| Bước | Amino Acid | Nucleotide |
|------|-----------|-----------|
| Mã hóa | 20 amino acids → integers 1–20 | 4 bases (A,C,G,T) → integers 1–4 |
| Padding | `pad_sequences()` với 'post' padding | `pad_sequences()` với 'post' padding |
| Giá trị padding | 0 | 0 |
| Vùng gene | NS3, NS5A, NS5B | NS3, NS5A, NS5B |

### 2.3. Chiến lược chia dữ liệu

- **Tỷ lệ**: Train 60% / Test 40%
- **Phân tầng**: `stratify=y`
- **Random state**: 42

---

## 3. Đánh Giá Chi Tiết Từng Thuật Toán

### 3.1. SVM (Support Vector Machine)

**Cấu hình**: Kernel RBF, `probability=True`, `class_weight="balanced"`, PCA(0.95)

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8411 | 0.8577 |
| Bootstrap AUC (mean±std) | 0.8115±0.0627 | 0.9093±0.0413 |

**Nhận xét**: SVM hoạt động tốt hơn đáng kể trên Nucleotide (bootstrap AUC 0.91) so với Amino Acid (0.81). Nucleotide features với PCA cho thấy khả năng phân tách tốt hơn.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Hiệu suất tốt, đặc biệt trên Nucleotide.

---

### 3.2. Elastic Net (Logistic Regression)

**Cấu hình**: `penalty="elasticnet"`, `solver="saga"`, `max_iter=5000`, `class_weight="balanced"`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8658 | 0.8884 |
| Bootstrap AUC (mean±std) | 0.8680±0.0530 | 0.8786±0.0524 |

**Nhận xét**: Hiệu suất ổn định trên cả hai loại features. Nucleotide CV AUC cao hơn (0.89). Khả năng diễn giải tốt.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Mô hình baseline mạnh với khả năng diễn giải.

---

### 3.3. Random Forest

**Cấu hình**: `class_weight="balanced"`, `n_estimators ∈ {300, 500}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8824 | 0.8652 |
| Bootstrap AUC (mean±std) | 0.9021±0.0457 | 0.8955±0.0492 |

**Nhận xét**: Hiệu suất CV cao nhất trên Amino Acid (0.8824). Bootstrap AUC vượt trội (>0.90 cho Aa). Ổn định trên cả hai loại features.

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất vượt trội và ổn định.

---

### 3.4. GBM (Gradient Boosting Machine)

**Cấu hình**: `n_estimators ∈ {200, 400}`, `learning_rate ∈ {0.05, 0.1}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8601 | 0.8354 |
| Bootstrap AUC (mean±std) | 0.9101±0.0465 | 0.8535±0.0552 |

**Nhận xét**: Bootstrap AUC trên Amino Acid (0.91) vượt trội CV AUC — cho thấy mô hình ổn định. Nucleotide performance thấp hơn.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Hiệu suất tốt, đặc biệt trên Amino Acid.

---

### 3.5. Decision Tree

**Cấu hình**: `class_weight="balanced"`, `max_depth ∈ {3, 5, None}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.7411 | 0.6499 |
| Bootstrap AUC (mean±std) | 0.8531±0.0449 | 0.6407±0.0735 |

**Nhận xét**: Hiệu suất thấp trên Nucleotide (CV AUC 0.65). Amino Acid khá hơn nhưng vẫn thấp. Decision Tree không phù hợp cho sequence data.

**Đánh giá**: ⭐⭐ (2/5) — Không phù hợp cho dữ liệu sequence.

---

### 3.6. FDA (Quadratic Discriminant Analysis)

**Cấu hình**: `reg_param ∈ {0.0, 0.1, 0.5}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.5756 | 0.7140 |
| Bootstrap AUC (mean±std) | 0.5423±0.0287 | 0.5204±0.0210 |

**Nhận xét**: ⚠️ Hiệu suất rất thấp, đặc biệt trên Amino Acid (CV AUC 0.58, gần random). Bootstrap AUC ~0.52–0.54 cho thấy mô hình hầu như không có khả năng phân loại. Giả định Gaussian không phù hợp cho sequence data.

**Đánh giá**: ⭐ (1/5) — Nên loại bỏ hoàn toàn.

---

### 3.7. KNN (K-Nearest Neighbors)

**Cấu hình**: `n_neighbors ∈ {3, 5, 7}`, `weights ∈ {"uniform", "distance"}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8015 | 0.7789 |
| Bootstrap AUC (mean±std) | 0.8623±0.0517 | 0.8468±0.0542 |

**Nhận xét**: Hiệu suất trung bình. Bootstrap AUC cao hơn CV AUC — dấu hiệu tốt. Tuy nhiên, nhạy cảm với chiều cao của sequence features.

**Đánh giá**: ⭐⭐⭐ (3/5) — Hiệu suất khá nhưng không nổi bật.

---

### 3.8. Logistic Regression

**Cấu hình**: `max_iter=5000`, `class_weight="balanced"`, `C ∈ {0.01, 0.1, 1, 10}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8685 | 0.8863 |
| Bootstrap AUC (mean±std) | 0.8863±0.0498 | 0.8683±0.0560 |

**Nhận xét**: Hiệu suất cao đáng ngạc nhiên cho mô hình tuyến tính. Nucleotide CV AUC (0.89) ngang ngửa Elastic Net. Khả năng diễn giải tốt.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Hiệu suất tốt bất ngờ, diễn giải dễ dàng.

---

### 3.9. Naive Bayes (Gaussian)

**Cấu hình**: `var_smoothing ∈ {1e-9, 1e-8, 1e-7}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.6649 | 0.7583 |
| Bootstrap AUC (mean±std) | 0.7667±0.0532 | 0.7018±0.0578 |

**Nhận xét**: Hiệu suất thấp. Giả định independence giữa các vị trí sequence là không hợp lý. Amino Acid thấp hơn Nucleotide trên CV nhưng ngược lại trên bootstrap.

**Đánh giá**: ⭐⭐ (2/5) — Nên cân nhắc loại bỏ.

---

### 3.10. Neural Network (MLP)

**Cấu hình**: `hidden_layer_sizes ∈ {(50,), (100,)}`, `alpha ∈ {0.0001, 0.001, 0.01}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8726 | 0.8762 |
| Bootstrap AUC (mean±std) | 0.9009±0.0441 | 0.8739±0.0483 |

**Nhận xét**: Hiệu suất tốt trên cả hai loại features. Bootstrap AUC 0.90 trên Amino Acid rất ấn tượng. MLP phù hợp hơn với sequence data so với RAS features nhờ khả năng học patterns phi tuyến.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Cải thiện đáng kể so với kết quả RAS.

---

### 3.11. Voting Classifier

**Cấu hình**: Soft voting — LR + RF(100) + GBM(100)

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8655 | 0.8628 |
| Bootstrap AUC (mean±std) | 0.9089±0.0434 | 0.8765±0.0503 |

**Nhận xét**: Ensemble ổn định. Bootstrap AUC 0.91 trên Amino Acid cho thấy kết hợp hiệu quả. Nucleotide cũng tốt.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Ensemble hiệu quả và ổn định.

---

### 3.12. Stacking Classifier

**Cấu hình**: Base: LR + RF(100) + GBM(100), Meta-learner: LogisticRegression

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8810 | 0.8815 |
| Bootstrap AUC (mean±std) | 0.9087±0.0426 | 0.8717±0.0504 |

**Nhận xét**: CV AUC cao và ổn định trên cả hai loại features (~0.88). Bootstrap AUC >0.90 trên Amino Acid. Stacking phát huy tốt hơn trên sequence data so với RAS data.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Phát huy hiệu quả trên sequence data.

---

### 3.13. XGBoost

**Cấu hình**: `scale_pos_weight` tự động, `n_estimators ∈ {300, 500}`, `subsample ∈ {0.8, 1.0}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8488 | 0.8973 |
| Bootstrap AUC (mean±std) | 0.8861±0.0461 | 0.8576±0.0540 |

**Nhận xét**: CV AUC cao nhất trên **Nucleotide (0.8973)**. Amino Acid thấp hơn (0.85). XGBoost phù hợp đặc biệt cho nucleotide sequence features.

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất cao nhất trên Nucleotide.

---

## 4. Bảng Tổng Hợp Xếp Hạng

### 4.1. Amino Acid Features

| Hạng | Thuật toán | CV AUC | Bootstrap AUC | Đánh giá |
|------|-----------|--------|---------------|----------|
| 1 | Random Forest | 0.8824 | 0.9021 | ⭐⭐⭐⭐⭐ |
| 2 | Stacking Classifier | 0.8810 | 0.9087 | ⭐⭐⭐⭐ |
| 3 | Neural Network | 0.8726 | 0.9009 | ⭐⭐⭐⭐ |
| 4 | Logistic Regression | 0.8685 | 0.8863 | ⭐⭐⭐⭐ |
| 5 | Elastic Net | 0.8658 | 0.8680 | ⭐⭐⭐⭐ |
| 6 | Voting Classifier | 0.8655 | 0.9089 | ⭐⭐⭐⭐ |
| 7 | GBM | 0.8601 | 0.9101 | ⭐⭐⭐⭐ |
| 8 | XGBoost | 0.8488 | 0.8861 | ⭐⭐⭐⭐ |
| 9 | SVM | 0.8411 | 0.8115 | ⭐⭐⭐⭐ |
| 10 | KNN | 0.8015 | 0.8623 | ⭐⭐⭐ |
| 11 | Decision Tree | 0.7411 | 0.8531 | ⭐⭐ |
| 12 | Naive Bayes | 0.6649 | 0.7667 | ⭐⭐ |
| 13 | FDA (QDA) | 0.5756 | 0.5423 | ⭐ |

### 4.2. Nucleotide Features

| Hạng | Thuật toán | CV AUC | Bootstrap AUC | Đánh giá |
|------|-----------|--------|---------------|----------|
| 1 | XGBoost | 0.8973 | 0.8576 | ⭐⭐⭐⭐⭐ |
| 2 | Elastic Net | 0.8884 | 0.8786 | ⭐⭐⭐⭐ |
| 3 | Logistic Regression | 0.8863 | 0.8683 | ⭐⭐⭐⭐ |
| 4 | Stacking Classifier | 0.8815 | 0.8717 | ⭐⭐⭐⭐ |
| 5 | Neural Network | 0.8762 | 0.8739 | ⭐⭐⭐⭐ |
| 6 | Random Forest | 0.8652 | 0.8955 | ⭐⭐⭐⭐ |
| 7 | Voting Classifier | 0.8628 | 0.8765 | ⭐⭐⭐⭐ |
| 8 | SVM | 0.8577 | 0.9093 | ⭐⭐⭐⭐ |
| 9 | GBM | 0.8354 | 0.8535 | ⭐⭐⭐ |
| 10 | KNN | 0.7789 | 0.8468 | ⭐⭐⭐ |
| 11 | Naive Bayes | 0.7583 | 0.7018 | ⭐⭐ |
| 12 | FDA (QDA) | 0.7140 | 0.5204 | ⭐ |
| 13 | Decision Tree | 0.6499 | 0.6407 | ⭐ |

---

## 5. So Sánh Amino Acid vs Nucleotide

| Khía cạnh | Amino Acid | Nucleotide | Nhận xét |
|-----------|-----------|-----------|----------|
| Best CV AUC | 0.8824 (RF) | 0.8973 (XGBoost) | Nucleotide nhỉnh hơn |
| Best Bootstrap AUC | 0.9101 (GBM) | 0.9093 (SVM) | Gần bằng nhau |
| Mô hình yếu nhất | FDA (0.5756) | Decision Tree (0.6499) | Cả hai đều có mô hình rất yếu |
| Số mô hình AUC > 0.85 | 7/13 | 8/13 | Nucleotide có nhiều mô hình mạnh hơn |
| MLP performance | 0.8726 | 0.8762 | Tương đương, cải thiện so với RAS |

---

## 6. Khuyến Nghị

### 6.1. Mô hình nên giữ lại
- **Random Forest** — hiệu suất CV cao nhất trên Amino Acid, ổn định
- **XGBoost** — hiệu suất CV cao nhất trên Nucleotide
- **Stacking Classifier** — ổn định trên cả hai loại features
- **Elastic Net / Logistic Regression** — khả năng diễn giải, hiệu suất tốt bất ngờ

### 6.2. Mô hình nên loại bỏ
- **FDA (QDA)** — bootstrap AUC ~0.52–0.54, gần random
- **Decision Tree** — hiệu suất thấp trên Nucleotide (0.65)
- **Naive Bayes** — giả định independence không phù hợp cho sequence

### 6.3. Phát hiện đáng chú ý
1. **Neural Network cải thiện** đáng kể so với RAS (0.87 vs 0.72) — sequence data phù hợp hơn cho MLP
2. **Logistic Regression** đạt hiệu suất cao bất ngờ (CV AUC ~0.87–0.89), cho thấy có tín hiệu tuyến tính mạnh trong sequence features
3. **Nucleotide features** nhìn chung cho kết quả tốt hơn Amino Acid, gợi ý rằng thông tin trình tự nucleotide chứa nhiều tín hiệu dự đoán hơn

---

*Tài liệu được tạo dựa trên phân tích kết quả từ notebook Predictive_model_3seq.ipynb.*
