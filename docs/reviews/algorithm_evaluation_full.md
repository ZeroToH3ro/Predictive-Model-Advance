# Đánh Giá Toàn Diện Các Thuật Toán — Predictive_model_full.ipynb

## 1. Tổng Quan

Tài liệu này đánh giá toàn diện các thuật toán Machine Learning được sử dụng trong notebook **Predictive_model_full.ipynb**, sử dụng **bộ đặc trưng đầy đủ** — toàn bộ trình tự sinh học (full-length sequences) thay vì chỉ 3 vùng gene riêng lẻ.

**Mục tiêu dự đoán**: Phân loại phản ứng điều trị — SVR (nhãn 0) vs Non-SVR (nhãn 1).

**Phân bố lớp**: `{0: 101, 1: 61}` — Mất cân bằng lớp (~62% vs ~38%).

**Đặc điểm riêng**: Notebook Full sử dụng **trình tự đầy đủ** (full-length) thay vì tách thành 3 vùng gene riêng biệt:
- **Amino Acid (Aa)**: Cột 'Full aa' — mã hóa trình tự amino acid đầy đủ → integers 1–20
- **Nucleotide (Nu)**: Cột 'Full Nu' — mã hóa trình tự nucleotide đầy đủ → integers 1–4

---

## 2. Kiến Trúc Pipeline

### 2.1. Pipeline tiền xử lý

```
VarianceThreshold → StandardScaler → SMOTE → FiniteClipper → Model
```

### 2.2. Feature Engineering đặc thù Full

| Bước | Amino Acid | Nucleotide |
|------|-----------|-----------|
| Nguồn | Cột 'Full aa' | Cột 'Full Nu' |
| Mã hóa | 20 amino acids → integers 1–20 | 4 bases (A,C,G,T) → integers 1–4 |
| Padding | `pad_sequences()` với 'post' padding | `pad_sequences()` với 'post' padding |
| Số features | = max sequence length (Aa) | = max sequence length (Nu) |

> **Khác biệt với 3seq**: Full notebook sử dụng **một trình tự duy nhất đầy đủ** thay vì ghép nối 3 vùng gene, do đó số chiều features khác nhau.

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
| **Best CV AUC** | 0.8342 | 0.8443 |
| Test AUC | 0.1209 | 0.1199 |
| Bootstrap AUC (mean±std) | 0.1237±0.0507 | 0.1222±0.0493 |

**Nhận xét**: ⚠️ **Anomaly nghiêm trọng** — CV AUC tốt (~0.84) nhưng Test AUC cực thấp (~0.12). Mô hình dự đoán gần như tất cả mẫu vào lớp positive (Recall=1.0, Accuracy=0.37). Đây là dấu hiệu overfitting nghiêm trọng hoặc vấn đề với PCA trên full-length sequence data.

**Đánh giá**: ⭐ (1/5) — Thất bại hoàn toàn trên test set. Cần điều tra nguyên nhân.

---

### 3.2. Elastic Net (Logistic Regression)

**Cấu hình**: `penalty="elasticnet"`, `solver="saga"`, `max_iter=5000`, `class_weight="balanced"`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | **0.9045** | 0.8935 |
| Bootstrap AUC (mean±std) | 0.8811±0.0507 | 0.8682±0.0560 |

**Nhận xét**: **CV AUC cao nhất trên Amino Acid (0.9045)** — cao nhất trong tất cả 13 mô hình. Bootstrap AUC ổn định (~0.88). Regularization L1+L2 giúp xử lý tốt chiều cao features.

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất cao nhất, ổn định, diễn giải được.

---

### 3.3. Random Forest

**Cấu hình**: `class_weight="balanced"`, `n_estimators ∈ {300, 500}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8997 | 0.8845 |
| Bootstrap AUC (mean±std) | 0.8916±0.0494 | 0.8995±0.0472 |

**Nhận xét**: Hiệu suất CV cao (top 3 cho cả hai loại features). Bootstrap AUC ổn định ~0.89–0.90. Nucleotide bootstrap AUC (0.90) cao hơn CV AUC (0.88) — dấu hiệu mô hình robust.

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất vượt trội và ổn định.

---

### 3.4. GBM (Gradient Boosting Machine)

**Cấu hình**: `n_estimators ∈ {200, 400}`, `learning_rate ∈ {0.05, 0.1}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8414 | 0.8491 |
| Bootstrap AUC (mean±std) | **0.9156±0.0346** | 0.8712±0.0524 |

**Nhận xét**: Bootstrap AUC trên Amino Acid (0.9156) là **cao nhất trong tất cả mô hình**. Khoảng cách lớn giữa CV AUC (0.84) và bootstrap AUC (0.92) cho thấy mô hình generalize tốt hơn kỳ vọng. Std thấp nhất (0.0346) cho thấy độ ổn định cao.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Bootstrap performance xuất sắc.

---

### 3.5. Decision Tree

**Cấu hình**: `class_weight="balanced"`, `max_depth ∈ {3, 5, None}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.7204 | 0.7116 |
| Bootstrap AUC (mean±std) | 0.8018±0.0583 | 0.7691±0.0557 |

**Nhận xét**: Hiệu suất thấp nhưng không tệ bằng FDA. Bootstrap AUC cao hơn CV AUC đáng kể, cho thấy test set có phân bố thuận lợi hơn.

**Đánh giá**: ⭐⭐⭐ (3/5) — Hữu ích cho diễn giải, không nên dùng làm mô hình chính.

---

### 3.6. FDA (Quadratic Discriminant Analysis)

**Cấu hình**: `reg_param ∈ {0.0, 0.1, 0.5}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | **0.4667** | 0.7446 |
| Bootstrap AUC (mean±std) | 0.5438±0.0298 | 0.5521±0.0366 |

**Nhận xét**: ⚠️ **Thất bại nghiêm trọng** — Amino Acid CV AUC 0.47 (dưới random baseline 0.50 khi tính CI). Bootstrap AUC ~0.54–0.55, gần như không có khả năng phân loại. Full-length sequence features với chiều rất cao phá vỡ hoàn toàn giả định Gaussian.

**Đánh giá**: ⭐ (1/5) — Nên loại bỏ hoàn toàn.

---

### 3.7. KNN (K-Nearest Neighbors)

**Cấu hình**: `n_neighbors ∈ {3, 5, 7}`, `weights ∈ {"uniform", "distance"}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8052 | 0.7923 |
| Bootstrap AUC (mean±std) | 0.8504±0.0559 | 0.8309±0.0561 |

**Nhận xét**: Hiệu suất trung bình. Bootstrap AUC tốt hơn CV AUC. Full-length features tạo curse of dimensionality nhưng KNN vẫn hoạt động khá nhờ `weights="distance"`.

**Đánh giá**: ⭐⭐⭐ (3/5) — Hiệu suất khá, nhạy cảm với chiều cao.

---

### 3.8. Logistic Regression

**Cấu hình**: `max_iter=5000`, `class_weight="balanced"`, `C ∈ {0.01, 0.1, 1, 10}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | **0.9000** | **0.8935** |
| Bootstrap AUC (mean±std) | 0.8814±0.0521 | 0.8659±0.0562 |

**Nhận xét**: Hiệu suất rất cao — top 3 trên cả hai loại features. CV AUC 0.90 trên Amino Acid gần bằng Elastic Net. Regularization qua C giúp xử lý chiều cao hiệu quả.

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất xuất sắc cho mô hình đơn giản.

---

### 3.9. Naive Bayes (Gaussian)

**Cấu hình**: `var_smoothing ∈ {1e-9, 1e-8, 1e-7}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.6649 | 0.7351 |
| Bootstrap AUC (mean±std) | 0.7992±0.0500 | 0.7221±0.0566 |

**Nhận xét**: Hiệu suất thấp. Giả định independence giữa positions trong sequence không hợp lý. Bootstrap AUC trên Amino Acid (0.80) khá hơn CV AUC, nhưng vẫn thấp hơn phần lớn mô hình khác.

**Đánh giá**: ⭐⭐ (2/5) — Nên cân nhắc loại bỏ.

---

### 3.10. Neural Network (MLP)

**Cấu hình**: `hidden_layer_sizes ∈ {(50,), (100,)}`, `alpha ∈ {0.0001, 0.001, 0.01}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8932 | 0.8759 |
| Bootstrap AUC (mean±std) | 0.8700±0.0509 | 0.8690±0.0502 |

**Nhận xét**: Hiệu suất tốt trên cả hai loại features. CV AUC 0.89 trên Amino Acid thuộc top 5. Full-length sequence phù hợp cho MLP — đủ pattern phi tuyến để khai thác.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Hiệu suất tốt, cải thiện lớn so với RAS.

---

### 3.11. Voting Classifier

**Cấu hình**: Soft voting — LR + RF(100) + GBM(100)

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8804 | 0.8777 |
| Bootstrap AUC (mean±std) | 0.8839±0.0505 | 0.8946±0.0478 |

**Nhận xét**: Ensemble ổn định trên cả hai loại. Nucleotide bootstrap AUC (0.89) cao hơn CV AUC (0.88). Kết hợp hiệu quả giữa mô hình tuyến tính và phi tuyến.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Ensemble hiệu quả và ổn định.

---

### 3.12. Stacking Classifier

**Cấu hình**: Base: LR + RF(100) + GBM(100), Meta-learner: LogisticRegression

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8824 | 0.8890 |
| Bootstrap AUC (mean±std) | 0.8869±0.0481 | 0.8969±0.0469 |

**Nhận xét**: Hiệu suất cao và ổn định. Nucleotide CV AUC (0.89) và bootstrap AUC (0.90) đều ấn tượng. Stacking phát huy tốt trên full-length data.

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Ensemble mạnh, đặc biệt trên Nucleotide.

---

### 3.13. XGBoost

**Cấu hình**: `scale_pos_weight` tự động, `n_estimators ∈ {300, 500}`, `subsample ∈ {0.8, 1.0}`

| Metric | Amino Acid | Nucleotide |
|--------|-----------|-----------|
| **Best CV AUC** | 0.8708 | 0.8726 |
| Bootstrap AUC (mean±std) | **0.9149±0.0381** | 0.8447±0.0563 |

**Nhận xét**: Bootstrap AUC trên Amino Acid (0.9149) ngang GBM và thuộc nhóm cao nhất. CV AUC trung bình (~0.87), nhưng bootstrap cho thấy generalization tốt. Nucleotide bootstrap thấp hơn (0.84).

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Bootstrap performance xuất sắc trên Amino Acid.

---

## 4. Phát Hiện Đáng Chú Ý

### 4.1. SVM Anomaly

SVM cho thấy hiện tượng **nghiêm trọng** trên full-length features:
- CV AUC ~0.84 (bình thường)
- Test AUC ~0.12 (dưới random, ngược lại)
- Test Accuracy = 0.3692, Recall = 1.0 → mô hình dự đoán **tất cả mẫu là positive**

**Nguyên nhân có thể**:
1. PCA(0.95) trên full-length sequence tạo ra components không đại diện
2. Overfitting nghiêm trọng trong CV nhưng không generalize
3. Interaction giữa SMOTE + class_weight + PCA gây thiên lệch

### 4.2. FDA Collapse

FDA với CV AUC = 0.4667 trên Amino Acid — **dưới random baseline**. Full-length sequence features vi phạm hoàn toàn giả định phân phối Gaussian.

---

## 5. Bảng Tổng Hợp Xếp Hạng

### 5.1. Amino Acid Features

| Hạng | Thuật toán | CV AUC | Bootstrap AUC | Đánh giá |
|------|-----------|--------|---------------|----------|
| 1 | Elastic Net | **0.9045** | 0.8811 | ⭐⭐⭐⭐⭐ |
| 2 | Logistic Regression | 0.9000 | 0.8814 | ⭐⭐⭐⭐⭐ |
| 3 | Random Forest | 0.8997 | 0.8916 | ⭐⭐⭐⭐⭐ |
| 4 | Neural Network | 0.8932 | 0.8700 | ⭐⭐⭐⭐ |
| 5 | Stacking Classifier | 0.8824 | 0.8869 | ⭐⭐⭐⭐ |
| 6 | Voting Classifier | 0.8804 | 0.8839 | ⭐⭐⭐⭐ |
| 7 | XGBoost | 0.8708 | 0.9149 | ⭐⭐⭐⭐ |
| 8 | GBM | 0.8414 | 0.9156 | ⭐⭐⭐⭐ |
| 9 | SVM | 0.8342 | 0.1237 | ⭐ |
| 10 | KNN | 0.8052 | 0.8504 | ⭐⭐⭐ |
| 11 | Decision Tree | 0.7204 | 0.8018 | ⭐⭐⭐ |
| 12 | Naive Bayes | 0.6649 | 0.7992 | ⭐⭐ |
| 13 | FDA (QDA) | 0.4667 | 0.5438 | ⭐ |

### 5.2. Nucleotide Features

| Hạng | Thuật toán | CV AUC | Bootstrap AUC | Đánh giá |
|------|-----------|--------|---------------|----------|
| 1 | Elastic Net | 0.8935 | 0.8682 | ⭐⭐⭐⭐⭐ |
| 2 | Logistic Regression | 0.8935 | 0.8659 | ⭐⭐⭐⭐⭐ |
| 3 | Stacking Classifier | 0.8890 | 0.8969 | ⭐⭐⭐⭐ |
| 4 | Random Forest | 0.8845 | 0.8995 | ⭐⭐⭐⭐⭐ |
| 5 | Neural Network | 0.8759 | 0.8690 | ⭐⭐⭐⭐ |
| 6 | Voting Classifier | 0.8777 | 0.8946 | ⭐⭐⭐⭐ |
| 7 | XGBoost | 0.8726 | 0.8447 | ⭐⭐⭐⭐ |
| 8 | GBM | 0.8491 | 0.8712 | ⭐⭐⭐⭐ |
| 9 | SVM | 0.8443 | 0.1222 | ⭐ |
| 10 | KNN | 0.7923 | 0.8309 | ⭐⭐⭐ |
| 11 | FDA (QDA) | 0.7446 | 0.5521 | ⭐ |
| 12 | Naive Bayes | 0.7351 | 0.7221 | ⭐⭐ |
| 13 | Decision Tree | 0.7116 | 0.7691 | ⭐⭐ |

---

## 6. So Sánh Amino Acid vs Nucleotide

| Khía cạnh | Amino Acid | Nucleotide | Nhận xét |
|-----------|-----------|-----------|----------|
| Best CV AUC | 0.9045 (Elastic Net) | 0.8935 (Elastic Net/LR) | Amino Acid nhỉnh hơn |
| Best Bootstrap AUC | 0.9156 (GBM) | 0.8995 (RF) | Amino Acid nhỉnh hơn |
| SVM thất bại | Test AUC 0.12 | Test AUC 0.12 | Cả hai đều thất bại |
| Mô hình yếu nhất | FDA (0.4667) | Decision Tree (0.7116) | Aa có mô hình yếu hơn nhiều |
| Mô hình tuyến tính | Rất tốt (0.90) | Rất tốt (0.89) | Full features có tín hiệu tuyến tính mạnh |

---

## 7. Khuyến Nghị

### 7.1. Mô hình nên giữ lại
- **Elastic Net** — hiệu suất CV cao nhất (0.90), diễn giải được, regularization xử lý chiều cao tốt
- **Logistic Regression** — hiệu suất ngang Elastic Net, đơn giản hơn
- **Random Forest** — ổn định, bootstrap AUC ~0.90
- **Stacking Classifier** — hiệu quả trên Nucleotide

### 7.2. Mô hình nên loại bỏ
- **SVM** — ⚠️ thất bại hoàn toàn trên test set, cần điều tra trước khi tái sử dụng
- **FDA (QDA)** — CV AUC dưới random trên Amino Acid
- **Naive Bayes** — giả định independence không phù hợp

### 7.3. Cải thiện cần thiết
1. **Điều tra SVM anomaly**: Kiểm tra PCA interaction với full-length features, thử bỏ PCA hoặc giảm `n_components`
2. **Feature selection trước khi train**: Full-length sequences có nhiều noise — áp dụng feature selection (mutual information, chi-squared) trước pipeline
3. **Mô hình tuyến tính phù hợp**: CV AUC 0.90 cho Elastic Net/LR gợi ý rằng feature engineering tốt hơn (e.g., k-mer counting) có thể cải thiện thêm
4. **Nested CV**: Đặc biệt quan trọng cho full features do nguy cơ overfitting cao với chiều cao

---

*Tài liệu được tạo dựa trên phân tích kết quả từ notebook Predictive_model_full.ipynb.*
