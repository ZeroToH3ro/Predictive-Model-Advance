# Nghiên Cứu Dự Đoán Phản Ứng Điều Trị Bằng Phương Pháp Học Máy

## Tóm Tắt (Abstract)

Nghiên cứu này trình bày một framework học máy toàn diện nhằm dự đoán phản ứng điều trị (SVR vs Non-SVR) dựa trên ba chiến lược trích xuất đặc trưng: (1) đặc trưng RAS (Resistance-Associated Substitutions), (2) đặc trưng 3seq (trình tự 3 vùng gene NS3/NS5A/NS5B), và (3) bộ đặc trưng đầy đủ (full-length sequences). Mười ba thuật toán phân loại — bao gồm SVM, Random Forest, XGBoost, Gradient Boosting, Elastic Net, và các mô hình ensemble — được đánh giá thông qua cross-validation 5-fold, tối ưu ngưỡng bằng chỉ số Youden's J, và phân tích độ nhạy bootstrap (1.000 lần lặp). Khả năng diễn giải mô hình được thực hiện bằng SHAP values.

**Kết quả nổi bật theo chiến lược đặc trưng:**
- **RAS**: Random Forest đạt CV AUC cao nhất (0.8845), SVM đạt test AUC 0.8303
- **3seq Amino Acid**: Random Forest dẫn đầu (CV AUC = 0.8824), bootstrap AUC trung bình đạt 0.9021
- **3seq Nucleotide**: XGBoost dẫn đầu (CV AUC = 0.8973), SVM bootstrap AUC đạt 0.9093
- **Full Amino Acid**: Elastic Net đạt CV AUC cao nhất toàn bộ nghiên cứu (0.9045)
- **Full Nucleotide**: Elastic Net và Logistic Regression cùng dẫn đầu (CV AUC = 0.8935)

Phân tích Fisher's exact test xác định mutation 31M là biomarker có ý nghĩa thống kê mạnh nhất (OR = 13.68, p < 0.001) cho Non-SVR.

---

## 1. Giới Thiệu (Introduction)

### 1.1. Bối cảnh nghiên cứu
Dự đoán phản ứng điều trị là một thách thức quan trọng trong y học cá nhân hóa. Việc xác định sớm bệnh nhân có khả năng không đáp ứng điều trị (Non-SVR) cho phép can thiệp lâm sàng kịp thời.

### 1.2. Mục tiêu
1. Xây dựng các mô hình dự đoán phân loại nhị phân SVR/Non-SVR
2. So sánh hiệu suất giữa các chiến lược đặc trưng khác nhau (RAS, 3seq, full)
3. Đánh giá độ ổn định của mô hình thông qua phân tích bootstrap
4. Xác định biomarkers có ý nghĩa thống kê

### 1.3. Đóng góp chính
- Framework đánh giá toàn diện với 13 thuật toán
- So sánh 3 chiến lược trích xuất đặc trưng
- Phân tích bootstrap 1.000 lần lặp cho mỗi mô hình
- Tích hợp SHAP explainability vào quy trình

---

## 2. Phương Pháp (Methodology)

### 2.1. Dữ liệu

| Thông số | Giá trị |
|----------|---------|
| Tổng số mẫu | 162 |
| Lớp SVR (0) | 101 (62.3%) |
| Lớp Non-SVR (1) | 61 (37.7%) |
| Tỷ lệ mất cân bằng | 1.66:1 |

### 2.2. Chiến lược đặc trưng

Ba chiến lược được đánh giá song song, mỗi chiến lược xử lý cả hai dạng mã hóa (Amino Acid và Nucleotide) khi áp dụng:

| Chiến lược | Mô tả | Mã hóa | Số lượng features |
|-----------|-------|--------|-------------------|
| **RAS** | Đặc trưng mutation RAS, chọn lọc qua Fisher's exact test | Numerical + one-hot genotype | Thấp |
| **3seq** | Trình tự 3 vùng gene (NS3, NS5A, NS5B) riêng biệt | Aa: integers 1–20; Nu: integers 1–4 | Trung bình |
| **Full** | Trình tự đầy đủ (full-length) từ cột 'Full aa' / 'Full Nu' | Aa: integers 1–20; Nu: integers 1–4 | Cao |

> **Lưu ý**: 3seq và Full đều sử dụng `pad_sequences()` (post padding, value=0) để đồng nhất chiều dài features. 3seq ghép nối 3 vùng gene, trong khi Full sử dụng trình tự nguyên vẹn.

### 2.3. Tiền xử lý dữ liệu

#### 2.3.1. Mã hóa đặc trưng
- Biến mục tiêu: "Non SVR" → 1, "SVR" → 0 (mapping: `{'Yes': 1, 'No': 0, 'SVR': 0, 'Non SVR': 1}`)
- **RAS**: Genotype features → One-hot encoding
- **3seq**: Amino acid → integers 1–20 (alphabet "ACDEFGHIKLMNPQRSTVWY"); Nucleotide → integers 1–4 ({A:1, C:2, G:3, T:4})
- **Full**: Tương tự 3seq nhưng từ trình tự full-length

#### 2.3.2. Pipeline tiền xử lý (cho mỗi mô hình)
```
VarianceThreshold(threshold=0.0) → StandardScaler() → SMOTE(random_state=42) → FiniteClipper()
```

| Bước | Chức năng |
|------|-----------|
| VarianceThreshold | Loại bỏ đặc trưng không đổi |
| StandardScaler | Chuẩn hóa (μ=0, σ=1) |
| SMOTE | Oversampling lớp thiểu số bằng synthetic samples |
| FiniteClipper | Xử lý giá trị vô hạn, thay thế NaN bằng median |

#### 2.3.3. Chia dữ liệu
- **Tỷ lệ**: 60% train / 40% test
- **Phân tầng**: Stratified splitting (`stratify=y`)
- **Random seed**: 42

### 2.4. Thuật toán phân loại

Mười ba thuật toán được đánh giá, bao gồm:

**Mô hình đơn:**

| Thuật toán | Đặc điểm chính |
|-----------|----------------|
| SVM (RBF kernel) | + PCA(0.95 variance), class_weight="balanced" |
| Elastic Net | LogisticRegression với L1+L2 regularization |
| Random Forest | class_weight="balanced", n_estimators ∈ {300, 500} |
| GBM | Gradient Boosting tuần tự |
| Decision Tree | class_weight="balanced" |
| FDA (QDA) | Quadratic Discriminant Analysis |
| KNN | K=3,5,7 với weights ∈ {"uniform", "distance"} |
| Logistic Regression | Baseline tuyến tính |
| Naive Bayes | Gaussian Naive Bayes |
| MLP | Neural network 1 hidden layer |
| XGBoost | scale_pos_weight tự động, subsample + colsample |

**Mô hình ensemble:**

| Thuật toán | Cơ chế |
|-----------|--------|
| Voting Classifier | Soft voting: LR + RF + GBM |
| Stacking Classifier | Meta-learner LR, base: LR + RF + GBM |

### 2.5. Tối ưu siêu tham số

- **Phương pháp**: GridSearchCV
- **Cross-validation**: 5-fold
- **Metric tối ưu**: ROC-AUC
- **Ngưỡng quyết định**: Tối ưu bằng Youden's J statistic (`J = max(TPR - FPR)`)

### 2.6. Đánh giá mô hình

#### Metrics đánh giá
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- ROC-AUC

#### Phân tích độ nhạy Bootstrap
- **Số lần lặp**: 1.000
- **Phương pháp**: Resampling có hoàn lại từ tập test
- **Output**: Mean, Standard Deviation, 95% Confidence Interval

#### Khả năng diễn giải
- **SHAP KernelExplainer**: Background=50, Samples=20
- **Output**: Summary plot (bee swarm), Heatmap

---

## 3. Kết Quả (Results)

### 3.1. Hiệu suất Cross-Validation (RAS Features)

| Thuật toán | Best CV AUC | Xếp hạng |
|-----------|-------------|----------|
| Random Forest | 0.8845 | 1 |
| XGBoost | 0.8598 | 2 |
| Elastic Net | 0.8199 | 3 |
| Voting Classifier | 0.8176 | 4 |
| KNN | 0.8140 | 5 |
| GBM | 0.8131 | 6 |
| Stacking Classifier | 0.8104 | 7 |
| SVM | 0.7911 | 8 |
| Decision Tree | 0.7756 | 9 |
| Logistic Regression | 0.7482 | 10 |
| Neural Network | 0.7205 | 11 |
| FDA (QDA) | 0.7009 | 12 |
| Naive Bayes | 0.6896 | 13 |

### 3.2. Hiệu suất Cross-Validation (3seq Features)

#### 3seq Amino Acid

| Thuật toán | Best CV AUC | Xếp hạng |
|-----------|-------------|----------|
| Random Forest | 0.8824 | 1 |
| Stacking Classifier | 0.8810 | 2 |
| Neural Network | 0.8726 | 3 |
| Logistic Regression | 0.8685 | 4 |
| Elastic Net | 0.8658 | 5 |
| Voting Classifier | 0.8655 | 6 |
| GBM | 0.8601 | 7 |
| XGBoost | 0.8488 | 8 |
| SVM | 0.8411 | 9 |
| KNN | 0.8015 | 10 |
| Decision Tree | 0.7411 | 11 |
| Naive Bayes | 0.6649 | 12 |
| FDA (QDA) | 0.5756 | 13 |

#### 3seq Nucleotide

| Thuật toán | Best CV AUC | Xếp hạng |
|-----------|-------------|----------|
| XGBoost | 0.8973 | 1 |
| Elastic Net | 0.8884 | 2 |
| Logistic Regression | 0.8863 | 3 |
| Stacking Classifier | 0.8815 | 4 |
| Neural Network | 0.8762 | 5 |
| Random Forest | 0.8652 | 6 |
| Voting Classifier | 0.8628 | 7 |
| SVM | 0.8577 | 8 |
| GBM | 0.8354 | 9 |
| KNN | 0.7789 | 10 |
| Naive Bayes | 0.7583 | 11 |
| FDA (QDA) | 0.7140 | 12 |
| Decision Tree | 0.6499 | 13 |

### 3.3. Hiệu suất Cross-Validation (Full Features)

#### Full Amino Acid

| Thuật toán | Best CV AUC | Xếp hạng |
|-----------|-------------|----------|
| Elastic Net | **0.9045** | 1 |
| Logistic Regression | 0.9000 | 2 |
| Random Forest | 0.8997 | 3 |
| Neural Network | 0.8932 | 4 |
| Stacking Classifier | 0.8824 | 5 |
| Voting Classifier | 0.8804 | 6 |
| XGBoost | 0.8708 | 7 |
| GBM | 0.8414 | 8 |
| SVM | 0.8342 | 9 |
| KNN | 0.8052 | 10 |
| Decision Tree | 0.7204 | 11 |
| Naive Bayes | 0.6649 | 12 |
| FDA (QDA) | 0.4667 | 13 |

#### Full Nucleotide

| Thuật toán | Best CV AUC | Xếp hạng |
|-----------|-------------|----------|
| Elastic Net | 0.8935 | 1 |
| Logistic Regression | 0.8935 | 2 |
| Stacking Classifier | 0.8890 | 3 |
| Random Forest | 0.8845 | 4 |
| Voting Classifier | 0.8777 | 5 |
| Neural Network | 0.8759 | 6 |
| XGBoost | 0.8726 | 7 |
| GBM | 0.8491 | 8 |
| SVM | 0.8443 | 9 |
| KNN | 0.7923 | 10 |
| FDA (QDA) | 0.7446 | 11 |
| Naive Bayes | 0.7351 | 12 |
| Decision Tree | 0.7116 | 13 |

### 3.4. Hiệu suất Test Set

| Notebook | Model | Test AUC | Ghi chú |
|----------|-------|----------|---------|
| RAS | SVM | 0.8303 | Tốt nhất trên test |
| 3seq Aa | SVM | 0.8105 | Ổn định |
| 3seq Nu | SVM | 0.9096 | Cao nhất toàn bộ test AUC |
| Full Aa | SVM | 0.1209 | ⚠️ Anomaly — overfitting |
| Full Nu | SVM | 0.1199 | ⚠️ Anomaly — overfitting |

> **Phát hiện quan trọng**: SVM thất bại nghiêm trọng trên Full features (test AUC ~0.12, predict tất cả positive). Nguyên nhân có thể là PCA(0.95) trên full-length sequences gây overfitting.

### 3.5. Phân tích RAS Significance (Fisher's Exact Test)

| Mutation | n_present | Non-SVR present | SVR present | Odds Ratio | p-value | Ý nghĩa |
|----------|-----------|-----------------|-------------|------------|---------|---------|
| **31M** | 26 | 22 | 4 | 13.68 | 1.16e-07 | Rất mạnh |
| 24R | 6 | 6 | 0 | ∞ | 2.43e-03 | Mạnh |
| 30Q | 27 | 17 | 10 | 3.52 | 4.33e-03 | Trung bình |

> **Phát hiện quan trọng**: Mutation 31M có mối liên hệ rất mạnh với thất bại điều trị (Non-SVR), với tỷ số chênh OR = 13.68 và p-value cực kỳ nhỏ (1.16 × 10⁻⁷).

### 3.6. Phân nhóm hiệu suất thuật toán (tổng hợp tất cả chiến lược)

**Nhóm A — Hiệu suất cao ổn định (CV AUC > 0.85 trên đa số chiến lược):**
- Random Forest, Elastic Net, XGBoost, Logistic Regression (trên 3seq/Full)

**Nhóm B — Hiệu suất khá (CV AUC 0.75–0.85):**
- GBM, Voting, Stacking, Neural Network, SVM, KNN

**Nhóm C — Hiệu suất thấp (CV AUC < 0.75 trên một hoặc nhiều chiến lược):**
- Decision Tree, FDA (QDA), Naive Bayes

### 3.7. So sánh hiệu suất tốt nhất theo chiến lược đặc trưng

| Chiến lược | Best Model | Best CV AUC | Best Bootstrap AUC | Best Model (Bootstrap) |
|-----------|-----------|-------------|--------------------|-----------------------|
| RAS | Random Forest | 0.8845 | — | — |
| 3seq Aa | Random Forest | 0.8824 | 0.9101 | GBM |
| 3seq Nu | XGBoost | 0.8973 | 0.9093 | SVM |
| Full Aa | Elastic Net | **0.9045** | 0.9156 | GBM |
| Full Nu | Elastic Net/LR | 0.8935 | 0.8995 | Random Forest |

> **Nhận xét**: Full Amino Acid features cho CV AUC cao nhất toàn bộ nghiên cứu (0.9045 — Elastic Net), trong khi 3seq Nucleotide cho test performance tốt nhất (SVM test AUC = 0.9096).

---

## 4. Thảo Luận (Discussion)

### 4.1. So sánh thuật toán

**Theo chiến lược RAS**: Random Forest và XGBoost đạt hiệu suất cao nhất, phù hợp với đặc điểm dữ liệu mutation (ít features, rời rạc).

**Theo chiến lược 3seq**: Random Forest dẫn đầu trên Amino Acid, XGBoost dẫn đầu trên Nucleotide. Neural Network cải thiện đáng kể (CV AUC 0.87 vs 0.72 trên RAS) — sequence data phù hợp hơn cho deep learning.

**Theo chiến lược Full**: Mô hình tuyến tính (Elastic Net, Logistic Regression) bất ngờ đạt hiệu suất cao nhất (CV AUC ~0.90), vượt trội các mô hình phi tuyến. Điều này gợi ý:
- Full-length sequence features chứa tín hiệu tuyến tính mạnh
- Regularization (L1+L2) hiệu quả trong xử lý chiều cao
- Tree-based models có thể bị noise từ nhiều features không liên quan

⚠️ **SVM Anomaly trên Full features**: SVM thất bại nghiêm trọng (test AUC ~0.12) mặc dù CV AUC ~0.84. PCA(0.95) trên full-length sequences có thể tạo components không representative, gây overfitting.

### 4.2. Xử lý mất cân bằng lớp

Pipeline sử dụng chiến lược kết hợp:
- **SMOTE** — oversampling synthetic cho tất cả mô hình
- **class_weight="balanced"** — cho SVM, RF, DT, LR, Elastic Net
- **scale_pos_weight** — cho XGBoost

⚠️ **Lưu ý**: Kết hợp SMOTE với class_weight có thể tạo thiên lệch kép. Cần thực nghiệm thêm để xác định chiến lược tối ưu.

### 4.3. So sánh chiến lược đặc trưng

| Khía cạnh | RAS | 3seq | Full |
|-----------|-----|------|------|
| Best CV AUC | 0.8845 | 0.8973 (Nu) | **0.9045** (Aa) |
| Mô hình tốt nhất | RF, XGBoost | RF (Aa), XGBoost (Nu) | Elastic Net, LR |
| Mô hình tuyến tính | Trung bình | Tốt (0.87–0.89) | Rất tốt (0.90) |
| Neural Network | Yếu (0.72) | Tốt (0.87) | Tốt (0.89) |
| SVM stability | Ổn định | Ổn định | ⚠️ Thất bại |
| FDA viability | Yếu (0.70) | Rất yếu (0.58) | Thất bại (0.47) |
| Ưu điểm đặc trưng | Biomarker discovery | Cân bằng tính-năng/hiệu-suất | CV AUC cao nhất |
| Nhược điểm | Ít features | Padding noise | Chiều rất cao, SVM fail |

### 4.4. Hạn chế

1. **Kích thước mẫu nhỏ** (n=162): Giới hạn khả năng tổng quát hóa
2. **Single train/test split**: Không phản ánh đầy đủ variance — cần nested CV
3. **SHAP analysis giới hạn**: Chỉ 20 samples → có thể không đại diện
4. **Thiếu external validation**: Chưa có tập dữ liệu độc lập để kiểm chứng
5. **SVM anomaly trên Full features**: Cần điều tra PCA interaction với full-length sequences
6. **FDA/Naive Bayes không phù hợp**: Giả định phân phối vi phạm trên tất cả chiến lược

### 4.5. Biomarker RAS

Mutation 31M nổi bật như biomarker tiềm năng:
- Odds ratio 13.68 cho thấy bệnh nhân mang mutation này có nguy cơ thất bại điều trị cao gấp ~14 lần
- p-value < 0.001 đảm bảo ý nghĩa thống kê mạnh
- Cần validation trên cohort độc lập

---

## 5. Kết Luận (Conclusion)

### 5.1. Kết quả chính
1. **Elastic Net** đạt CV AUC cao nhất toàn bộ nghiên cứu (0.9045) trên Full Amino Acid features
2. **Random Forest** và **XGBoost** là hai thuật toán hiệu quả nhất trên RAS và 3seq features
3. **Mô hình tuyến tính** (Elastic Net, LR) bất ngờ vượt trội trên Full features, gợi ý tín hiệu tuyến tính mạnh trong full-length sequences
4. **Nucleotide features** nhìn chung cho kết quả test tốt hơn Amino Acid (3seq Nu SVM test AUC = 0.9096)
5. **Mutation 31M** được xác định là biomarker có ý nghĩa thống kê mạnh nhất (OR = 13.68, p < 0.001)
6. **FDA, Naive Bayes** hiệu suất thấp nhất quán trên tất cả chiến lược — nên loại bỏ
7. **Neural Network** cải thiện đáng kể trên sequence data (0.87–0.89) so với RAS (0.72)
8. **SVM thất bại** trên Full features (test AUC ~0.12) — PCA anomaly cần điều tra

### 5.2. Hướng nghiên cứu tiếp theo
1. **Nested cross-validation** để ước lượng generalization error không thiên lệch
2. **External validation** trên dataset độc lập
3. **Feature selection tích hợp** (embedded methods) thay vì chỉ dựa vào VarianceThreshold
4. **TreeSHAP** cho mô hình tree-based (nhanh hơn KernelSHAP đáng kể)
5. **Calibration analysis** để đánh giá chất lượng xác suất dự đoán
6. **Tối ưu ensemble** — loại bỏ mô hình yếu (FDA, Naive Bayes) và tập trung vào top-performing models
7. **Điều tra SVM anomaly** trên Full features — thử bỏ PCA hoặc giảm n_components
8. **K-mer features** — thay thế integer encoding bằng k-mer counting có thể cải thiện hiệu suất
9. **So sánh 3seq vs Full trên cùng mô hình** — xác định chiến lược tối ưu cho từng thuật toán

---

## 6. Phương Pháp Luận Bổ Sung (Supplementary)

### 6.1. Custom Transformer — FiniteClipper

```python
class FiniteClipper(BaseEstimator, TransformerMixin):
    """
    Xử lý giá trị vô hạn và NaN trong pipeline:
    - +Inf → max(finite values)
    - -Inf → min(finite values)
    - NaN → median(finite values)
    """
```

### 6.2. Công thức Youden's J Index
$$
J = \max_t \left( \text{TPR}(t) - \text{FPR}(t) \right)
$$
- $t$: ngưỡng quyết định
- TPR: True Positive Rate (Sensitivity)
- FPR: False Positive Rate (1 - Specificity)

### 6.3. SMOTE (Synthetic Minority Over-sampling Technique)
- Tạo synthetic samples bằng cách nội suy giữa các samples lớp thiểu số
- Áp dụng chỉ trong training (không áp dụng cho test set)
- Sử dụng `imblearn.pipeline.Pipeline` để đảm bảo đúng thực hành

### 6.4. Bootstrap Confidence Interval
$$
\text{CI}_{95\%} = \left[ P_{2.5}, P_{97.5} \right]
$$
- Percentile method
- 1.000 resamples có hoàn lại từ test set

---

## 7. Tài Liệu Tham Khảo Phần Mềm

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| scikit-learn | ≥1.0 | Pipeline, models, metrics |
| imbalanced-learn | ≥0.8 | SMOTE, imblearn Pipeline |
| xgboost | ≥1.5 | XGBoost classifier |
| shap | ≥0.40 | Model explainability |
| numpy | ≥1.20 | Numerical computing |
| pandas | ≥1.3 | Data manipulation |
| matplotlib | ≥3.4 | Visualization |
| tensorflow/keras | ≥2.0 | pad_sequences cho 3seq/Full encoding |

---

*Tài liệu này phục vụ mục đích hỗ trợ viết báo cáo nghiên cứu khoa học. Các kết quả cần được xác minh và bổ sung thêm trước khi xuất bản.*
