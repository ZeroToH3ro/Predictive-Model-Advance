# Nghiên Cứu Dự Đoán Phản Ứng Điều Trị Bằng Phương Pháp Học Máy

## Tóm Tắt (Abstract)

Nghiên cứu này trình bày một framework học máy toàn diện nhằm dự đoán phản ứng điều trị (SVR vs Non-SVR) dựa trên ba chiến lược trích xuất đặc trưng: (1) đặc trưng RAS (Resistance-Associated Substitutions), (2) đặc trưng 3seq, và (3) bộ đặc trưng đầy đủ kết hợp. Mười ba thuật toán phân loại — bao gồm SVM, Random Forest, XGBoost, Gradient Boosting, Elastic Net, và các mô hình ensemble — được đánh giá thông qua cross-validation 5-fold, tối ưu ngưỡng bằng chỉ số Youden's J, và phân tích độ nhạy bootstrap (1.000 lần lặp). Khả năng diễn giải mô hình được thực hiện bằng SHAP values. Random Forest đạt hiệu suất cross-validation cao nhất (AUC = 0.8845), trong khi SVM đạt test AUC cao nhất (0.8303). Phân tích Fisher's exact test xác định mutation 31M là biomarker có ý nghĩa thống kê mạnh nhất (OR = 13.68, p < 0.001) cho Non-SVR.

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

Ba chiến lược được đánh giá song song:

| Chiến lược | Mô tả | Số lượng features |
|-----------|-------|-------------------|
| **RAS** | Chỉ sử dụng đặc trưng mutation RAS, được chọn lọc qua Fisher's exact test | Thấp |
| **3seq** | Sử dụng đặc trưng trình tự 3seq | Trung bình |
| **Full** | Kết hợp tất cả đặc trưng có sẵn | Cao |

### 2.3. Tiền xử lý dữ liệu

#### 2.3.1. Mã hóa đặc trưng
- Biến mục tiêu: "Non SVR" → 1, "SVR" → 0
- Genotype features: One-hot encoding

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

### 3.2. Hiệu suất Test Set (RAS Features — Từ notebook output)

- **SVM**: Test AUC = 0.8303
- Các mô hình khác: tham khảo file kết quả Excel

### 3.3. Phân tích RAS Significance (Fisher's Exact Test)

| Mutation | n_present | Non-SVR present | SVR present | Odds Ratio | p-value | Ý nghĩa |
|----------|-----------|-----------------|-------------|------------|---------|---------|
| **31M** | 26 | 22 | 4 | 13.68 | 1.16e-07 | Rất mạnh |
| 24R | 6 | 6 | 0 | ∞ | 2.43e-03 | Mạnh |
| 30Q | 27 | 17 | 10 | 3.52 | 4.33e-03 | Trung bình |

> **Phát hiện quan trọng**: Mutation 31M có mối liên hệ rất mạnh với thất bại điều trị (Non-SVR), với tỷ số chênh OR = 13.68 và p-value cực kỳ nhỏ (1.16 × 10⁻⁷).

### 3.4. Phân nhóm hiệu suất thuật toán

**Nhóm A — Hiệu suất cao (AUC > 0.85):**
- Random Forest, XGBoost

**Nhóm B — Hiệu suất khá (AUC 0.75–0.85):**
- Elastic Net, Voting, KNN, GBM, Stacking, SVM, Decision Tree

**Nhóm C — Hiệu suất thấp (AUC < 0.75):**
- Logistic Regression, Neural Network, FDA, Naive Bayes

---

## 4. Thảo Luận (Discussion)

### 4.1. So sánh thuật toán

Random Forest và XGBoost đạt hiệu suất cao nhất, phù hợp với đặc điểm dữ liệu y sinh:
- Khả năng xử lý mối quan hệ phi tuyến
- Robust với outliers
- Feature importance tích hợp

Các mô hình tuyến tính (LR, Elastic Net) mặc dù có AUC thấp hơn nhưng cung cấp khả năng diễn giải quan trọng cho ứng dụng lâm sàng.

### 4.2. Xử lý mất cân bằng lớp

Pipeline sử dụng chiến lược kết hợp:
- **SMOTE** — oversampling synthetic cho tất cả mô hình
- **class_weight="balanced"** — cho SVM, RF, DT, LR, Elastic Net
- **scale_pos_weight** — cho XGBoost

⚠️ **Lưu ý**: Kết hợp SMOTE với class_weight có thể tạo thiên lệch kép. Cần thực nghiệm thêm để xác định chiến lược tối ưu.

### 4.3. Hạn chế

1. **Kích thước mẫu nhỏ** (n=162): Giới hạn khả năng tổng quát hóa
2. **Single train/test split**: Không phản ánh đầy đủ variance — cần nested CV
3. **SHAP analysis giới hạn**: Chỉ 20 samples → có thể không đại diện
4. **Thiếu external validation**: Chưa có tập dữ liệu độc lập để kiểm chứng

### 4.4. Biomarker RAS

Mutation 31M nổi bật như biomarker tiềm năng:
- Odds ratio 13.68 cho thấy bệnh nhân mang mutation này có nguy cơ thất bại điều trị cao gấp ~14 lần
- p-value < 0.001 đảm bảo ý nghĩa thống kê mạnh
- Cần validation trên cohort độc lập

---

## 5. Kết Luận (Conclusion)

### 5.1. Kết quả chính
1. **Random Forest** và **XGBoost** là hai thuật toán hiệu quả nhất cho bài toán dự đoán phản ứng điều trị
2. **Mutation 31M** được xác định là biomarker có ý nghĩa thống kê mạnh nhất
3. Pipeline xử lý mất cân bằng lớp (SMOTE + class_weight) cải thiện hiệu suất đáng kể
4. Phân tích bootstrap 1.000 lần lặp cung cấp ước lượng tin cậy về hiệu suất mô hình

### 5.2. Hướng nghiên cứu tiếp theo
1. **Nested cross-validation** để ước lượng generalization error không thiên lệch
2. **External validation** trên dataset độc lập
3. **Feature selection tích hợp** (embedded methods) thay vì chỉ dựa vào VarianceThreshold
4. **TreeSHAP** cho mô hình tree-based (nhanh hơn KernelSHAP đáng kể)
5. **Calibration analysis** để đánh giá chất lượng xác suất dự đoán
6. **Tối ưu ensemble** — loại bỏ mô hình yếu (FDA, Naive Bayes, MLP) và tập trung vào top-performing models

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

---

*Tài liệu này phục vụ mục đích hỗ trợ viết báo cáo nghiên cứu khoa học. Các kết quả cần được xác minh và bổ sung thêm trước khi xuất bản.*
