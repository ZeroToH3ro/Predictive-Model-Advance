# Đánh Giá Toàn Diện Các Thuật Toán Dự Đoán Mô Hình

## 1. Tổng Quan

Tài liệu này đánh giá toàn diện các thuật toán Machine Learning được sử dụng trong ba notebooks dự đoán:

- **Predictive_model_ras.ipynb** — Sử dụng đặc trưng RAS (Resistance-Associated Substitutions)
- **Predictive_model_3seq.ipynb** — Sử dụng đặc trưng 3seq
- **Predictive_model_full.ipynb** — Sử dụng bộ đặc trưng đầy đủ (kết hợp)

**Mục tiêu dự đoán**: Phân loại phản ứng điều trị — SVR (Sustained Virological Response, nhãn 0) vs Non-SVR (nhãn 1).

**Phân bố lớp** (từ notebook RAS): `{0: 101, 1: 61}` — Mất cân bằng lớp đáng kể (~62% vs ~38%).

---

## 2. Kiến Trúc Pipeline

Tất cả ba notebooks đều sử dụng pipeline thống nhất từ `imblearn.pipeline.Pipeline`:

```
VarianceThreshold → StandardScaler → SMOTE → FiniteClipper → Model
```

### 2.1. Các bước tiền xử lý

| Bước | Mục đích | Đánh giá |
|------|----------|----------|
| **VarianceThreshold** | Loại bỏ đặc trưng có phương sai bằng 0 | ✅ Phù hợp — loại bỏ đặc trưng hằng số |
| **StandardScaler** | Chuẩn hóa dữ liệu (mean=0, std=1) | ✅ Cần thiết cho SVM, KNN, Neural Network |
| **SMOTE** | Oversampling lớp thiểu số | ⚠️ Hiệu quả nhưng cần lưu ý rủi ro overfitting |
| **FiniteClipper** | Xử lý giá trị vô hạn/NaN | ✅ Transformer tùy chỉnh rất hữu ích |

### 2.2. Chiến lược chia dữ liệu

- **Tỷ lệ**: Train 60% / Test 40% (`test_size=0.4`)
- **Phân tầng**: `stratify=y` — đảm bảo tỷ lệ lớp được duy trì
- **Random state**: 42 (tái lập kết quả)

### 2.3. Tối ưu ngưỡng quyết định

- Sử dụng chỉ số **Youden's J** (`J = TPR - FPR`) để tìm ngưỡng tối ưu
- Áp dụng cho tất cả các mô hình, không chỉ dùng ngưỡng mặc định 0.5

> **Nhận xét**: Đây là phương pháp tốt, đặc biệt khi dữ liệu mất cân bằng. Ngưỡng mặc định 0.5 thường không tối ưu cho các bài toán phân loại nhị phân với lớp không cân bằng.

---

## 3. Đánh Giá Chi Tiết Từng Thuật Toán

### 3.1. SVM (Support Vector Machine)

**Cấu hình**:
- Kernel: RBF
- `probability=True`, `class_weight="balanced"`
- Bước PCA bổ sung: `PCA(n_components=0.95)`
- **Hyperparameters**: `C ∈ {0.1, 1, 10}`, `gamma ∈ {"scale", 0.1, 0.01}`

**Kết quả (RAS)**: Best CV AUC = **0.7911**, Test AUC = **0.8303**

**Ưu điểm**:
- Hiệu quả trong không gian chiều cao
- `class_weight="balanced"` xử lý mất cân bằng lớp
- PCA giảm chiều giúp tránh curse of dimensionality

**Nhược điểm**:
- Tốc độ huấn luyện chậm với dữ liệu lớn
- Khó diễn giải kết quả
- PCA thêm vào chỉ cho SVM có thể gây ra sự khác biệt so sánh không công bằng

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Hiệu suất tốt, đặc biệt khi kết hợp với PCA.

---

### 3.2. Elastic Net (Logistic Regression)

**Cấu hình**:
- `penalty="elasticnet"`, `solver="saga"`, `max_iter=5000`
- `class_weight="balanced"`
- **Hyperparameters**: `C ∈ {0.01, 0.1, 1, 10}`, `l1_ratio ∈ {0.1, 0.5, 0.9}`

**Kết quả (RAS)**: Best CV AUC = **0.8199**

**Ưu điểm**:
- Kết hợp L1 và L2 regularization → chọn đặc trưng + ổn định
- Diễn giải được (hệ số có ý nghĩa)
- Huấn luyện nhanh

**Nhược điểm**:
- Giả định tuyến tính giữa đặc trưng và log-odds
- Có thể không nắm bắt mối quan hệ phi tuyến

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Mô hình baseline mạnh, có khả năng diễn giải.

---

### 3.3. Random Forest

**Cấu hình**:
- `class_weight="balanced"`, `random_state=42`
- **Hyperparameters**: `n_estimators ∈ {300, 500}`, `max_depth ∈ {None, 20}`, `min_samples_leaf ∈ {1, 2}`

**Kết quả (RAS)**: Best CV AUC = **0.8845** (cao nhất trong cross-validation)

**Ưu điểm**:
- Hiệu suất cao nhất trong CV
- Xử lý dữ liệu phi tuyến tốt
- Tính năng importance có ý nghĩa
- Ít nhạy cảm với outliers

**Nhược điểm**:
- Có xu hướng overfitting với dữ liệu nhỏ
- Khó diễn giải hơn mô hình tuyến tính

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất vượt trội, đặc biệt phù hợp cho dữ liệu y sinh.

---

### 3.4. GBM (Gradient Boosting Machine)

**Cấu hình**:
- `random_state=42`
- **Hyperparameters**: `n_estimators ∈ {200, 400}`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {2, 3}`

**Kết quả (RAS)**: Best CV AUC = **0.8131**

**Ưu điểm**:
- Boosting tuần tự → sửa lỗi từ mô hình trước
- Learning rate thấp giúp tránh overfitting

**Nhược điểm**:
- Không có `class_weight` — có thể bị ảnh hưởng bởi mất cân bằng lớp (SMOTE bù đắp phần nào)
- Huấn luyện chậm hơn Random Forest

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Hiệu suất tốt, ổn định.

---

### 3.5. Decision Tree

**Cấu hình**:
- `class_weight="balanced"`, `random_state=42`
- **Hyperparameters**: `max_depth ∈ {3, 5, None}`, `min_samples_leaf ∈ {1, 2}`

**Kết quả (RAS)**: Best CV AUC = **0.7756**

**Ưu điểm**:
- Dễ diễn giải nhất
- Nhanh trong huấn luyện và dự đoán

**Nhược điểm**:
- Dễ overfitting
- Hiệu suất thấp hơn ensemble methods
- Không ổn định — thay đổi nhỏ trong dữ liệu → cây hoàn toàn khác

**Đánh giá**: ⭐⭐⭐ (3/5) — Hữu ích cho diễn giải, nhưng không nên dùng làm mô hình chính.

---

### 3.6. FDA (Quadratic Discriminant Analysis)

**Cấu hình**:
- **Hyperparameters**: `reg_param ∈ {0.0, 0.1, 0.5}`

**Kết quả (RAS)**: Best CV AUC = **0.7009** (thấp nhất)

**Ưu điểm**:
- Không cần hyperparameter tuning phức tạp
- Nhanh

**Nhược điểm**:
- Giả định phân phối Gaussian cho mỗi lớp
- ⚠️ Hiệu suất thấp nhất — có thể không phù hợp với dạng dữ liệu
- Không có `class_weight`

**Đánh giá**: ⭐⭐ (2/5) — Nên cân nhắc loại bỏ khỏi ensemble.

---

### 3.7. KNN (K-Nearest Neighbors)

**Cấu hình**:
- **Hyperparameters**: `n_neighbors ∈ {3, 5, 7}`, `weights ∈ {"uniform", "distance"}`

**Kết quả (RAS)**: Best CV AUC = **0.8140**

**Ưu điểm**:
- Đơn giản, không tham số
- `weights="distance"` cải thiện hiệu suất

**Nhược điểm**:
- Nhạy cảm với curse of dimensionality
- Tốc độ dự đoán chậm với dữ liệu lớn

**Đánh giá**: ⭐⭐⭐ (3/5) — Hiệu suất khá, nhưng có thể không ổn định.

---

### 3.8. Logistic Regression

**Cấu hình**:
- `max_iter=5000`, `class_weight="balanced"`, `random_state=42`
- **Hyperparameters**: `C ∈ {0.01, 0.1, 1, 10}`

**Kết quả (RAS)**: Best CV AUC = **0.7482**

**Ưu điểm**:
- Mô hình cơ sở tốt
- Diễn giải được hoàn toàn

**Nhược điểm**:
- Giả định tuyến tính
- Hiệu suất CV thấp hơn các phương pháp ensemble

**Đánh giá**: ⭐⭐⭐ (3/5) — Baseline hữu ích để so sánh.

---

### 3.9. Naive Bayes (Gaussian)

**Cấu hình**:
- **Hyperparameters**: `var_smoothing ∈ {1e-9, 1e-8, 1e-7}`

**Kết quả (RAS)**: Best CV AUC = **0.6896**

**Ưu điểm**:
- Rất nhanh
- Hoạt động tốt với dữ liệu nhỏ

**Nhược điểm**:
- Giả định independence giữa đặc trưng (thường không đúng)
- ⚠️ Hiệu suất thấp thứ hai

**Đánh giá**: ⭐⭐ (2/5) — Nên cân nhắc loại bỏ.

---

### 3.10. Neural Network (MLP)

**Cấu hình**:
- `max_iter=1000`, `random_state=42`
- **Hyperparameters**: `hidden_layer_sizes ∈ {(50,), (100,)}`, `alpha ∈ {0.0001, 0.001, 0.01}`

**Kết quả (RAS)**: Best CV AUC = **0.7205**

**Ưu điểm**:
- Có thể nắm bắt mối quan hệ phi tuyến phức tạp

**Nhược điểm**:
- Cần nhiều dữ liệu hơn để hiệu quả
- ⚠️ Với ~162 mẫu, MLP có thể overfitting
- Khó diễn giải

**Đánh giá**: ⭐⭐ (2/5) — Không phù hợp với kích thước mẫu nhỏ.

---

### 3.11. Voting Classifier

**Cấu hình**:
- Soft voting
- Base estimators: LR + RF (100 trees) + GBM (100 trees)
- Không có hyperparameter tuning

**Kết quả (RAS)**: Best CV AUC = **0.8176**

**Ưu điểm**:
- Kết hợp nhiều mô hình → giảm variance
- Soft voting sử dụng xác suất → linh hoạt hơn hard voting

**Nhược điểm**:
- Không tuning hyperparameters cho base estimators
- Base estimators dùng `n_estimators=100` (nhỏ hơn RF chính: 300-500)

**Đánh giá**: ⭐⭐⭐⭐ (4/5) — Ensemble hiệu quả.

---

### 3.12. Stacking Classifier

**Cấu hình**:
- Base estimators: LR + RF (100 trees) + GBM (100 trees)
- Meta-learner: `LogisticRegression(max_iter=1000)`
- Không có hyperparameter tuning

**Kết quả (RAS)**: Best CV AUC = **0.8104**

**Ưu điểm**:
- Meta-learner học cách kết hợp tối ưu
- Thường vượt trội Voting trong lý thuyết

**Nhược điểm**:
- Phức tạp hơn → rủi ro overfitting cao hơn
- Với dữ liệu nhỏ, có thể không phát huy ưu thế

**Đánh giá**: ⭐⭐⭐ (3/5) — Chưa phát huy hết tiềm năng.

---

### 3.13. XGBoost

**Cấu hình**:
- `eval_metric="logloss"`, `tree_method="hist"`
- `scale_pos_weight` tính tự động từ tỷ lệ lớp
- **Hyperparameters**: `n_estimators ∈ {300, 500}`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {3, 5}`, `subsample ∈ {0.8, 1.0}`, `colsample_bytree ∈ {0.8, 1.0}`

**Kết quả (RAS)**: Best CV AUC = **0.8598**

**Ưu điểm**:
- `scale_pos_weight` xử lý mất cân bằng lớp trực tiếp
- Regularization tích hợp (L1, L2)
- Grid search rộng nhất (32 tổ hợp)

**Nhược điểm**:
- Huấn luyện chậm nhất do grid search lớn
- Có thể overfitting với dữ liệu nhỏ

**Đánh giá**: ⭐⭐⭐⭐⭐ (5/5) — Hiệu suất cao, xử lý imbalance tốt.

---

## 4. Xử Lý Mất Cân Bằng Lớp

### Phương pháp hiện tại

| Chiến lược | Mô hình áp dụng |
|-----------|-----------------|
| `SMOTE` (trong pipeline) | Tất cả |
| `class_weight="balanced"` | SVM, Elastic Net, RF, Decision Tree, LR |
| `scale_pos_weight` | XGBoost |
| Không có xử lý bổ sung | GBM, FDA, KNN, Naive Bayes, MLP, Voting, Stacking |

> **Nhận xét**: SMOTE cung cấp lớp bảo vệ cơ bản cho tất cả mô hình, nhưng việc kết hợp thêm `class_weight` cho một số mô hình tạo ra hiệu ứng "kép" có thể gây thiên lệch quá mức về lớp thiểu số.

---

## 5. Phân Tích Độ Nhạy Bootstrap

### Phương pháp
- 1000 lần resampling (có hoàn lại) từ tập test
- Tính toán 5 metrics: Accuracy, Precision, Recall, F1, AUC
- Báo cáo: mean, std, 95% CI

### Đánh giá
✅ **Ưu điểm**:
- Cung cấp ước lượng phân phối metrics → đáng tin cậy hơn single-point estimate
- 95% CI hữu ích cho báo cáo nghiên cứu
- Phát hiện mô hình không ổn định

⚠️ **Hạn chế**:
- Bootstrap từ tập test → kích thước tập test cố định
- Với tỷ lệ 40% test: ~65 mẫu → CI có thể rộng
- Một số bootstrap iterations bị bỏ qua khi chỉ có 1 lớp

---

## 6. Khả Năng Giải Thích (SHAP)

### Phương pháp
- `KernelExplainer` trên mẫu con (background=50, samples=20)
- Focus vào genotype features
- Output: Summary plot + Heatmap

### Đánh giá
✅ Cung cấp insight về đóng góp từng đặc trưng
⚠️ `KernelExplainer` chậm → giới hạn mẫu nhỏ → có thể không đại diện
💡 **Gợi ý**: Dùng `TreeExplainer` cho RF, GBM, XGBoost (nhanh hơn đáng kể)

---

## 7. Phân Tích Ý Nghĩa Thống Kê RAS

### Phương pháp
- Fisher's exact test trên RAS mutations
- Ngưỡng: `alpha=0.05`, `min_support=3`

### Kết quả đáng chú ý (từ notebook RAS)
| Mutation | n_present | fail_present | svr_present | odds_ratio | p_value |
|----------|-----------|-------------|-------------|------------|---------|
| 31M | 26 | 22 | 4 | 13.68 | 1.16e-07 |
| 24R | 6 | 6 | 0 | ∞ | 2.43e-03 |
| 30Q | 27 | 17 | 10 | 3.52 | 4.33e-03 |

> **Nhận xét**: 31M có mối liên hệ rất mạnh với Non-SVR (p < 0.001), là biomarker tiềm năng.

---

## 8. Bảng Tổng Hợp Xếp Hạng

| Hạng | Thuật toán | CV AUC (RAS) | Đánh giá | Ghi chú |
|------|-----------|--------------|----------|---------|
| 1 | Random Forest | 0.8845 | ⭐⭐⭐⭐⭐ | Hiệu suất CV cao nhất |
| 2 | XGBoost | 0.8598 | ⭐⭐⭐⭐⭐ | Xử lý imbalance tốt nhất |
| 3 | Elastic Net | 0.8199 | ⭐⭐⭐⭐ | Diễn giải tốt nhất |
| 4 | Voting Classifier | 0.8176 | ⭐⭐⭐⭐ | Ensemble ổn định |
| 5 | KNN | 0.8140 | ⭐⭐⭐ | Đơn giản nhưng hiệu quả |
| 6 | GBM | 0.8131 | ⭐⭐⭐⭐ | Boosting truyền thống |
| 7 | Stacking Classifier | 0.8104 | ⭐⭐⭐ | Chưa phát huy hết |
| 8 | SVM | 0.7911 | ⭐⭐⭐⭐ | Ổn định với PCA |
| 9 | Decision Tree | 0.7756 | ⭐⭐⭐ | Dễ diễn giải |
| 10 | Logistic Regression | 0.7482 | ⭐⭐⭐ | Baseline tốt |
| 11 | Neural Network | 0.7205 | ⭐⭐ | Không phù hợp dữ liệu nhỏ |
| 12 | FDA (QDA) | 0.7009 | ⭐⭐ | Giả định không phù hợp |
| 13 | Naive Bayes | 0.6896 | ⭐⭐ | Hiệu suất thấp nhất |

---

## 9. Khuyến Nghị

### 9.1. Mô hình nên giữ lại
- **Random Forest** — hiệu suất cao nhất, ổn định
- **XGBoost** — hiệu suất cao, xử lý imbalance tốt
- **Elastic Net / Logistic Regression** — khả năng diễn giải cho báo cáo khoa học
- **SVM** — bổ sung đa dạng phương pháp

### 9.2. Mô hình nên cân nhắc loại bỏ
- **FDA (QDA)** — hiệu suất thấp, giả định không phù hợp
- **Naive Bayes** — giả định independence quá mạnh
- **Neural Network (MLP)** — không đủ dữ liệu để hiệu quả

### 9.3. Cải thiện pipeline
1. **Nested cross-validation** thay vì single split → ước lượng tổng quát hóa tốt hơn
2. **Giảm grid search size** cho XGBoost → dùng RandomizedSearchCV
3. **TreeSHAP** thay vì KernelSHAP cho tree-based models
4. **Calibration curves** để đánh giá chất lượng xác suất dự đoán
5. **Kết hợp SMOTE + class_weight** cần được đánh giá cẩn thận — có thể chỉ dùng một trong hai

---

*Tài liệu được tạo tự động — Cần được chuyên gia xác nhận trước khi sử dụng trong báo cáo chính thức.*
