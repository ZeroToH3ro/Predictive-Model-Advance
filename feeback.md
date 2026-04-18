NHẬN XÉT MANUSCRIPT
Predicting Direct-Acting Antiviral Response in Hepatitis C:
A Comparison of Whole-Genome, Target Gene, and RAS Profiles
in a Multi-Genotype Cohort
Ngày nhận xét: 12/04/2026

1. ĐIỂM MẠNH
1.1. Tính mới (Novelty)
Đây là nghiên cứu đầu tiên so sánh đồng thời ba cấp độ thông tin genomic (whole-genome, target gene, RAS) cho dự đoán đáp ứng DAA trong cùng một cohort. Việc enrichment genotype 6 — genotype đa dạng nhất nhưng thiếu đại diện nhất trong các clinical trials — lấp đầy một khoảng trống thực sự trong y văn.
Phát hiện GT6r và GT6e có hướng tác động ngược nhau (GT6r → non-SVR, GT6e → SVR) là genuinely novel và có ý nghĩa lâm sàng cao cho quần thể Đông Nam Á.
1.2. Phương pháp có chiều sâu
13 classifiers × 5 feature sets = 65 configurations, kết hợp bootstrap 1,000 lần và SHAP analysis — đây là mức độ comprehensive hiếm thấy trong lĩnh vực hepatology. Pipeline preprocessing (được mô tả rõ ràng: VarianceThreshold → StandardScaler → SMOTE → FiniteClipper) và việc công khai code trên GitHub thể hiện tính minh bạch và tái lập.
1.3. SHAP Interpretability xuất sắc
Phần SHAP analysis là điểm sáng của bài báo. Việc phân tích đồng thời global importance, direction of effect, category analysis, interaction analysis, và individual force plots giúp bài vượt qua rào cản “black box” thường gặp ở ML papers trong y khoa. Đặc biệt, việc xác định hai pathways to treatment failure (drug resistance-driven và genotype-driven) rất có giá trị lâm sàng.
1.4. Clinical framing tốt
Bài viết liên tục nhấn mạnh ML là “decision support tool” chứ không phải thay thế bác sĩ. Đề xuất hệ thống 3 tầng (genotype identification → targeted sequencing + ML → SHAP explanation) là một lộ trình translational rõ ràng và khả thi.
1.5. Viết tốt và cấu trúc logic
Tiếng Anh trôi chảy, cấu trúc bài rõ ràng, phần Discussion liên kết chặt chẽ với literature. Ba câu hỏi nghiên cứu được nêu rõ trong Introduction và được trả lời tuần tự trong Results/Discussion. Tham khảo 35 tài liệu cập nhật (đến 2025).

2. ĐIỂM YẾU VÀ CÁCH KHẮC PHỤC
2.1. Threshold optimization trên test set — Data leakage (ĐỘ NGHIÊM TRỌNG: CAO)
Vấn đề: Phần 2.6 mô tả việc tối ưu hóa threshold bằng Youden’s J trên ROC của test set. Đây là data leakage vì test set được dùng để vừa chọn threshold vừa đánh giá performance, dẫn đến ước tính quá lạc quan về accuracy, precision, recall, và F1. AUC không bị ảnh hưởng vì nó độc lập với threshold, nhưng các metric khác đều bị inflated.
Cách khắc phục: Thực hiện threshold optimization trong vòng cross-validation trên training set (5-fold stratified CV). Cụ thể: trong mỗi fold, chọn threshold tối ưu bằng Youden’s J trên validation fold, lấy trung bình các threshold tối ưu, rồi áp dụng threshold trung bình đó lên test set. Hoặc đơn giản hơn: tách một validation set riêng (ví dụ 60% train / 20% validation / 20% test) để chọn threshold trên validation và đánh giá trên test.
2.2. Single train/test split và SMOTE trên cỡ mẫu nhỏ (ĐỘ NGHIÊM TRỌNG: CAO)
Vấn đề: Chỉ dùng 1 train/test split cố định (random_state=42) trên 162 mẫu là rủi ro cao. Kết quả có thể thay đổi đáng kể với random_state khác. Bootstrap trên cùng 65 test samples chỉ đo sampling variability, không đo variability do split khác nhau. Ngoài ra, SMOTE trên chỉ ~37 minority samples trong training set tạo synthetic samples rất gần nhau, dễ gây overfitting, đặc biệt với 9,237 nucleotide features.
Cách khắc phục: (c) Nếu giữ single split, chạy lại với nhiều random_state khác nhau và báo cáo phân bố kết quả.
2.3. Confounding từ SOF/LDV (ĐỘ NGHIÊM TRỌNG: CAO)
Vấn đề: 21/21 bệnh nhân dùng SOF/LDV đều non-SVR, tất cả đều là genotype 1. Điều này tạo ra perfect confounder: model có thể đơn giản học rằng các genotype 1 subtypes đặc biệt (1l, 1e, 1b, 1g) = non-SVR, thực chất là detect regimen failure chứ không phải genomic resistance. 21/61 non-SVR samples (34.4%) đến từ cùng một nguồn confounding này.
Cách khắc phục: (a) Chạy lại toàn bộ phân tích sau khi loại bỏ 21 SOF/LDV samples và so sánh AUC. Nếu AUC giảm mạnh, cần thảo luận minh bạch về điều này.
