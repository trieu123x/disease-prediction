# Website Chẩn đoán Tiểu đường (Pima Diabetes AI)

> **Môn học:** Intelligent System Development — TS. Trần Đình Quế  
> **Phần 1:** Binary Classification — Pima Indians Diabetes  

---

## 📁 Cấu trúc thư mục dự án

```
website_chandoan_tieu_duong/
├── index.html                ← Single Page Web App (Frontend + ML Engine)
├── model.json                ← Mô hình Random Forest đã nén (166.8 KB)
├── vercel.json               ← Cấu hình deploy Vercel
├── notebook_part1.ipynb      ← Notebook huấn luyện & thí nghiệm (22 sections)
├── data/
│   └── pima_diabetes.csv     ← Dataset Pima Indians Diabetes (768 mẫu)
└── README.md                 ← Hướng dẫn sử dụng & deploy
```

---

## ⚡ Deploy Vercel (Serverless / Static Site)

Mô hình Random Forest được nhúng trực tiếp dưới dạng `model.json` (166.8 KB). Thuật toán inference duyệt cây (Tree Traversal) được viết bằng **Pure JavaScript** chạy 100% trong trình duyệt client, không cần backend server.

### Deploy bằng Vercel CLI
```bash
cd website_chandoan_tieu_duong
vercel --prod
```

### Deploy qua Vercel Dashboard
1. Push thư mục này lên repository GitHub.
2. Mở Vercel Dashboard → Import Project.
3. Chọn `Root Directory` là `website_chandoan_tieu_duong`.
4. Nhấn **Deploy** — Vercel sẽ tự động deploy thành công!

---

## 🚀 Chạy Local

```bash
# Sử dụng Python HTTP server đơn giản:
py -3 -m http.server 8001
```
👉 Truy cập: `http://localhost:8001`

---

## 📊 Thông tin Mô hình Machine Learning

- **Bài toán:** Binary Classification (Chẩn đoán nguy cơ tiểu đường)
- **Dataset:** Pima Indians Diabetes (768 mẫu, 8 đặc trưng y tế)
- **Mô hình:** Random Forest Classifier (50 cây quyết định, max_depth=6)
- **Độ chính xác (Accuracy):** ~74% - 76%
- **F1-Score:** ~0.60
- **Thời gian Inference:** `< 1 ms` trên trình duyệt

---

# 🧠 Assignment 03 — Mạng nơ-ron sâu 5 tầng (NumPy from scratch)

Trang web nay chạy **song song hai mô hình** và cho phép đối sánh trực tiếp:

| Engine | Mô hình | Nguồn |
|---|---|---|
| 🧠 **Deep MLP-5** (mặc định) | `8 → 128 → 64 → 32 → 16 → 1` (Sigmoid) | Assignment 03 |
| 🌲 Random Forest | 50 cây, `max_depth=6` | Assignment 01 |
| ⚖️ Đối sánh | Chạy cả hai, hiển thị chênh lệch | — |

## Kiến trúc mạng

```
Input(8) ──W1──► 128 ──ReLU──► ──W2──► 64 ──ReLU──►
         ──W3──► 32  ──ReLU──► ──W4──► 16 ──ReLU──► ──W5──► 1 ──Sigmoid──► P(tiểu đường)
```

| Tầng | W shape | Kích hoạt | Tham số |
|---|---|---|---|
| Layer 1 | (8, 128) | ReLU | 1.152 |
| Layer 2 | (128, 64) | ReLU | 8.256 |
| Layer 3 | (64, 32) | ReLU | 2.080 |
| Layer 4 | (32, 16) | ReLU | 528 |
| Layer 5 | (16, 1) | **Sigmoid** | 17 |
| **Tổng** | | | **12.033** |

Toàn bộ forward / backward / Adam được **viết tay bằng NumPy** trong
[`notebook_deep_part1.ipynb`](notebook_deep_part1.ipynb); một bản PyTorch cùng kiến trúc
được huấn luyện song song để đối chiếu.

## Kết quả trên tập Test (116 mẫu)

| Mô hình | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | **0.7672** | **0.6757** | 0.6250 | **0.6494** | 0.8293 |
| **Deep MLP-5 (NumPy)** | 0.7500 | 0.6341 | **0.6500** | 0.6420 | **0.8382** |
| Deep MLP-5 (PyTorch) | 0.7328 | 0.6098 | 0.6250 | 0.6173 | 0.8418 |
| Logistic Regression | 0.7069 | 0.5789 | 0.5500 | 0.5641 | 0.8345 |
| Decision Tree | 0.7155 | 0.6061 | 0.5000 | 0.5479 | 0.7709 |

**Hiệu chỉnh ngưỡng** (dò trên tập validation theo F1, không chạm tập test): ngưỡng 0.43
thay cho 0.5 nâng F1 lên **0.6889**, recall lên **0.775**, và giảm số ca bệnh bị bỏ sót
từ 14 xuống **9** — trong khi accuracy vẫn tăng lên 0.7586.

**Kết luận trung thực:** trên bộ dữ liệu bảng chỉ 768 mẫu, mạng sâu **không** vượt trội
áp đảo so với Random Forest. Nó thắng về chất lượng xếp hạng xác suất (ROC-AUC) — điều
trở nên hữu ích khi được phép hiệu chỉnh ngưỡng.

## Triển khai serverless

`model_deep.json` (118.7 KB) chứa toàn bộ trọng số, median điền thiếu, tham số
StandardScaler và các chỉ số đánh giá. Notebook **tự động nhúng** bundle này vào dòng
`const DEEP_MODEL = ...;` trong `index.html`, nên trang vẫn là một file tĩnh duy nhất —
không `fetch`, không backend.

Suy luận trong `index.html` chỉ gồm nhân ma trận + ReLU + Sigmoid:

```javascript
function mlpForward(model, x) {
  let a = x;
  const L = model.layers.length;
  for (let i = 0; i < L; i++) {
    a = denseLayer(a, model.layers[i], i < L - 1 ? 'relu' : 'linear');
  }
  return sigmoid(a[0]);
}
```

## Kiểm chứng

Notebook nạp lại `model_deep.json` và chạy thuật toán suy luận "kiểu JavaScript" trên
toàn bộ tập test: sai lệch tối đa **5.9 × 10⁻⁷**. Biểu đồ phân tích nằm trong
[`figures/`](figures/).
