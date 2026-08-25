# Website Chẩn đoán Tiểu đường (Pima Diabetes AI)

> **Môn học:** Intelligent System Development — TS. Đinh Quế Trần  
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
