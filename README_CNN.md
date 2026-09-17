# Assignment 04 — CNN 1D cho bài toán Chẩn đoán Tiểu đường

> **Môn học:** Intelligent System Development — TS. Trần Đình Quế
> **Sinh viên:** Đinh Hải Triều — B23DCCN843 — Lớp 06

Bổ sung cho `README.md` (Assignment 01 và 03). Tài liệu này mô tả phần **mạng
nơ-ron tích chập 1D** được thêm vào ở Assignment 04.

## Tệp mới

| File | Nội dung |
|---|---|
| `notebook_cnn_part1.ipynb` | Notebook huấn luyện CNN 1D bằng **NumPy from scratch**, PyTorch và TensorFlow |
| `model_cnn.json` | Trọng số CNN + tham số tiền xử lý (30 KB) — **frontend `fetch` file này** |
| `model_cnn_samples.json` | 12 mẫu tham chiếu để kiểm tra parity JavaScript ↔ notebook |
| `figures/c1_fig*.png` | 6 biểu đồ do notebook sinh ra |

## Kiến trúc

```
(1, 8) → Conv1D(16, K=3) → ReLU → Conv1D(16, K=3) → ReLU → MaxPool(2)
       → Conv1D(32, K=2) → ReLU → Flatten → Dropout(0.3)
       → Dense(16) → ReLU → Dense(1) → Sigmoid
```

**2.449 tham số · 5 tầng huấn luyện được.** Độ dài chuỗi đi qua các tầng:
$8 \to 6 \to 4 \to 2 \to 1$. Tám đặc trưng lâm sàng được xem như một chuỗi 1D
độ dài 8 với 1 kênh.

Ba bản NumPy / PyTorch / TensorFlow có **đúng cùng số tham số** (được `assert`
trong notebook) và đạt ROC-AUC lần lượt 0,8355 / 0,8336 / 0,8368.

## Triển khai serverless trên Vercel

Khác với Assignment 03 (nhúng inline vào `index.html`), lần này trọng số nằm
trong một file JSON riêng được **commit thẳng lên GitHub**; Vercel phục vụ nó như
một **tệp tĩnh** và frontend `fetch` về khi tải trang:

```javascript
const res = await fetch('model_cnn.json', { cache: 'no-cache' });
CNN_MODEL = await res.json();
```

Không có hàm serverless nào chạy, không có máy chủ suy luận. Toàn bộ phép tính
diễn ra trên trình duyệt và dữ liệu người dùng nhập **không rời khỏi máy họ**.

Suy luận trong `index.html` chỉ gồm nhân–cộng, ReLU, max và Sigmoid:

```javascript
function cnnForward(model, xScaled) {
  let a = { c: 1, l: xScaled.length, v: Float64Array.from(xScaled) };
  for (const layer of model.layers) {
    switch (layer.type) {
      case 'conv1d': a = cnnConv1d(a, layer); break;
      case 'relu': a = cnnRelu(a); break;
      case 'maxpool1d': a = cnnMaxPool1d(a, layer.p); break;
      case 'flatten': a = { d: a.c * a.l, v: a.v }; break;
      case 'dense': a = cnnDense(a, layer); break;
      case 'dropout': break;              // suy luận: dropout vô hiệu
    }
  }
  return sigmoid(a.v[0]);
}
```

Nếu `fetch` thất bại, trang **tự động lùi về** Deep MLP (Assignment 03) và Random
Forest (Assignment 01) vốn vẫn được nhúng inline, nên không bao giờ hỏng hoàn toàn.

## Kết quả trên tập kiểm thử (116 mẫu)

| Mô hình | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| MLP-5 (Assignment 03) | 0,7328 | 0,6076 | **0,8391** |
| CNN-5 (TensorFlow) | 0,7241 | 0,6098 | 0,8368 |
| **CNN-5 (NumPy from scratch)** | 0,7500 | 0,5915 | 0,8355 |
| Logistic Regression | 0,7069 | 0,5641 | 0,8345 |
| CNN-5 (PyTorch) | 0,7328 | 0,6076 | 0,8336 |
| Random Forest | **0,7672** | **0,6400** | 0,8250 |

**CNN không vượt được các mô hình khác** — và đó là kết quả *đúng*, không phải
lỗi. Bảy mô hình nằm gọn trong dải ROC-AUC 0,825–0,839.

## Phát hiện: thứ tự cột ảnh hưởng nhiều hơn cả mô hình

Notebook chạy một thí nghiệm **bốn nghiệm thức** để kiểm chứng Cảnh báo Mô hình
hoá (dữ liệu bảng không có tính cục bộ không gian như điểm ảnh):

| Nghiệm thức | ROC-AUC |
|---|---|
| Thứ tự gốc | 0,8355 |
| Đảo ngược *(phép đối xứng của kiến trúc)* | 0,8355 (lệch $10^{-16}$) |
| Hoán vị ngẫu nhiên (n = 10) | 0,7936 ± 0,0172 |
| Dịch riêng cột `Glucose` qua 8 vị trí | 0,7967 – 0,8523 |

Chỉ **dịch một cột duy nhất** đã làm ROC-AUC dao động **0,056**, trong khi toàn
bộ chênh lệch giữa CNN và mọi baseline chỉ cỡ 0,001–0,014. Mức dao động này
**không** bám theo độ phủ vị trí ($r = -0{,}23$), và Logistic Regression — bất
biến chính xác với hoán vị — cho độ lệch chuẩn $10^{-16}$.

**Kết luận:** thứ tự cột chỉ ảnh hưởng như một **tạo tác kiến trúc**; không có
bằng chứng nào cho thấy tích chập khai thác được quan hệ ngữ nghĩa giữa các cột
liền kề. Chi tiết ở Chương III báo cáo `A04_06_trieu.843.PDF`.

## Kiểm chứng

```bash
# Parity: mã JavaScript THẬT trong index.html so với notebook
node "../../tuan 4/lib/check_html_cnn.mjs" . diabetes
```

Kết quả: 12/12 mẫu khớp, sai lệch xác suất tối đa **2,18 × 10⁻⁶** (đúng bằng ngân
sách làm tròn 6 chữ số khi ghi JSON).

## Chạy local

```bash
py -3 -m http.server 8001
```

Mở `http://localhost:8001`. Cần chạy qua HTTP server (không mở trực tiếp file)
vì trang dùng `fetch` để nạp `model_cnn.json`.
