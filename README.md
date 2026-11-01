# Elden Ring Steam Reviews — Phân tích cảm xúc

Đồ án môn Khai phá dữ liệu, phân loại đánh giá tiếng Anh về *Elden Ring* trên Steam. Nhãn `voted_up` biểu thị người dùng có đề xuất trò chơi (`1`) hay không (`0`); đây là nhãn khuyến nghị, không phải nhãn cảm xúc được gán thủ công.

**Nguồn dữ liệu:** [Elden Ring Steam Reviews — Kaggle](https://www.kaggle.com/datasets/noahx1/elden-ring-steam-reviews). Dữ liệu gốc nằm tại `data/raw/elden_ring_steam_reviews.csv` và không bị chỉnh sửa.

## Cấu trúc dự án

```text
.
├── data/
│   ├── raw/                         # CSV gốc
│   └── processed/                   # train.csv, validation.csv, test.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_sentence_transformer.ipynb
│   ├── 05_snowflake_embed.ipynb
│   └── 06_final_evaluation.ipynb
├── report/
│   └── report_draft.md
├── results/
│   └── final_comparison.csv
└── .venv/                            # Môi trường Python cục bộ
```

## Quy trình

1. `01_eda.ipynb`: khảo sát dữ liệu, nhãn, review trống, giá trị bất thường và độ dài văn bản.
2. `02_preprocessing.ipynb`: bỏ review trống, làm sạch HTML/khoảng trắng, ánh xạ nhãn và tạo các tập dữ liệu ban đầu.
3. `03_logistic_regression.ipynb`: phát hiện review trùng/nhãn mâu thuẫn, tạo lại các phân hoạch không giao nhau theo nội dung, rồi thử TF-IDF + Logistic Regression trên Validation. **Notebook này cập nhật các CSV trong `data/processed/`**; chạy trước các notebook mô hình còn lại.
4. `04_sentence_transformer.ipynb`: thử `sentence-transformers/all-MiniLM-L6-v2` + Logistic Regression trên Validation.
5. `05_snowflake_embed.ipynb`: thử `Snowflake/snowflake-arctic-embed-s` + Logistic Regression trên Validation.
6. `06_final_evaluation.ipynb`: đánh giá một lần các cấu hình đã chọn trên Test và lưu bảng kết quả tại `results/final_comparison.csv`.

Quá trình loại nhóm review có nhãn mâu thuẫn và gộp nội dung lặp làm số mẫu giảm từ 9.763 xuống 7.946. Tập cuối gồm 5.562 mẫu Train, 1.192 Validation và 1.192 Test. Các notebook không sử dụng metadata tác giả/thời gian chơi làm đặc trưng.

## Phương pháp

- **TF-IDF + Logistic Regression:** vector hóa theo từ và học bộ phân loại tuyến tính.
- **MiniLM + Logistic Regression:** embedding câu 384 chiều từ `sentence-transformers/all-MiniLM-L6-v2`, kết hợp Logistic Regression.
- **Snowflake Arctic Embed S + Logistic Regression:** embedding 384 chiều từ `Snowflake/snowflake-arctic-embed-s` (prompt `query`), kết hợp Logistic Regression.

Các mô hình cuối dùng `class_weight="balanced"`. Chỉ TF-IDF và Logistic Regression được fit trên Train; Transformer pretrained chỉ tạo embedding, không fine-tune. Precision, Recall và F1-score dưới đây tính cho lớp 1; F1 lớp 0 được báo cáo riêng.

## Kết quả cuối trên Test

| Phương pháp | Accuracy | Precision | Recall | F1-score | Macro-F1 | F1 lớp 0 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0,8943 | 0,9748 | 0,9097 | 0,9411 | **0,7124** | **0,4836** |
| MiniLM + Logistic Regression | 0,8456 | 0,9843 | 0,8473 | 0,9107 | 0,6714 | 0,4321 |
| Snowflake Arctic Embed S + Logistic Regression | 0,8641 | **0,9846** | 0,8672 | 0,9222 | 0,6929 | 0,4636 |

> TF-IDF đạt Macro-F1 cao nhất trên Test. Các chỉ số là kết quả của một lần đánh giá với cấu hình đã chọn trên Validation; Test không được dùng để điều chỉnh mô hình.

## Cài đặt và chạy

Yêu cầu **Python 3.11**. Trên Windows PowerShell, từ thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas==3.1.0 numpy==2.4.6 matplotlib==3.11.2 scikit-learn==1.9.1 jupyter==1.1.1 sentence-transformers==6.2.0 torch==2.15.0
jupyter lab
```

Mở và chạy tuần tự các notebook `01_eda.ipynb` đến `06_final_evaluation.ipynb` trong `notebooks/`. Lần đầu chạy notebook 04–05 cần Internet để tải mô hình Hugging Face. Các Transformer được chạy trên CPU.

Thư viện Python trực tiếp dùng trong notebook: **pandas, NumPy, Matplotlib, scikit-learn, Sentence Transformers, PyTorch và Jupyter/IPython**. Các phiên bản trong môi trường `.venv` hiện tại được ghi trong lệnh cài đặt phía trên.
