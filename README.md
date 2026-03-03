# Autoscaling Analysis — Autoscaling

## 1. Tóm tắt
- **Vấn đề cần giải quyết:**
  Trong quản trị hệ thống đám mây, việc cấp phát tài nguyên tĩnh không thể thích ứng với lưu lượng thay đổi theo thời gian. Điều này dẫn đến các vấn đề như lãng phí tài nguyên khi tải thấp và quá tải hệ thống khi tải tăng đột biến. Hệ thống cần được tự động mở rộng và thu hẹp để tối ưu hóa chi phí mà vẫn duy trì hiệu suất cao.

- **Ý tưởng và cách tiếp cận:**
  Dự án này sử dụng mô hình **dự báo tải ngắn hạn** (short-term load forecasting) để dự báo nhu cầu tài nguyên của hệ thống, từ đó điều chỉnh tài nguyên (scale-out hoặc scale-in) một cách linh hoạt và chính xác. Các mô hình như **XGBoost** được sử dụng để dự báo số lượng truy cập và dung lượng dữ liệu trong tương lai. Các chính sách **autoscaling** sẽ được mô phỏng để kiểm tra hiệu quả của hệ thống.

- **Giá trị thực tiễn:**
  - **Giảm chi phí vận hành:** Tối ưu hóa việc cấp phát tài nguyên, giảm thiểu lãng phí khi tải thấp.
  - **Cải thiện hiệu suất:** Giảm thiểu tình trạng quá tải, bảo vệ hệ thống khỏi tình trạng giảm hiệu suất hoặc sập hệ thống khi có lưu lượng đột ngột.
  - **Tăng tính linh hoạt:** Hệ thống có thể tự động điều chỉnh theo nhu cầu mà không cần sự can thiệp thủ công.

## 2. Dữ liệu
- **Nguồn:** [NASA HTTP Access Log Dataset](https://www.kaggle.com/datasets/aaraki/nasa-http-logs)
- **Mô tả trường dữ liệu chính:**
  - **datetime:** Thời gian yêu cầu HTTP.
  - **host:** Địa chỉ máy chủ nhận yêu cầu.
  - **method:** Phương thức HTTP (GET, POST,...).
  - **url:** Đường dẫn truy cập.
  - **status:** Mã trạng thái HTTP.
  - **bytes:** Dung lượng dữ liệu trả về.
- **Tiền xử lý đã thực hiện:**
  - **Missing values:** Loại bỏ các dòng dữ liệu không có thông tin hoặc có lỗi.
  - **Outliers:** Phát hiện và loại bỏ các giá trị bất thường có thể gây ảnh hưởng đến mô hình.
  - **Normalization:** Chuẩn hóa thời gian và chuyển đổi dữ liệu dạng chuỗi thành dạng số để mô hình có thể sử dụng.
  - **Feature Engineering:** Tạo các chỉ số như tổng số lượt truy cập, tổng dung lượng trả về, tỷ lệ lỗi (error rate), và phát hiện các spikes.

## 3. Mô hình & Kiến trúc

- **Kiến trúc tổng thể:**
  Hệ thống được chia thành các bước chính sau:
  
  1. **Ingest logs:** Đọc và parse các file log HTTP.
  2. **Time series aggregation:** Chuyển đổi dữ liệu thành các chuỗi thời gian với các độ phân giải khác nhau (1m, 5m, 15m).
  3. **Feature engineering:** Tạo các đặc trưng như hits, bytes_sum, error_rate, và phát hiện các spikes.
  4. **Model training:** Huấn luyện các mô hình dự báo sử dụng XGBoost và Seasonal Naive.
  5. **Scaling simulation:** Mô phỏng các chính sách autoscaling và tính toán các chỉ số hiệu suất (chi phí, SLA vi phạm, v.v.).
  6. **API Deployment:** Cung cấp API cho các dự báo và điều chỉnh tài nguyên.
  
  **Tổng quan luồng công việc:**

  ```text
  RAW HTTP LOGS
       ↓
  INGEST & PARSE
       ↓
  TIME SERIES AGGREGATION 
       ↓
  FEATURE ENGINEERING
       ↓
  FORECAST MODEL 
       ↓
  LOAD FORECAST 
       ↓
  SCALING POLICY ENGINE
       ↓
  SCALE OUT / SCALE IN DECISION
       ↓
  DEPLOYMENT / CLOUD INFRA CONTROL

## Mô Hình Sử Dụng

### 1. Seasonal Naive (Baseline Model)
Mô hình cơ bản dùng để so sánh.

### 2. XGBoost
Mô hình hồi quy cây quyết định, sử dụng để dự báo tải của hệ thống dựa trên các đặc trưng đã tạo.

## Chiến Lược Validation/Training

1. Cross-validation: Sử dụng phương pháp k-fold cross-validation time series để đánh giá mô hình và chọn tham số tối ưu.

2. Train/Test Split: Chia dữ liệu theo thời gian, đảm bảo rằng mô hình chỉ sử dụng dữ liệu quá khứ để huấn luyện và không học từ tương lai.

## Tránh Data Leakage Bằng Cách:

1. Strict Time-based Split: Đảm bảo rằng dữ liệu huấn luyện không chứa thông tin của tương lai bằng cách chia dữ liệu theo mốc thời gian.

2. Không Sử Dụng Thông Tin Tương Lai Trong Feature Engineering: Các đặc trưng chỉ sử dụng dữ liệu quá khứ và hiện tại, không có thông tin của tương lai.

3. Rolling-window Validation: Thực hiện kiểm tra mô hình theo cửa sổ trượt để mô phỏng tình huống thực tế khi hệ thống phải dự báo theo từng thời điểm.

### Cấu trúc thư mục
 ```text
autoscaling-analysis/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ pyproject.toml
├─ .gitignore
├─ Makefile
│
├─ configs/
│  └─ config.yaml
│
├─ data/
│  ├─ raw/
│  │  ├─ train.txt
│  │  └─ test.txt
│  ├─ interim/
│  └─ processed/
│
├─ notebooks/
│  ├─ auto_scaling.ipynb
│
├─ artifacts/
│  ├─ models/
│  ├─ predictions/
│  ├─ metrics/
│  └─ scaling/
│
├─ scripts/
│  ├─ preprocess.py
│  ├─ features.py
│  ├─ train.py
│  ├─ benchmark.py
│  ├─ simulate_scaling.py
│  └─ run_ui.sh
│
├─ src/
│  └─ autoscaling_analysis/
│     ├─ __init__.py
│     ├─ cli.py
│     ├─ config.py
│     ├─ ingest/
│     │  ├─ __init__.py
│     │  ├─ schemas.py
│     │  └─ parse_logs.py
│     ├─ timeseries/
│     │  ├─ __init__.py
│     │  ├─ gaps.py
│     │  └─ build_ts3.py
│     ├─ eda/
│     │  ├─ __init__.py
│     │  ├─ plots.py
│     │  └─ eda_report.py
│     ├─ features/
│     │  ├─ __init__.py
│     │  ├─ segments.py
│     │  ├─ transforms.py
│     │  └─ make_features.py
│     ├─ models/
│     │  ├─ __init__.py
│     │  ├─ metrics.py
│     │  ├─ xgb_model.py
│     │  └─ seasonal_naive.py
│     ├─ benchmark/
│     │  ├─ __init__.py
│     │  └─ build_benchmark.py
│     ├─ scaling/
│     │  ├─ __init__.py
│     │  ├─ policy.py
│     │  ├─ anomaly.py
│     │  ├─ latency.py
│     │  └─ simulate.py
│     └─ ui/
│        ├─ __init__.py
│        └─ streamlit_app.py
│
└─ tests/
   ├─ __init__.py
   ├─ test_capacity_key_normalize.py
   ├─ test_feature_leakage.py
   ├─ test_required_instances.py
   └─ test_parse_regex.py
```
## 4. Đánh giá

### Metrics:
- **MAE (Mean Absolute Error):** Đo sai số tuyệt đối trung bình giữa giá trị dự đoán và giá trị thực.
- **MAPE (Mean Absolute Percentage Error):** Đo sai số phần trăm tuyệt đối giữa giá trị dự đoán và giá trị thực.
- **RMSE (Root Mean Square Error):** Đo sai số bình phương trung bình căn bậc 2, tập trung vào các sai số lớn.
- **MSE (Mean Squared Error):** Đo lường sai số bình phương trung bình, sử dụng trong quá trình huấn luyện.

### Kết quả:
#### Target: `bytes_sum`

| Window | MAE (SN) | MAE (XGB) | MAPE% (SN) | MAPE% (XGB) | RMSE (SN) | RMSE (XGB) |
|--------|----------|-----------|------------|-------------|-----------|------------|
| **1m** | 4.82e+05 | 4.63e+05  | 182.34     | 88.17       | 6.83e+05  | 7.00e+05   |
| **5m** | 1.55e+06 | 1.32e+06  | 69.10      | 43.37       | 2.07e+06  | 1.86e+06   |
| **15m**| 3.68e+06 | 2.83e+06  | 48.72      | 31.81       | 4.94e+06  | 3.89e+06   |

#### Target: `hits`

| Window | MAE (SN) | MAE (XGB) | MAPE% (SN) | MAPE% (XGB) | RMSE (SN) | RMSE (XGB) |
|--------|----------|-----------|------------|-------------|-----------|------------|
| **1m** | 18.21    | 15.58     | 74.15      | 64.46       | 24.34     | 21.59      |
| **5m** | 67.51    | 52.61     | 50.15      | 38.36       | 92.50     | 70.98      |
| **15m**| 174.63   | 112.97    | 35.54      | 20.77       | 245.75    | 158.22     |

---

## 5. Triển khai & Demo

### Hướng dẫn chạy:
Để chạy dự án, cần cài đặt các phụ thuộc và môi trường như sau:

1. **Tạo môi trường ảo** và cài đặt các thư viện yêu cầu:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2. **Huấn luyện mô hình**:
    Để huấn luyện các mô hình dự báo (XGBoost, Seasonal Naive, v.v.) từ dữ liệu đã tiền xử lý, sử dụng câu lệnh sau:
    ```bash
    python scripts/train.py --config configs/train.yaml

3. Chạy mô phỏng autoscaling:
    Để mô phỏng các chính sách autoscaling và tính toán các chỉ số hiệu suất, sử dụng câu lệnh sau:
    ```bash
    python scripts/simulate_scaling.py
4. Demo UI:
    Xem demo ứng dụng UI với Streamlit bằng cách chạy:
    ```bash
    sh scripts/run_ui.sh
    Sau khi chạy lệnh trên, truy cập vào ứng dụng UI qua đường dẫn http://localhost:8501 để xem các dự báo và mô phỏng autoscaling trực quan.

## 6. Giới hạn & Hướng phát triển

- **Giới hạn hiện tại:**
  - Chỉ sử dụng dữ liệu HTTP log, không có thông tin về tài nguyên hệ thống như CPU hoặc RAM.
  - Dự báo chỉ dựa trên một số chỉ số (hits, bytes), không tích hợp nhiều yếu tố khác.
  
- **Kế hoạch cải tiến:**
  - **Drift detection**: Phát hiện sự thay đổi trong phân phối dữ liệu để điều chỉnh mô hình.
  - **Uncertainty quantification**: Đánh giá sự không chắc chắn của dự báo.
  - **Scaling policy tuning**: Tinh chỉnh các chính sách autoscaling để giảm thiểu chi phí và cải thiện hiệu suất.

## 7. Tác động & Ứng dụng

- **Lợi ích định tính/định lượng:**
  - Giảm chi phí vận hành ~36-37% so với cấp phát tài nguyên cố định (static provisioning).
  - Tăng cường hiệu suất hệ thống với 0% vi phạm SLA.

- **Kịch bản triển khai trong doanh nghiệp:**
  - **Web applications**: Tối ưu hóa tài nguyên cho các trang web thương mại điện tử trong các mùa cao điểm.
  - **FinTech systems**: Dự báo tải cho các dịch vụ tài chính trong các dịp cao điểm.
  - **SaaS platforms**: Dự báo tải cho các nền tảng đa người dùng.

## 8. Tác giả & Giấy phép

- **Tác giả**: Vu Manh Hiep
- **License**: MIT

