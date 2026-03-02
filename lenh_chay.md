# Di chuyển vào thư mục project
cd C:\Users\PC\Downloads\autoscaling-analysis

# Tạo môi trường ảo Python tên là .venv (giúp tách biệt dependency với máy)
python -m venv .venv

# Kích hoạt môi trường ảo 
.\.venv\Scripts\Activate.ps1

# Nâng cấp pip lên phiên bản mới nhất (tránh lỗi cài package)
python -m pip install --upgrade pip

# Cài đặt toàn bộ thư viện cần thiết từ file requirements.txt
pip install -r requirements.txt

# Thiết lập biến môi trường để Python nhận diện thư mục src là root module
$env:PYTHONPATH="src"
$env:AUTOSCALING_CONFIG="configs/config.yaml"


# Chạy toàn bộ pipeline thông qua CLI (từ preprocess → train → benchmark → simulate...)
python -m autoscaling_analysis.cli --config configs/config.yaml all


# ====== CHẠY TỪNG BƯỚC RIÊNG LẺ ======

# 1. Tiền xử lý dữ liệu (clean, chuẩn hóa, chia tập...)
python scripts\preprocess.py --config configs/config.yaml

# 2. Tạo feature cho mô hình (feature engineering)
python scripts\features.py --config configs/config.yaml

# 3. Huấn luyện mô hình
python scripts\train.py --config configs/config.yaml

# 4. Đánh giá hiệu năng mô hình trên tập test
python scripts\benchmark.py --config configs/config.yaml --split test

# 5. Mô phỏng autoscaling dựa trên metric "hits"
# --window 5m: cửa sổ thời gian 5 phút
# --model xgb: sử dụng model XGBoost
python scripts\simulate_scaling.py --config configs/config.yaml --metric hits --window 5m --model xgb


# ====== CHẠY GIAO DIỆN WEB ======

# Khởi động Streamlit UI để xem dashboard trực quan
streamlit run src\autoscaling_analysis\ui\streamlit_app.py