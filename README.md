# Resnet-50 Re-implementation (PyTorch)
Dự án này **cài đặt lại Resnet-50** "theo hướng tự viết pipeline" và cung cấp:
- Load dữ liệu kiểu *ImageFolder/list-file* (train_list.txt / val_list.txt)
- Train + validate, tính **Top-1 / Top-5**
- Log ra `train.log` và `metrics.csv`
- Lưu checkpoint `last.pth` và `best.pth`

# 1. Cấu trúc thư mục & chức năng
```
Resnet-50/
├─ data/ # THÔNG TIN DỮ LIỆU
│  ├─ splits/ # train_list.txt, val_list.txt
│  ├─ meta/ # class_map.json, classes.txt, dataset_stats.json
│  └─ tools/ # script tạo splits/meta từ dataset_root
│     ├─ prepare_subtest_and_split.py 
│     └─ verify.py
│
├─ src/ # CODE CHÍNH
│  ├─ __init__.py
│  │
│  ├─ models/
│  │  ├─ __init__.py
│  │  └─ resnet50/
│  │     ├─ __init__.py
│  │     ├─ resnet50.py
│  │     └─ blocks/
│  │        ├─ __init__.py
│  │        ├─ bottleneck.py
│  │        ├─ conv_helpers.py
│  │        └─ stem.py
│  │
│  ├─ datasets/
│  │  ├─ __init__.py
│  │  ├─ imagenet_style/
│  │  │  ├─ __init__.py
│  │  │  └─ list_dataset.py
│  │  └─ transforms/
│  │     ├─ __init__.py
│  │     └─ imagenet_transforms.py
│  │
│  ├─ engine/
│  │  ├─ __init__.py
│  │  ├─ eval/
│  │  │  ├─ __init__.py
│  │  │  └─ evaluate.py
│  │  └─ train/
│  │     ├─ __init__.py
│  │     ├─ train_one_epoch.py
│  │     └─ loop.py
│  │
│  ├─ losses/
│  │  ├─ __init__.py
│  │  ├─ cross_entropy.py
│  │  └─ label_smoothing.py
│  │
│  ├─ optim/
│  │  ├─ __init__.py
│  │  ├─ optimizer_factory.py
│  │  └─ cheduler_factory.py
│  │
│  └─ utils/
│     ├─ __init__.py
│     ├─ checkpoint.py
│     ├─ config.py
│     ├─ logger.py
│     ├─ metrics.py
│     └─ seed.py
│
├─ scripts/ # Chạy train/eval (người dùng chỉ sửa path)
│  ├─ train_run.py 
│  └─ eval.py
│
├─ notebook/ # Vẽ biểu đồ + tổng hợp best epoch/last epoch
│  └─ plot_metrics.ipynb
│
├─ outputs/ # SINH RA SAU KHI CHẠY (bị .gitignore)
│  └─ exp_001/
│     ├─ train.log
│     ├─ metrics.csv
│     ├─ last.pth
│     └─ best.pth
│
├─ .gitignore
├─ README.md
└─ requirements.txt
```
# 2. Cài môi trường (Windows) - khuyên dùng Python 3.12
## 2.1. Tạo venv
Mở PowerShell/CMD trong folder **Resnet50**:
```bash
py -3.12 -m venv .venv
```
## 2.2. Kích hoạt venv
```bash
.\.venv\Scripts\python.exe -m pip install -U pip
```
# 3. Cài PyTorch và thư viện 
## 3.1. Cài PyTorch
```bash
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
## 3.2. Cài requirement
```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
## 3.3. Cài ipykernel nếu muốn dùng notebook vẽ biểu đồ
```bash
.\.venv\Scripts\python.exe -m pip install -U ipykernel
```
# 4. Hướng dẫn lấy Dataset
## 4.1. Dataset dùng trong demo
- Dự án dùng **CUB-200-2011 (Caltech-UCSD Birds 200-2011)** cho bài toán phân loại hình ảnh **200 lớp**
- Cấu hình demo: **10,000 ảnh**, split **90/10** (train 9,000 / val 1,000).
## 4.2. Cách tải Dataset
- Mở đường link: **https://data.caltech.edu/records/65de6-vp158**
- Tìm mục Files và download *CUB_200_2011.tgz*
- Chọn thư mục lưu và giải nén (ví dụ: `C:\Dataset\`)
- Dataset đã ổn nếu trong dataset_root có các file/thư mục quan trọng sau:
```
C:\Dataset\CUB_200_2011\ (nơi bạn lưu file)
├─ images\ # chứa toàn bộ ảnh theo class folder
├─ images.txt # mapping id → path ảnh
├─ image_class_labels.txt # mapping ảnh → label
├─ train_test_split.txt # split train/test gốc của bộ CUB
└─ ...
```
# 5. Xử lý dữ liệu: tạo train/val list + meta (splits & meta)
## 5.1 Mục đích
Bước này dùng để tạo dữ liệu đầu vào chuẩn cho dự án, cụ thể là:
- `train_list.txt` / `val_list.txt`: danh sách ảnh + nhãn để loader đọc  
- `class_map.json`, `classes.txt`: map nhãn về dạng `0..C-1`  
- `dataset_stats.json`: thống kê số ảnh, số lớp, seed…

-> “chuẩn hoá CUB” để project train/eval dễ, chạy được trên máy khác chỉ cần đổi `dataset_root`

## 5.2 Tool dùng để xử lý
- Script: `data/tools/prepare_subset_and_split.py`
- Script này sẽ:
    - đọc dataset từ `dataset_root`
    - lấy ra đúng **10,000 ảnh** (demo) và chia **90/10** thành train/val
    - sinh ra các file trong `data/splits/` và `data/meta/`

## 5.3 Cách chạy tool
Trong root chính **Resnet50**, chạy:

**Windows (PowerShell/CMD):**
```bash
.\.venv\Scripts\python.exe data\tools\prepare_subset_and_split.py ^
  --dataset_root "C:\Dataset\CUB_200_2011" ^
  --train_images_total 10000 ^
  --seed 42
```
Sau khi chạy xong sẽ thấy các file sau:
- *data/splits/train_list.txt*
- *data/splits/val_list.txt*
- *data/meta/class_map.json*
- *data/meta/classes.txt*
- *data/meta/dataset_stats.json*
Như vậy là đẫ xong cho chuẩn bị dữ liệu để train

# 6. Train: chạy huấn luyện ResNet-50

## 6.1 Trước khi train cần chú ý:
Mở `scripts/train_run.py` và sửa:
- `dataset_root` : trỏ đúng nơi bạn giải nén CUB.
- `out_dir` : thư mục lưu kết quả (Ví dụ: `outputs/exp_001`)
Sau khi chạy sẽ tự tạo folder `outputs/exp_001` (Có thể tùy chỉnh)

```python
train(
    #Sửa chính 2 đoạn này:
    dataset_root=r"C:\Dataset\CUB_200_2011",
    out_dir="outputs/exp_001",

    #Các thông số có thể sửa tùy nhu cầu train
    num_classes=200,
    epochs=20,
    batch_size=32,
    num_workers=0,
    lr=0.01,
    amp=True,
    log_interval=50,
)
```
## 6.2 Chạy train
```bash
.\.venv\Scripts\python.exe scripts\train.py
```
## 6.3 Output
Trong `out_dir` sẽ có:
- `train.log`: log quá trình train  
- `metrics.csv`: loss/top1/top5 theo epoch
- `last.pth`: checkpoint epoch cuối  
- `best.pth`: checkpoint có val top-1 tốt nhất  

# 7. Eval: đánh giá checkpoint (dùng để chốt Top-1/Top-5)
## 7.1 Mục đích
`eval.py` dùng để:
- load checkpoint (`best.pth` hoặc `last.pth`)
- chạy inference trên tập validation
- in ra **Top-1 / Top-5** cuối cùng

## 7.2 Cách chạy eval
```bash
.\.venv\Scripts\python.exe scripts\eval.py
```
Nếu không chạy thì vẫn có thể xem Val Top-1/Top-5 bằng cách mở `outputs/exp_001/metrics.csv` vì val metrics theo từng epoch đã được lưu sẵn trong đó.

# 8. Vẽ biểu đồ + tổng hợp best epoch trong Notebook
Nếu muốn vẽ biểu đồ dựa trên dữ liệu train (tự điều chỉnh) hãy mở `notebook/plot_metrics.ipynb` và chỉnh:

```python
from pathlib import Path
EXP_DIR = Path(r"D:\24022440\ML\Resnet 50\outputs\exp_001") # Sửa trỏ đến đúng folder output tạo
```