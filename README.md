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
│  ├─ train.py 
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
