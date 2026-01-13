# Resnet-50 Re-implementation (PyTorch)
Dự án này **cài đặt lại Resnet-50** "theo hướng tự viết pipeline" và cung cấp:
- Load dữ liệu kiểu *ImageFolder/list-file* (train_list.txt / val_list.txt)
- Train + validate, tính **Top-1 / Top-5**
- Log ra `train.log` và `metrics.csv`
- Lưu checkpoint `last.pth` và `best.pth`

---

## 1. Cấu trúc thư mục & chức năng