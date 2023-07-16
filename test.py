import matplotlib.pyplot as plt

# Tạo các bước trong mô hình tổng quan
steps = [
    "Thu thập dữ liệu",
    "Tiền xử lý dữ liệu",
    "Xây dựng mô hình",
    "Huấn luyện mô hình",
    "Đánh giá mô hình",
    "Tinh chỉnh và tối ưu mô hình",
    "Triển khai và sử dụng"
]

# Tạo biểu đồ sơ đồ khái quát
plt.figure(figsize=(8, 5))
plt.bar(steps, range(len(steps)), color='lightblue')
plt.xlabel('Bước trong quá trình xây dựng mô hình')
plt.ylabel('Thứ tự bước')
plt.title('Mô hình tổng quan về nhận dạng điệu nhảy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
