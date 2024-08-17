import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from openpyxl import load_workbook
from pre_processing import (
    MaxResize, outputs_to_objects, visualize_detected_tables,
    objects_to_crops, apply_ocr, get_cell_coordinates_by_row,
    plot_results, TableTransformerForObjectDetection, AutoModelForObjectDetection,
    Predictor, Cfg, fig2img
)


# Tải mô hình phát hiện bảng
model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Xử lý ảnh đầu vào
file_path = 'images/Bang1demo.png'
image = Image.open(file_path).convert("RGB")
width, height = image.size
resized_image = image.resize((int(0.6 * width), int(0.6 * height)))
resized_image.save('Bang1demo_resized.png')

# Áp dụng các bước tiền xử lý cho ảnh
detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pixel_values = detection_transform(image).unsqueeze(0)
pixel_values = pixel_values.to(device)

# Dự đoán bounding box và nhãn cho các bảng trong ảnh
with torch.no_grad():
    outputs = model(pixel_values)

# Hậu xử lý kết quả dự đoán
id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"
objects = outputs_to_objects(outputs, image.size, id2label)

# Hiển thị các bảng đã được phát hiện
fig = visualize_detected_tables(image, objects)
plt.show()
visualized_image = fig2img(fig)

# Cắt các bảng từ ảnh
tokens = []
detection_class_thresholds = {
    "table": 0.5,
    "table rotated": 0.5,
    "no object": 10
}
crop_padding = 8
tables_crops = objects_to_crops(image, tokens, objects, detection_class_thresholds, padding=crop_padding)
cropped_table = tables_crops[0]['image'].convert("RGB")
plt.imshow(cropped_table)
plt.axis('off')
plt.show()
cropped_table.save("table_cropped.png")

# Tải mô hình nhận diện cấu trúc bảng
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
structure_model.to(device)

# Áp dụng các bước tiền xử lý cho ảnh cắt được từ bảng
structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pixel_values = structure_transform(cropped_table).unsqueeze(0)
pixel_values = pixel_values.to(device)

# Dự đoán các thành phần trong bảng (hàng, cột)
with torch.no_grad():
    outputs = structure_model(pixel_values)
structure_id2label = structure_model.config.id2label
structure_id2label[len(structure_id2label)] = "no object"
cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)

# Hiển thị các ô được nhận diện trong bảng
cropped_table_visualized = cropped_table.copy()
draw = ImageDraw.Draw(cropped_table_visualized)
for cell in cells:
    draw.rectangle(cell["bbox"], outline="red")
plt.imshow(cropped_table_visualized)
plt.axis('off')
plt.show()

# Hiển thị các hàng trong bảng
plot_results(cells, structure_model,cropped_table,class_to_visualize="table row" )

# Cấu hình vietOCR
cfg = Cfg.load_config_from_name('vgg_transformer')
cfg['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
cfg['cnn']['pretrained'] = False
cfg['device'] = 'cpu'
predictor = Predictor(cfg)

# Áp dụng OCR để nhận diện văn bản trong các ô của bảng
cell_coordinates = get_cell_coordinates_by_row(cells)
data = apply_ocr(cell_coordinates, cropped_table, predictor)
for row, row_data in data.items():
    print(row_data)

# Lưu kết quả OCR thành file Excel
df = pd.DataFrame.from_dict(data, orient='index')
df.to_excel('output/output_vietocr.xlsx', index=False)
workbook = load_workbook('output/output_vietocr.xlsx')
worksheet = workbook.active
for column in worksheet.columns:
    max_length = 0
    column = list(column)
    for cell in column:
        try:
            max_length = max(max_length, len(str(cell.value)))
        except:
            pass
    adjusted_width = max_length + 2
    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
workbook.save('output/output_vietocr.xlsx')
