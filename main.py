# pip install vietocr
# pip install -q easyocr
# pip install -q transformers
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoModelForObjectDetection
from transformers import TableTransformerForObjectDetection
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download
from torchvision import transforms
from matplotlib.patches import Patch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from tqdm.auto import tqdm
from openpyxl import load_workbook


model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
model.config.id2label


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("")

# Nếu sử dụng file từ Hugging Face Hub, bạn có thể sử dụng đoạn mã này:
# from huggingface_hub import hf_hub_download
# file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="image.png")

file_path = 'images/Bang1demo.png'
image = Image.open(file_path).convert("RGB")

width, height = image.size
resized_image = image.resize((int(0.6 * width), int(0.6 * height)))

# save resized_image
resized_image.save('Bang1demo_resized.png')

class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

detection_transform = transforms.Compose([
    MaxResize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pixel_values = detection_transform(image).unsqueeze(0)
pixel_values = pixel_values.to(device)
print(pixel_values.shape)



with torch.no_grad():
  outputs = model(pixel_values)
print(outputs.logits.shape)

### POST-PROCESSING
def adjust_bbox(bbox, img_size, margin=0.02):
    """
    Điều chỉnh bounding box bằng cách thêm khoảng lề.
    bbox: (x1, y1, x2, y2) - tọa độ của bounding box
    img_size: kích thước của hình ảnh (width, height)
    margin: tỷ lệ phần trăm của khoảng lề
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = img_size

    width = x2 - x1
    height = y2 - y1

    # Tính toán khoảng lề
    dx = width * margin
    dy = height * margin

    # Điều chỉnh tọa độ
    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(img_w, x2 + dx)
    y2 = min(img_h, y2 + dy)

    return [x1, y1, x2, y2]
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# update id2label to include "no object"
id2label = model.config.id2label
id2label[len(model.config.id2label)] = "no object"


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

objects = outputs_to_objects(outputs, image.size, id2label)
print(objects)

### VISUALIZATION
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img


def visualize_detected_tables(img, det_tables, out_path=None):
    plt.imshow(img, interpolation="lanczos")
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    ax = plt.gca()

    img_size = img.size

    for det_table in det_tables:
        bbox = det_table['bbox']
        bbox = adjust_bbox(bbox, img_size)

        if det_table['label'] == 'table':
            facecolor = (1, 0, 0.45)
            edgecolor = (1, 0, 0.45)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        elif det_table['label'] == 'table rotated':
            facecolor = (0.95, 0.6, 0.1)
            edgecolor = (0.95, 0.6, 0.1)
            alpha = 0.3
            linewidth = 2
            hatch='//////'
        else:
            continue

        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor='none',facecolor=facecolor, alpha=0.1)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=linewidth,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', alpha=alpha)
        ax.add_patch(rect)
        rect = patches.Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=0,
                                    edgecolor=edgecolor,facecolor='none',linestyle='-', hatch=hatch, alpha=0.2)
        ax.add_patch(rect)

    plt.xticks([], [])
    plt.yticks([], [])

    legend_elements = [Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45),
                                label='Table', hatch='//////', alpha=0.3),
                        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1),
                                label='Table (rotated)', hatch='//////', alpha=0.3)]
    plt.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0,
                    fontsize=10, ncol=2)
    plt.gcf().set_size_inches(7, 5)
    plt.axis('off')

    if out_path is not None:
      plt.savefig(out_path, bbox_inches='tight', dpi=120)

    return fig

fig = visualize_detected_tables(image, objects)
plt.show()
visualized_image = fig2img(fig)

### CROP TABLES
def objects_to_crops(img, tokens, objects, class_thresholds, padding=10):
    """
    Process the bounding boxes produced by the table detection model into
    cropped table images and cropped tokens.
    """

    table_crops = []
     # Kích thước của ảnh gốc
    img_w, img_h = img.size
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]

        bbox = [
            max(0, bbox[0]),
            max(0, bbox[1]),
            min(img_w, bbox[2]),
            min(img_h, bbox[3])
        ]

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [
                token['bbox'][0]-bbox[0],
                token['bbox'][1]-bbox[1],
                token['bbox'][2]-bbox[0],
                token['bbox'][3]-bbox[1]]

        # If table is predicted to be rotated, rotate cropped image and tokens/words:
        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops


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
plt.axis('off')  # Tắt hiển thị trục
plt.show()
cropped_table.save("table_cropped.png")

### LOAD STRUCTURE RECOGNITION MODEL


# new v1.1 checkpoints require no timm anymore
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
structure_model.to(device)
print("")


structure_transform = transforms.Compose([
    MaxResize(1000),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
pixel_values = structure_transform(cropped_table).unsqueeze(0)
pixel_values = pixel_values.to(device)
print(pixel_values.shape)

# forward pass
with torch.no_grad():
  outputs = structure_model(pixel_values)

# update id2label to include "no object"
structure_id2label = structure_model.config.id2label
structure_id2label[len(structure_id2label)] = "no object"

cells = outputs_to_objects(outputs, cropped_table.size, structure_id2label)
#print(cells)

### VISUALIZATION CELLS
cropped_table_visualized = cropped_table.copy()
draw = ImageDraw.Draw(cropped_table_visualized)

for cell in cells:
    draw.rectangle(cell["bbox"], outline="red")

plt.imshow(cropped_table_visualized)
plt.axis('off')  # Tắt hiển thị trục
plt.show()

### VISUALIZATION ROWS
def plot_results(cells, class_to_visualize):
    if class_to_visualize not in structure_model.config.id2label.values():
      raise ValueError("Class should be one of the available classes")

    plt.figure(figsize=(8,4))
    plt.imshow(cropped_table)
    ax = plt.gca()

    for cell in cells:
        score = cell["score"]
        bbox = cell["bbox"]
        label = cell["label"]

        if label == class_to_visualize:
          xmin, ymin, xmax, ymax = tuple(bbox)

          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=3))
          text = f'{cell["label"]}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')  # Turn off axis
    plt.show() 
          
plot_results(cells, class_to_visualize="table row")

### APPLYING OCR
def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

cell_coordinates = get_cell_coordinates_by_row(cells)
print("Number cell row: ", len(cell_coordinates)) ##  number cell row
print("number column: ", len(cell_coordinates[0]["cells"])) ## number column
print("")

# ##bbox coordinates of each cell in each row
# for row in cell_coordinates:
#    print(row["cells"]) 
   
### Recognition with vietOCR
# cfg = Cfg.load_config_from_file('./config/config_after_trainer.yml')
# cfg['weights'] = './weight/transformerocr.pth'
cfg = Cfg.load_config_from_name('vgg_transformer')
cfg['weights'] = 'https://vocr.vn/data/vietocr/vgg_transformer.pth'
cfg['cnn']['pretrained'] = False
cfg['device'] = 'cpu' 
predictor = Predictor(cfg)

def apply_ocr(cell_coordinates):
    data = dict()
    max_num_columns = 0

    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell in row["cells"]:
            # Crop cell ra khỏi hình ảnh
            cell_image = cropped_table.crop(cell["cell"])
            cell_image = np.array(cell_image)
            cell_image_pil = Image.fromarray(cell_image)

            # Áp dụng OCR
            result = predictor.predict(cell_image_pil)

            if result:
                text = "".join(result)
                row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # Padding các hàng không có đủ số lượng cột
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

# Giả định rằng bạn đã có biến cell_coordinates và cropped_table
data = apply_ocr(cell_coordinates)

for row, row_data in data.items():
    print(row_data)

# ### save and open csv
# with open('output_vietocr.csv','w', encoding='utf-8') as result_file:
#     wr = csv.writer(result_file, dialect='excel')

#     for row, row_text in data.items():
#       wr.writerow(row_text)
df = pd.DataFrame.from_dict(data, orient='index')

# Lưu DataFrame thành file Excel
df.to_excel('output_vietocr.xlsx', index=False)
# Mở lại file Excel vừa tạo để điều chỉnh độ rộng cột
workbook = load_workbook('output_vietocr.xlsx')
worksheet = workbook.active

# Tự động điều chỉnh độ rộng các cột
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

# Lưu lại file Excel sau khi điều chỉnh độ rộng cột
workbook.save('output_vietocr.xlsx')