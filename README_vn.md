
🌎 [English] | Vietnamese(README.md)
# Recognition-Table-with-Table_Transformer-and-vietOCR
Repo này được phát triển và sử dụng để phát hiện và nhận dạng các bảng bằng Table Transformer và vietOCR. Đây là một mô-đun được phát triển từ [Vietnamese_OCR_documents](https://github.com/KaiKenju/Vietnamese_OCR_documents) repository của tôi.

## 🛠️ Cài đặt

- Clone  project:

```[bash]
git clone https://github.com/KaiKenju/Recognition-Table-with-Table_Transformer-and-vietOCR.git
```

- Khởi tạo môi trường với Miniconda:

```[bash]
conda create -n <env_name> python=3.8
```
- Kích hoạt conda
```[bash]
conda activate <env_name> 
```
- Tải pre-train weight
download from here: [transformerocr](https://drive.google.com/file/d/1g3-Hi4oigfbrrNFZxQCh5qhEYjZU2_Ar/view?usp=drive_link) và
```[bash]
Tải weight file và đặt theo đường dẫn: weight/transformerocr.pth
```
- Run the commands:
```[bash]
pip install -r requirements.txt
```

## ▶️ Khởi chạy
* 🔥 Nếu bạn muốn hiểu cách mà mô hình hoạt động, từng bước thì chạy:
```[bash]
python run.py
```
* ✅ còn nếu ko thì:
```[bash]
python main.py
```

## 📝 Kết quả

<p align="center" >
  <img src="images/bang2demo.png" alt="Image 1" width="200" style="margin-right: 25px;"/>
  <img src="files/detect_table.png" alt="Image 2" width="200" style="margin-right: 25px;"/>
  <img src="files/table_cropped.png" alt="Image 3" width="200" />
</p>
<p align="center">
  <strong>Input </strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Phát hiện bảng </strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Cắt bảng </strong>
</p>

<p align="center">
  <img src="files/detect_cells.png" alt="Image 4" width="200" style="margin-right: 25px;"/>
  <img src="files/detect_cell_row.png" alt="Image 5" width="200" style="margin-right: 25px;"/>
  <img src="files/recognize_cell.png" alt="Image 6" width="200" />
</p>
<p align="center">
  <strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</strong>
  <strong>Phát hiện các cells</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>&nbsp;&nbsp;&nbsp;&nbsp; Phát hiện từng dòng một</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Nhận diện từng dòng</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>



## 🚀 Cấu trúc Project
```[bash]
Recognition-Table-with-Table_Transformer-and-vietOCR/
          ├── config/                   # cấu hình và tùy chọn cho hệ thống OCR
          ├── files/                    # 
          ├── images/                   # ảnh để thử nghiệm
          ├── output/                   # đầu ra  excel file .xlsx
          ├── weight/                   # file weight mô hình
          ├── TATR.ipynb                # core model
          ├── Core_OCR.ipynb            # notebook Table transformer + vietOCR/EasyOCR
          ├── main.py                   # 
          ├── pre_processing.py         # pre-processing
          ├── run.py                    # chạy để hiểu mô hình hoạt động như nào
          ├── requirements.txt     
          ├── README.md                 # bản tiếng anh
          ├── README_vn.md              # bản tiếng việt
```
## 🚀 Tổng Quan
Dự án này chủ yếu chứa ba mô hình

- Table detection-Table Transformer(microsoft/table-transformer-detection)
- Table Structure Recognition-(microsoft/table-structure-recognition-v1.1-all) 
- Single line text recognition-vietOCR

Biểu đồ luồng nhận dạng bảng như sau
<img src="files\overview_table.jpg" alt="Image 1" width="80%"/>

## ⚠️ Chú ý

- ✅ Mô hình này được tối ưu hóa cho các bảng có văn bản một dòng.
- ⚠️ Hiệu suất có thể giảm đáng kể đối với các bảng chứa nhiều dòng văn bản trong một ô.
- 🚫 Không hỗ trợ hợp nhất các ô
- ⚠️ Văn bản nhiều dòng trong một ô có thể gây ra lỗi khi trích xuất dữ liệu từ một bảng.
- ✅ Sử dụng mô hình này với một bảng đơn giản, mỗi ô chứa một dòng văn bản.
- ✅ Chỉ hỗ trợ tiếng Việt và tiếng Anh

## 📚 Tài liệu tham khảo

- https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/table/README.md
- https://github.com/microsoft/table-transformer
- https://viblo.asia/p/deep-learning-table-recognition-simple-is-better-than-complex-bai-toan-tai-cau-truc-du-lieu-bang-bieu-voi-deep-learning-Qbq5QBYLKD8
- https://github.com/pbcquoc/vietocr

