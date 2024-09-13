<div align="center">

<img alt="table_ocr_logo" src="assets/logo_ocr.png" width=120 height=120>
<h1>TableOCR</h1>

[English](README.md) / Vietnamese

<img src="assets/bg_image.png" width=700>

</div>

## Recognition-Table-with-Table_Transformer-and-vietOCR
- Sự kết hợp giữa Table Transformer và vietOCR cung cấp một giải pháp mạnh mẽ để nhận dạng và trích xuất thông tin từ các bảng trong tài liệu tiếng Việt. 
- Table Transformer đảm nhiệm việc nhận dạng cấu trúc bảng, trong khi vietOCR đảm bảo nhận dạng ký tự quang học chính xác, giúp tái tạo dữ liệu từ các tài liệu dạng bảng hiệu quả và chính xác hơn.

<br>

> **Related Projects**：
>
> - [Vietnamese_OCR_documents](https://github.com/KaiKenju/Vietnamese_OCR_documents): is used to convert text from images or scanned documents into digital format, allowing automatic processing and analysis of text data. This technology is very useful in extracting information from Vietnamese documents, making information searching and management easier and more efficient.

<br>
<br>

# Table of Contents
- [Table of Contents](#table-of-contents)
  - [🛠️ Cài đặt](#️-cài-đặt)
  - [▶️ Khởi chạy](#️-khởi-chạy)
  - [📝 Kết quả](#-kết-quả)
  - [🚀 Cấu trúc Project](#-cấu-trúc-project)
  - [🚀 Tổng Quan](#-tổng-quan)
  - [⚠️ Chú ý](#️-chú-ý)
- [🗃️ Dataset](#️-dataset)
  - [PubTables-1M (TATR)](#pubtables-1m-tatr)
  - [Custome-Vietnamese (vietOCR)](#custome-vietnamese-vietocr)
  - [📚 Tài liệu tham khảo](#-tài-liệu-tham-khảo)
- [📧 Contact Us](#-contact-us)
- [Contributors](#contributors)


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
- Tải pre-train weight: [transformerocr](https://drive.google.com/file/d/1g3-Hi4oigfbrrNFZxQCh5qhEYjZU2_Ar/view?usp=drive_link) và
```[bash]
đặt theo đường dẫn: weight/transformerocr.pth
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
# 🗃️ Dataset
## PubTables-1M (TATR)
[PubTables-1M](https://arxiv.org/pdf/2110.00061) là một tập dữ liệu lớn và chi tiết được thiết kế để đào tạo và đánh giá các mô hình về phát hiện bảng, nhận dạng cấu trúc bảng và các tác vụ phân tích chức năng.
## Custome-Vietnamese (vietOCR)
[VietOCR](https://github.com/pbcquoc/vietocr) là một tập dữ liệu được sử dụng để đào tạo các mô hình OCR của Việt Nam, bao gồm nhiều văn bản có chú thích khác nhau để nhận dạng và trích xuất văn bản theo từng dòng.
## 📚 Tài liệu tham khảo

- https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/table/README.md
- https://github.com/microsoft/table-transformer
- https://viblo.asia/p/deep-learning-table-recognition-simple-is-better-than-complex-bai-toan-tai-cau-truc-du-lieu-bang-bieu-voi-deep-learning-Qbq5QBYLKD8
- https://github.com/pbcquoc/vietocr

<br>

# 📧 Contact Us

Nếu bạn có thắc mắc hãy liên hệ email hiepdv.tb288@gmail.com

<br>

# Contributors

<a href="https://github.com/KaiKenju/Recognition-Table-with-Table_Transformer-and-vietOCR/graphs/contributors">
 <img src="assets/avt-removebg-preview.png" width="100" />
</a>

[Kai-Kenju](https://github.com/KaiKenju)