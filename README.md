<div align="center">

<img alt="table_ocr_logo" src="assets/logo_ocr.png" width=120 height=120>
<h1>TableOCR</h1>

English / [Vietnamese](README_vn.md) / 

<img src="assets/bg_image.png" width=700>

</div>

## Recognition-Table-with-Table_Transformer-and-vietOCR
- The combination of Table Transformer and vietOCR provides a powerful solution for recognizing and extracting information from tables in Vietnamese documents. 
- Table Transformer takes care of recognizing the table structure, while vietOCR ensures accurate optical character recognition, helping to reproduce data from tabular documents more efficiently and accurately.

<br>

> **Related Projects**：
>
> - [Vietnamese_OCR_documents](https://github.com/KaiKenju/Vietnamese_OCR_documents): is used to convert text from images or scanned documents into digital format, allowing automatic processing and analysis of text data. This technology is very useful in extracting information from Vietnamese documents, making information searching and management easier and more efficient.

<br>
<br>

# Table of Contents
- [Table of Contents](#table-of-contents)
  - [🛠️ Setup](#️-setup)
  - [▶️ Run](#️-run)
  - [📝 Result](#-result)
  - [🚀 Structure Project](#-structure-project)
  - [🚀 Overview](#-overview)
  - [⚠️ Pay-Attention](#️-pay-attention)
- [🗃️ Dataset](#️-dataset)
  - [PubTables-1M (TATR)](#pubtables-1m-tatr)
  - [Custome-Vietnamese (vietOCR)](#custome-vietnamese-vietocr)
  - [📚 References](#-references)
- [📧 Contact Us](#-contact-us)
- [Contributors](#contributors)

## 🛠️ Setup

- Clone  this project:

```[bash]
git clone https://github.com/KaiKenju/Recognition-Table-with-Table_Transformer-and-vietOCR.git
```

- Initial enviromment with Miniconda:

```[bash]
conda create -n <env_name> python=3.8
```
- Activate conda
```[bash]
conda activate <env_name> 
```
- Download pre-train weight: [transformerocr.pth](https://drive.google.com/file/d/1g3-Hi4oigfbrrNFZxQCh5qhEYjZU2_Ar/view?usp=drive_link) and following step:

```[bash]
replace it following path: weight/transformerocr.pth
```
 
- Run the commands:
```[bash]
pip install -r requirements.txt
```

## ▶️ Run
* 🔥 if you want to understand how the system works, please run:
```[bash]
python main.py
```
* ✅ else, end to end process:
```[bash]
python run.py
```

## 📝 Result

<p align="center" >
  <img src="images/bang2demo.png" alt="Image 1" width="200" style="margin-right: 25px;"/>
  <img src="files/detect_table.png" alt="Image 2" width="200" style="margin-right: 25px;"/>
  <img src="files/table_cropped.png" alt="Image 3" width="200" />
</p>
<p align="center">
  <strong>Original Image </strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Detect Table </strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Cropped Table </strong>
</p>

<p align="center">
  <img src="files/detect_cells.png" alt="Image 4" width="200" style="margin-right: 25px;"/>
  <img src="files/detect_cell_row.png" alt="Image 5" width="200" style="margin-right: 25px;"/>
  <img src="files/recognize_cell.png" alt="Image 6" width="200" />
</p>
<p align="center">
  <strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</strong>
  <strong>Detect Cells</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>&nbsp;&nbsp;&nbsp;&nbsp; Detect row by row</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Recognition each cell</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>



## 🚀 Structure Project
```[bash]
Recognition-Table-with-Table_Transformer-and-vietOCR/
          ├── config/                   # configuration files and options for OCR system
          ├── files/                    # file pdf,word after OCR for web-app using Fast-api
          ├── images/                   # images to test
          ├── output/                   # output excel file .xlsx
          ├── weight/                   # weight file
          ├── TATR.ipynb                # core model
          ├── Core_OCR.ipynb            # notebook paddleOCR + vietOCR
          ├── main.py                   # 
          ├── pre_processing.py         # pre-processing
          ├── run.py                    # how to model work well
          ├── requirements.txt     
          ├── README.md                 # english version
          ├── README_vn.md              # vietnamese version
```
## 🚀 Overview
The project mainly contains three models

- Table detection-Table Transformer(microsoft/table-transformer-detection)
- Table Structure Recognition-(microsoft/table-structure-recognition-v1.1-all) 
- Single line text recognition-vietOCR

The table recognition flow chart is as follows
<img src="files\overview_table.jpg" alt="Image 1" width="80%"/>

## ⚠️ Pay-Attention

- ✅ This model is optimized for tables with single line text.
- ⚠️ Performance can be significantly reduced for tables containing multiple lines of text in a cell.
- 🚫 Merge cells not supported
- ⚠️ Multi-line text in a cell can cause errors when extracting data from a table.
- ✅ Use this model with a simple table, each cell containing one line of text.
- ✅ Only supports Vietnamese and English
- 
# 🗃️ Dataset
## PubTables-1M (TATR)
[PubTables-1M](https://arxiv.org/pdf/2110.00061) is a large and detailed dataset designed for training and evaluating models on table detection, table structure recognition, and functional analysis tasks.
## Custome-Vietnamese (vietOCR)
[VietOCR](https://github.com/pbcquoc/vietocr) is a dataset used to train Vietnamese OCR models, including various annotated texts for line-by-line text recognition and extraction.

## 📚 References

- https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/table/README.md
- https://github.com/microsoft/table-transformer
- https://viblo.asia/p/deep-learning-table-recognition-simple-is-better-than-complex-bai-toan-tai-cau-truc-du-lieu-bang-bieu-voi-deep-learning-Qbq5QBYLKD8
- https://github.com/pbcquoc/vietocr

<br>

# 📧 Contact Us

If you have any questions, please email hiepdv.tb288@gmail.com

<br>

# Contributors

<a href="https://github.com/KaiKenju/Recognition-Table-with-Table_Transformer-and-vietOCR/graphs/contributors">
 <img src="assets/avt-removebg-preview.png" width="100" />
</a>

[Kai-Kenju](https://github.com/KaiKenju)





