
🌎 English | [Vietnamese](README_vn.md)
# Recognition-Table-with-Table_Transformer-and-vietOCR
This repository is used to detect and recognize tables using Table Transformer and vietOCR. It is a module developed from my [Vietnamese_OCR_documents](https://github.com/KaiKenju/Vietnamese_OCR_documents) repository

## 🛠️ Setup

- Clone  this project:

```[bash]
git clone https://github.com/KaiKenju/Recognition-Table-with-Table_Transformer-and-vietOCR?tab=readme-ov-file
```

- Initial enviromment with Miniconda:

```[bash]
conda create -n <env_name> python=3.8
```
- Activate conda
```[bash]
conda activate <env_name> 
cd Vietnamese_OCR_documents
```
- Download pre-train weight
download from here: [transformerocr](https://drive.google.com/file/d/1g3-Hi4oigfbrrNFZxQCh5qhEYjZU2_Ar/view?usp=drive_link)
 
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

<p align="center">
  <img src="images/bang2demo.png" alt="Image 1" width="200"/>
  <img src="files/detect_table.png" alt="Image 2" width="200"/>
  <img src="files/table_cropped.png" alt="Image 3" width="200"/>
</p>
<p align="center">
  <strong>Title 1</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Title 2</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Title 3</strong>
</p>

<p align="center">
  <img src="files/detect_cells.png" alt="Image 4" width="200"/>
  <img src="files/detect_cell_row.png" alt="Image 5" width="200"/>
  <img src="files/recognize_cell.png" alt="Image 6" width="200"/>
</p>
<p align="center">
  <strong>Title 4</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Title 5</strong> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <strong>Title 6</strong>
</p>







## 🚀 Structure Project
```[bash]
Vietnamese_OCR_Documents/
          ├── assets/                   # contains image to OCR
          ├── config/                   #  configuration files and options for OCR system
          ├── cropped_images/           # Images are cropped for recognition purposes
          ├── folder_file_api/          # file pdf,word after OCR for web-app using Fast-api
          ├── weight/                   # The weight  of system
          ├── PaddleOCR/                # Paddle repositories
          ├── static/                   # front-end 
          ├── app.py                    # demo web(local-host) using FastAPI 
          ├── Core_OCR.ipynb            # notebook paddleOCR + vietOCR
          ├── image_processing.py       # image processing
          ├── main.py 
          ├── performance_thread.py     # performance optimization (faster main.py using thread)
          ├── Pretrained_vietOCR.ipynb  # training VietOCR
          ├── requirements.txt     
          ├── README.md                 # english version
          ├── README_vn.md              # vietnamese version
```
## 🚀 Detection: PaddleOCR

neww
## ⚡ Recognition: VietOCR

