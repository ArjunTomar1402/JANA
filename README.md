# JANA Phase 1 - Japanese Language Analysis App

A comprehensive Streamlit application for Japanese language processing, including translation, morphological analysis, furigana generation, and data export capabilities.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10%252B-blue?style=for-the-badge&logo=python)

---

## Features

- **Automatic Language Detection:** Identifies Japanese or other languages using FastText  
- **Translation:** Translates non-Japanese text to Japanese  
- **Morphological Analysis:** Detailed parsing of Japanese text using SudachiPy  
- **Furigana Generation:** Optional phonetic annotations for Japanese characters  
- **Data Export:** CSV export functionality for processed results  
- **Processing Metadata:** Logs and metadata viewing capabilities  
- **Flexible Input:** Supports text files and PDF documents  
- **Device Selection:** CPU/GPU toggle for translation (CPU-only on Streamlit Cloud)  

---

## Prerequisites

- Python 3.10 or higher (tested with Python 3.13)  
- pip package manager  

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/jana.git
cd jana
```

### Install required dependencies:

```bash
Copy code
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Project Structure
```
jana/
├── app.py                    # Main application file
├── requirements.txt          # Python dependencies
├── translation_cache.sqlite  # SQLite cache for translations
├── lid.176.ftz               # FastText language detection model
├── jana_app.log              # Application log file
├── .streamlit/               # Streamlit configuration
│   └── config.toml           # Configuration settings
├── .gitignore                # Git ignore rules
├── LICENSE                   # MIT License
└── README.md                 # Project documentation
```

### Usage
Run the application locally:

```bash
Copy code
streamlit run app.py
Open your browser and navigate to the local URL provided (typically http://localhost:8501).
```

### Use the application:

- Upload a text file or PDF for processing

- Toggle furigana generation if needed

- Select translation device (CPU/GPU)

- View processed results

- Export results as CSV if needed

- Check processing metadata

### Configuration
The application includes a Streamlit configuration file (.streamlit/config.toml) with:

Increased upload size limit (200MB)

Watchdog disabled to prevent inotify limits on Streamlit Cloud

### Deployment
```
- Streamlit Cloud Deployment

- Push your code to a GitHub repository

- Connect your repository at Streamlit Cloud

- Streamlit Cloud will automatically detect:

- app.py as the main application file

- requirements.txt for dependency installation

- The SQLite cache (translation_cache.sqlite) will be created automatically
```

#### Notes on Deployment
```
- Streamlit Cloud only supports CPU execution

- For GPU support, deploy on a cloud VM with CUDA support

- SQLite permissions are handled automatically on Streamlit Cloud

- The FastText model (lid.176.ftz) is included in the repository
```
#### Troubleshooting
```
- Watchdog errors: Already handled by configuration (enableWatchdog=false)

- Upload size limits: Increased to 200MB in configuration

- GPU unavailable: Streamlit Cloud only supports CPU execution
```
### Contributing
```
- Fork the repository

- Create a feature branch (git checkout -b feature/amazing-feature)

- Commit your changes (git commit -m 'Add some amazing feature')

- Push to the branch (git push origin feature/amazing-feature)

- Open a Pull Request

- Please ensure your contributions adhere to the existing code style and include appropriate tests.
```
### License
```
This project is licensed under the MIT License - see the LICENSE file for details.
```
### Acknowledgments
```
- Built with Streamlit

- Language detection using FastText

- Japanese processing with SudachiPy

- Translation capabilities powered by Transformers
```
