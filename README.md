# NCZarr Viewer

### A Modern, Interactive NetCDF and Zarr Dataset Viewer

A powerful web application for exploring and visualizing geospatial NetCDF and Zarr datasets with an intuitive interface. Built with Dash and optimized for both local development and cloud deployment.

## ✨ Features

- **🌍 Multi-Format Support**: NetCDF, Zarr datasets
- **☁️ Cloud Storage**: Direct S3, Minio, and cloud storage access
- **🌊 Marine Data**: Copernicus Marine Service (CMEMS) integration
- **📊 Interactive Visualization**: Dynamic plotting with Plotly and Matplotlib
- **🔍 Smart Data Exploration**: Variable browsing, dimension filtering, and statistics
- **🐳 Docker Ready**: Containerized for easy deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ 
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Option 1: Using uv (Recommended)
```bash
# Clone the repository
git clone https://github.com/samuelfooks/nczarr-viewer.git
cd nczarr-viewer

# Install dependencies and run
uv sync
python run.py
```

### Option 2: Using pip
```bash
# Clone the repository
git clone https://github.com/samuelfooks/nczarr-viewer.git
cd nczarr-viewer

# Install in development mode
pip install -e .

# Run the viewer
python run.py
```

The viewer will be available at **http://localhost:8050**

## 📁 Supported Data Sources

- **Local Files**: NetCDF, Zarr, HDF5, GRIB
- **Cloud Storage**: S3, Google Cloud Storage, Azure Blob
- **Personal Cloud Storage**: [Minio storage](https://datalab.dive.edito.eu/file-explorer) on EDITO
- **Marine Data**: Copernicus Marine Service (CMEMS)
- **EDITO Integration**: STAC catalogs and ARCO data

## 🔧 Configuration

### S3/Cloud Storage Access
For NetCDF files stored on S3-compatible storage (like Minio), add `#mode=bytes` to the end of your URL:

```
https://minio.lab.dive.edito.eu/oidc-YOURUSERNAME/folder/mynetcdf.nc#mode=bytes
```

This ensures proper binary data access for NetCDF files in object storage.

### CMEMS Integration
To access CMEMS datasets, you may need an account using [Copernicus Marine Toolbox](https://help.marine.copernicus.eu/en/articles/7949409-copernicus-marine-toolbox-introduction#h_9172b5c79a):

```bash
# Set environment variables
export CMEMS_USERNAME=your_username
export CMEMS_PASSWORD=your_password
```

## 🐳 Docker Deployment

### Quick Docker Run
```bash
# Pull and run the latest image
docker run -p 8050:8050 samfooks/nczarr-viewer:latest
```

### With Local Files
```bash
# Mount a local directory with NetCDF files
docker run -v $(pwd)/myfiles:/app/myfiles -p 8050:8050 \
  samfooks/nczarr-viewer:latest
```

### Accessing Data
Once the container is running:
1. Open your browser to `http://localhost:8050`
2. Enter the URL or path to your dataset:
   - **Local files**: Use `/app/myfiles/yourfile.nc` (if mounted)
   - **S3/Cloud storage**: Use the full URL with `#mode=bytes` if needed
   - **Zarr datasets**: Use the direct Zarr store URL

## 🏗️ Development

### Project Structure
```
nczarr-viewer/
├── src/                    # Source code
│   ├── main.py               # Main Dash application
│   ├── data.py               # Data loading and processing
│   ├── variables.py           # Variable selection logic
│   ├── dimension.py           # Dimension handling
│   └── assets/               # CSS and static assets
├── docs/                   # Documentation
├── run.py                  # Local development runner
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Locked dependency versions
└── Dockerfile              # Container configuration
```

### Development Workflow
1. **Install dependencies**: `uv sync`
2. **Make changes**: Edit files in `src/`
3. **Test locally**: `python run.py`
4. **Auto-reload**: Changes automatically reload in debug mode

### Key Dependencies
- **Dash**: Web framework for building analytical web applications
- **xarray**: N-D labeled arrays and datasets
- **zarr**: Chunked, compressed, N-dimensional arrays
- **plotly**: Interactive plotting library
- **cartopy**: Cartographic projections and transformations
- **copernicusmarine**: Access to Copernicus Marine data

## 📚 Documentation

- **API Reference**: See `docs/source/` for detailed documentation
- **Presentation**: Check `docs/nczarr_viewer_presentation.md` for usage examples
- **Examples**: Explore `docs/explore_data/` for sample datasets and workflows

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Samuel Fooks** - [samuel.fooks@gmail.com](mailto:samuel.fooks@gmail.com)

- **GitHub**: [https://github.com/samuelfooks/nczarr-viewer](https://github.com/samuelfooks/nczarr-viewer)
- **Docker Hub**: [samfooks/nczarr-viewer](https://hub.docker.com/r/samfooks/nczarr-viewer)

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/samuelfooks/nczarr-viewer/issues)
- **Documentation**: Check the `docs/` folder
- **Examples**: See `docs/explore_data/` for sample workflows

---

**Happy Data Exploring! 🌊📊**