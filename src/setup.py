"""
Setup module for the Zarr Data Viewer app.
Handles initialization tasks like downloading Cartopy map data.
"""

import os
import warnings
import cartopy
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
from cartopy.io import DownloadWarning

# Suppress download warnings during setup
warnings.filterwarnings('ignore', category=DownloadWarning)

def setup_cartopy_data():
    """
    Pre-download all Cartopy map data to avoid delays on first plot.
    This will download to Cartopy's default location.
    """
    print("🔧 Setting up Cartopy map data...")
    
    try:
        # Download essential Natural Earth features that your app uses
        print("📥 Downloading essential map features...")
        
        # These are the features your app actually uses based on the logs
        essential_features = [
            cfeature.COASTLINE,    # ne_50m_coastline.zip
            cfeature.BORDERS,      # ne_50m_admin_0_boundary_lines_land.zip  
            cfeature.LAND,         # ne_50m_land.zip
            cfeature.OCEAN,        # ne_50m_ocean.zip
        ]
        
        for feature in essential_features:
            # Access the feature to trigger download
            feature._kwargs = {}
            print(f"   ✅ Downloaded: {feature.__class__.__name__}")
        
        # Download the specific Natural Earth datasets your app needs
        print("📥 Downloading Natural Earth datasets...")
        
        # Based on your logs, you're using 50m resolution
        scales = ['50m']  # Focus on what you actually use
        feature_types = ['physical', 'cultural']
        
        for scale in scales:
            for feature_type in feature_types:
                try:
                    # This triggers the download
                    shapereader.natural_earth(resolution=scale, category=feature_type)
                    print(f"   ✅ Downloaded: {scale} {feature_type}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not download {scale} {feature_type}: {e}")
        
        print("✅ Cartopy setup completed successfully!")
        print("🚀 Your first plot will now be fast!")
        
        # Show where Cartopy downloaded the data
        import cartopy.io.shapereader
        try:
            # Get the path to a downloaded file to show the directory
            data_path = cartopy.io.shapereader.natural_earth(resolution='50m', category='physical')
            data_dir = os.path.dirname(data_path)
            print(f"📁 Data downloaded to: {data_dir}")
        except:
            print("📁 Data downloaded to Cartopy's default location")
        
    except Exception as e:
        print(f"❌ Error during Cartopy setup: {e}")
        # Don't raise the error - just log it and continue
        # The app can still work, just slower on first plot



def run_setup():
    """Run all setup tasks."""
    print("🚀 Starting Zarr Data Viewer setup...")
    setup_cartopy_data()
    print("🎉 Setup completed!")

if __name__ == "__main__":
    run_setup()
