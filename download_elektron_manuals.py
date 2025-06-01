import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import os
import re

def download_image(url, timeout=10):
    """Download image with error handling"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Failed to download {url} - Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def save_images_as_pdf(images, pdf_filename):
    """Convert images to PDF"""
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    
    for image in images:
        if image is None:
            continue
            
        image_width, image_height = image.size
        aspect_ratio = image_width / image_height
        
        # Scale to fit page
        new_width = width
        new_height = width / aspect_ratio
        
        if new_height > height:
            new_height = height
            new_width = height * aspect_ratio
        
        # Convert PIL image to byte stream
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Draw image
        c.drawImage(
            ImageReader(img_byte_arr), 
            (width - new_width) / 2,  # Center horizontally
            height - new_height, 
            width=new_width, 
            height=new_height
        )
        c.showPage()
    
    c.save()
    print(f"✅ PDF saved as: {pdf_filename}")

def find_manual_id(manual_url):
    """Extract manual ID from the webpage source"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(manual_url, headers=headers)
        
        # Look for viewer URL pattern in the page source
        viewer_pattern = r'/viewer/(\d+)/'
        match = re.search(viewer_pattern, response.text)
        
        if match:
            return match.group(1)
        else:
            print(f"Could not find manual ID for {manual_url}")
            return None
    except Exception as e:
        print(f"Error finding manual ID for {manual_url}: {str(e)}")
        return None

def download_elektron_manual(manual_url, total_pages, output_name):
    """Download manual from manua.ls"""
    # First, find the manual ID
    manual_id = find_manual_id(manual_url)
    if not manual_id:
        print(f"❌ Could not find manual ID for {output_name}")
        return False
    
    print(f"📥 Found manual ID {manual_id} for {output_name}")
    
    base_url = f"https://www.manua.ls/viewer/{manual_id}/"
    images = []
    
    print(f"⬇️  Downloading {output_name} ({total_pages} pages)...")
    
    # Download images with progress bar
    for i in tqdm(range(1, total_pages + 1), desc=f"Downloading {output_name}"):
        # Main expected pattern: lowercase hex, no padding for the hex part if it's single digit
        # e.g., /viewer/manual_id/page_number/bg[hex_page_number_lower].png
        # hex(i)[2:] correctly gives 'f' for 15, '10' for 16 etc.
        hex_i_lower = hex(i)[2:].lower()
        url = f"{base_url}{i}/{hex_i_lower}.png" # Common pattern observed where filename is just hex page num
        
        image = download_image(url)
        if image:
            images.append(image)
            continue # Got the image, move to next page
        
        # Try with 'bg' prefix if the simple hex name didn't work
        url_with_bg = f"{base_url}{i}/bg{hex_i_lower}.png"
        image = download_image(url_with_bg)
        if image:
            images.append(image)
            continue

        # Define more alternative URL patterns if the primary ones fail
        # These patterns try different common variations for image naming on such sites
        alt_patterns = [
            f"{base_url}{i}/page{i}.png",                        # page[decimal_page_number].png
            f"{base_url}{i}/page_{i}.png",                       # page_[decimal_page_number].png
            f"{base_url}{i}/{i}.png",                             # [decimal_page_number].png
            f"{base_url}{i}/image{i}.png",                       # image[decimal_page_number].png
            f"{base_url}{i}/img{i}.png",                         # img[decimal_page_number].png
            f"{base_url}{i:03d}/bg{hex_i_lower}.png",             # [3_digit_padded_page_number]/bg[hex_page_number_lower].png
            f"{base_url}{i:03d}/{hex_i_lower}.png",              # [3_digit_padded_page_number]/[hex_page_number_lower].png
            f"{base_url}{i}/page{hex_i_lower}.png",               # page[hex_page_number_lower].png
            f"{base_url}{i}/bg{i}.png",                           # bg[decimal_page_number].png (already tried similar one but good to have explicitly)
            f"{base_url}{i}/bg{hex_i_lower.upper()}.png",       # bg[hex_page_number_UPPER].png
            f"{base_url}{i}/BG{hex_i_lower.upper()}.png",       # BG[hex_page_number_UPPER].png
            f"{base_url}{i}/{i:02d}.png",                        # [2_digit_padded_page_number].png
            f"{base_url}{i}/{i:03d}.png",                        # [3_digit_padded_page_number].png
            f"{base_url}{i}/image_{i:03d}.png",                   # image_[3_digit_padded_page_number].png
        ]
            
        for alt_url in alt_patterns:
            image = download_image(alt_url)
            if image:
                images.append(image)
                break # Found image with an alternative pattern
        else: # If loop completed without break (no alt pattern worked)
            print(f"\n⚠️ Failed to download page {i} for {output_name} after trying multiple patterns.")
            print(f"   Last attempted primary URL: {url_with_bg}")
            # To aid debugging, you could print all attempted alt_urls here if needed
    
    if images:
        # Create manuals directory
        os.makedirs("manuals", exist_ok=True)
        pdf_filename = f"manuals/{output_name}.pdf"
        save_images_as_pdf(images, pdf_filename)
        print(f"✅ Successfully downloaded {len(images)}/{total_pages} pages for {output_name}")
        return True
    else:
        print(f"❌ Failed to download any pages for {output_name}")
        return False

def main():
    """Download all popular Elektron manuals"""
    
    print("🎵 Elektron Manual Downloader")
    print("=============================")
    
    # List of manuals to download
    # Format: (manual_url, total_pages, output_filename)python download_elektron_manuals.py
    elektron_manuals = [
        ("https://www.manua.ls/elektron/digitakt/manual", 96, "elektron_digitakt"),
        ("https://www.manua.ls/elektron/digitone/manual", 104, "elektron_digitone"), 
        ("https://www.manua.ls/elektron/analog-rytm-mkii/manual", 150, "elektron_analog_rytm_mkii"),  # Estimated pages
        ("https://www.manua.ls/elektron/overbridge/manual", 50, "elektron_overbridge"),
        ("https://www.manua.ls/elektron/syntakt/manual", 120, "elektron_syntakt"),  # Estimated pages
        ("https://www.manua.ls/elektron/model-samples/manual", 80, "elektron_model_samples"),  # Estimated pages
        ("https://www.manua.ls/elektron/octatrack-mkii/manual", 146, "elektron_octatrack_mkii"),
    ]
    
    print(f"📚 Found {len(elektron_manuals)} manuals to download\n")
    
    successful_downloads = 0
    failed_downloads = 0
    
    # Download each manual
    for manual_url, pages, name in elektron_manuals:
        print(f"\n🔄 Processing: {name}")
        try:
            success = download_elektron_manual(manual_url, pages, name)
            if success:
                successful_downloads += 1
                print(f"✅ {name} completed successfully")
            else:
                failed_downloads += 1
                print(f"❌ {name} failed")
        except Exception as e:
            failed_downloads += 1
            print(f"❌ {name} failed with error: {str(e)}")
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"✅ Successful: {successful_downloads}")
    print(f"❌ Failed: {failed_downloads}")
    print(f"📁 PDFs saved to: ./manuals/")
    
    if successful_downloads > 0:
        print(f"\n🎉 Ready to upload to your Music Gear Chat app!")

if __name__ == "__main__":
    main()