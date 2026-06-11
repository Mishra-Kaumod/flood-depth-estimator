import os
import shutil
from bing_image_downloader import downloader

def print_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

# The 20 Edge-Case Test Matrix
test_matrix = {
    # Floods (Expected: > 0cm)
    "01_deep_flood_day": "deep urban street flood daylight cars submerged",
    "02_shallow_flood": "shallow water flooding city street",
    "03_muddy_flood": "muddy brown flood water urban street",
    "04_night_flood": "street flood at night reflections",
    "05_rushing_water": "fast rushing flood water city street",
    "06_debris_flood": "flood water floating debris street",
    "07_submerged_car": "car partially submerged in flood water",
    "08_submerged_bus": "bus driving through deep flood water",
    "09_curb_flood": "flood water covering sidewalk curb",
    "10_storm_surge": "hurricane storm surge flooding street",
    
    # Dry / Illusions (Expected: 0.0cm)
    "11_dry_street_day": "empty city street daylight sunny",
    "12_dry_street_night": "empty city street night time dark",
    "13_blue_bus_dry": "large blue electric bus on sunny street",
    "14_small_puddles": "small puddles on street after rain",
    "15_wet_asphalt": "wet shiny asphalt road no flood",
    "16_dark_shadows": "heavy dark shadows on empty road",
    "17_traffic_jam": "heavy traffic jam dry road",
    "18_blue_tarp": "large blue tarp construction on street",
    "19_raindrops_lens": "raindrops on camera lens looking at street",
    "20_snow_street": "snow covered city street driving"
}

def build_dataset():
    print_header("INITIATING BING WEB SCRAPER: 20-SCENARIO MATRIX")
    
    final_folder = "test_images"
    temp_folder = "downloads"
    
    # Clean up existing folders
    os.makedirs(final_folder, exist_ok=True)
    for f in os.listdir(final_folder):
        os.remove(os.path.join(final_folder, f))
        
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    # Scrape 1 image per scenario
    for case_name, search_query in test_matrix.items():
        print(f"\n[*] Scraping: {case_name}...")
        try:
            downloader.download(search_query, limit=1, output_dir=temp_folder, adult_filter_off=True, force_replace=True, timeout=10, verbose=False)
            
            # Find the downloaded file in the Bing subfolder
            query_folder = os.path.join(temp_folder, search_query)
            downloaded_files = os.listdir(query_folder)
            
            if downloaded_files:
                original_file = os.path.join(query_folder, downloaded_files[0])
                # Ensure it has a standard extension
                ext = os.path.splitext(original_file)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png']:
                    ext = '.jpg' 
                    
                final_path = os.path.join(final_folder, f"{case_name}{ext}")
                shutil.move(original_file, final_path)
                print(f"    -> [SUCCESS] Saved as {case_name}{ext}")
        except Exception as e:
            print(f"    -> [FAILED] Could not download {case_name}: {e}")

    # Clean up the temp folder Bing creates
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder)

    print("\n[+] Dataset built! Run 'python terminal_test.py' to evaluate.")

if __name__ == "__main__":
    build_dataset()
