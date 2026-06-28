#!/usr/bin/env python3
"""
Create 10 Edge Case Test Images for Flood Detection Model
Covers: night scenes, rain, occlusion, reflections, dry roads, etc.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

class EdgeCaseImageGenerator:
    """Generate synthetic edge case images for testing."""
    
    def __init__(self, output_dir="test_images/batch_upload"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all_images(self):
        """Generate all 10 edge case images."""
        images = {
            1: ("urban_flooded_day", self.urban_flooded_day),
            2: ("urban_flooded_night", self.urban_flooded_night),
            3: ("rural_flooded_water", self.rural_flooded_water),
            4: ("wet_road_no_flood", self.wet_road_no_flood),
            5: ("heavy_rain_reflection", self.heavy_rain_reflection),
            6: ("occluded_water_debris", self.occluded_water_debris),
            7: ("shallow_water_edge", self.shallow_water_edge),
            8: ("deep_water_dark", self.deep_water_dark),
            9: ("specular_reflection_wet", self.specular_reflection_wet),
            10: ("barren_dry_dark", self.barren_dry_dark),
        }
        
        results = []
        for idx, (name, generator) in images.items():
            img = generator()
            path = self.output_dir / f"{idx:02d}_{name}.jpg"
            cv2.imwrite(str(path), img)
            print(f"✅ Created: {path}")
            results.append({
                'id': idx,
                'name': name,
                'path': str(path),
                'expected_flood': 1 if idx in [1, 2, 3, 6, 8] else 0,
                'expected_depth_cm': [50, 45, 80, 0, 0, 35, 15, 60, 0, 0][idx-1]
            })
        
        return results
    
    def urban_flooded_day(self):
        """Urban street flooded with daylight - HIGH CONFIDENCE POSITIVE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Gray road
        
        # Add water body (blue)
        img[250:450, 100:600] = [100, 150, 200]  # Water
        
        # Add buildings (gray/brown)
        img[0:200, 0:200] = [150, 130, 110]
        img[0:200, 400:640] = [160, 140, 120]
        
        # Add reflections in water
        img[300:400, 200:400] = [120, 170, 210]
        
        # Add some trash/debris
        cv2.rectangle(img, (200, 350), (250, 380), (100, 100, 100), -1)
        
        # Add watermark text
        cv2.putText(img, "Urban Flooded Day", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        return img
    
    def urban_flooded_night(self):
        """Urban street flooded at night - CHALLENGING CASE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 30  # Dark
        
        # Add water body (dark blue)
        img[250:450, 100:600] = [60, 80, 100]
        
        # Add street lights reflection
        cv2.circle(img, (200, 300), 50, (200, 220, 255), -1)
        cv2.circle(img, (450, 320), 50, (180, 200, 240), -1)
        
        # Buildings (very dark)
        img[0:180, 0:150] = [40, 30, 20]
        img[0:180, 500:640] = [45, 35, 25]
        
        # Add watermark
        cv2.putText(img, "Urban Flooded Night", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 1)
        
        return img
    
    def rural_flooded_water(self):
        """Rural area with natural water body - HIGH CONFIDENCE POSITIVE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Base color
        
        # Sky (blue)
        img[0:150, :] = [200, 200, 100]  # Sky
        
        # Water body (dark blue-green)
        img[200:480, :] = [80, 120, 100]  # Natural water
        
        # Trees
        for x in [100, 250, 400, 550]:
            cv2.circle(img, (x, 150), 40, (50, 100, 30), -1)
        
        # Add ripples
        cv2.circle(img, (320, 320), 30, (100, 140, 120), 2)
        cv2.circle(img, (350, 340), 20, (100, 140, 120), 1)
        
        cv2.putText(img, "Rural Flooded Water", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        return img
    
    def wet_road_no_flood(self):
        """Wet road after rain - NO FLOODING (confusion case)"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 150  # Road surface
        
        # Add wet appearance (slightly darker streaks)
        img[200:350, 100:600] = [130, 130, 140]  # Wet asphalt
        
        # Add road markings
        cv2.line(img, (100, 240), (600, 240), (255, 255, 255), 3)
        cv2.line(img, (100, 400), (600, 400), (255, 255, 255), 3)
        
        # Add puddles (not connected - shallow water)
        cv2.circle(img, (200, 300), 30, (100, 120, 150), -1)
        cv2.circle(img, (450, 280), 25, (100, 120, 150), -1)
        
        # Wet shine/reflection
        cv2.circle(img, (200, 290), 15, (180, 200, 220), -1)
        
        cv2.putText(img, "Wet Road - No Flood", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        return img
    
    def heavy_rain_reflection(self):
        """Heavy rain with specular reflections - EDGE CASE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 120
        
        # Base (wet road)
        img[200:450, :] = [100, 110, 130]
        
        # Rain streaks
        for _ in range(50):
            x1, y1 = np.random.randint(0, 640), np.random.randint(0, 250)
            x2, y2 = x1 + np.random.randint(-10, 10), y1 + 40
            cv2.line(img, (x1, y1), (x2, y2), (200, 210, 220), 1)
        
        # Strong reflections (sky reflection)
        cv2.ellipse(img, (320, 250), (150, 80), 0, 0, 180, (200, 210, 240), -1)
        
        # Road edges
        img[450:480, :] = [150, 150, 150]
        
        cv2.putText(img, "Heavy Rain + Reflection", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return img
    
    def occluded_water_debris(self):
        """Water with debris occlusion - MODERATE CONFIDENCE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 150
        
        # Base (urban area)
        img[100:400, :] = [140, 140, 140]
        
        # Water patches (visible)
        img[250:350, 100:200] = [80, 120, 160]
        img[280:380, 400:550] = [85, 125, 165]
        
        # Debris/trash covering water
        cv2.rectangle(img, (120, 270), (170, 320), (80, 80, 80), -1)
        cv2.rectangle(img, (420, 300), (480, 360), (90, 90, 90), -1)
        cv2.circle(img, (300, 200), 40, (100, 100, 100), -1)
        
        # Visible water edges
        cv2.line(img, (100, 250), (200, 250), (80, 120, 160), 3)
        cv2.line(img, (400, 280), (550, 280), (85, 125, 165), 3)
        
        cv2.putText(img, "Water + Debris Occlusion", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        return img
    
    def shallow_water_edge(self):
        """Shallow water at edge - LOW CONFIDENCE BOUNDARY"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 160
        
        # Dry area (brown/sandy)
        img[0:320, :] = [180, 160, 140]
        
        # Shallow water line
        img[320:350, :] = [120, 140, 170]
        
        # Very shallow water (transition)
        img[350:380, :] = [140, 160, 190]
        
        # Mud/sand at waterline
        img[380:420, :] = [160, 140, 120]
        
        # Darker water (deeper)
        img[420:480, :] = [100, 120, 150]
        
        # Water ripples
        cv2.ellipse(img, (320, 400), (200, 30), 0, 0, 180, (110, 130, 160), 1)
        
        cv2.putText(img, "Shallow Water Edge", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        return img
    
    def deep_water_dark(self):
        """Deep dark water - HIGH CONFIDENCE POSITIVE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 40
        
        # Dark water
        img[:, :] = [50, 70, 90]
        
        # Darker patches (depth variation)
        img[100:200, 100:300] = [30, 50, 70]
        img[300:400, 350:550] = [35, 55, 75]
        
        # Slight reflections (ripples)
        cv2.circle(img, (320, 240), 80, (70, 90, 110), 1)
        cv2.circle(img, (320, 240), 120, (60, 80, 100), 1)
        
        # Flooded street elements visible underwater
        cv2.line(img, (150, 200), (200, 250), (60, 80, 100), 3)
        
        cv2.putText(img, "Deep Dark Water", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,100), 2)
        
        return img
    
    def specular_reflection_wet(self):
        """Specular reflection (mirror surface) - NEGATIVE CASE"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200
        
        # Base (wet surface)
        img[200:400, :] = [180, 180, 190]
        
        # Very bright specular reflection (sky)
        cv2.ellipse(img, (320, 250), (180, 100), 0, 0, 180, (230, 230, 250), -1)
        
        # Horizontal reflection gradient
        for i in range(250, 400):
            brightness = int(180 + (i - 250) * 0.1)
            img[i, :] = [brightness, brightness, brightness + 10]
        
        # Some dark areas (trees/buildings reflection)
        cv2.rectangle(img, (50, 220), (120, 280), (100, 100, 120), -1)
        cv2.rectangle(img, (550, 240), (610, 290), (110, 110, 130), -1)
        
        cv2.putText(img, "Specular Reflection", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        return img
    
    def barren_dry_dark(self):
        """Barren dark dry area - FALSE POSITIVE RISK"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 70
        
        # Dark barren ground
        img[:, :] = [80, 80, 90]
        
        # Darker patches (shadows, rocks)
        img[100:200, 100:300] = [50, 50, 70]
        img[300:400, 350:550] = [60, 60, 80]
        
        # Some texture (cracks, rocks)
        cv2.line(img, (100, 100), (300, 200), (70, 70, 85), 2)
        cv2.line(img, (350, 300), (550, 350), (75, 75, 90), 2)
        cv2.circle(img, (200, 150), 20, (60, 60, 75), -1)
        cv2.circle(img, (450, 250), 25, (65, 65, 80), -1)
        
        cv2.putText(img, "Barren Dry Dark Area", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)
        
        return img


def create_metadata_csv(images):
    """Create CSV manifest for batch upload."""
    import csv
    
    csv_path = Path("test_images/batch_upload/manifest.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'filename', 'expected_flood', 'expected_depth_cm', 'scene_type', 'difficulty'])
        writer.writeheader()
        
        difficulty_map = {
            1: 'easy',      # urban_flooded_day
            2: 'hard',      # urban_flooded_night
            3: 'easy',      # rural_flooded_water
            4: 'hard',      # wet_road_no_flood
            5: 'hard',      # heavy_rain_reflection
            6: 'medium',    # occluded_water_debris
            7: 'hard',      # shallow_water_edge
            8: 'easy',      # deep_water_dark
            9: 'hard',      # specular_reflection_wet
            10: 'medium',   # barren_dry_dark
        }
        
        for img in images:
            scene_types = {
                1: 'urban', 2: 'urban', 3: 'rural', 4: 'urban', 5: 'urban',
                6: 'urban', 7: 'rural', 8: 'urban', 9: 'urban', 10: 'barren'
            }
            
            writer.writerow({
                'image_id': f"{img['id']:02d}",
                'filename': img['name'],
                'expected_flood': img['expected_flood'],
                'expected_depth_cm': img['expected_depth_cm'],
                'scene_type': scene_types.get(img['id'], 'unknown'),
                'difficulty': difficulty_map.get(img['id'], 'unknown')
            })
    
    print(f"\n✅ CSV Manifest created: {csv_path}")
    return csv_path


if __name__ == "__main__":
    print("🎨 Creating 10 Edge Case Test Images...")
    print("=" * 60)
    
    generator = EdgeCaseImageGenerator()
    images = generator.generate_all_images()
    
    print("\n" + "=" * 60)
    print("📊 EDGE CASE TEST IMAGES CREATED:")
    print("=" * 60)
    
    for img in images:
        flood_label = "🌊 FLOOD" if img['expected_flood'] == 1 else "✅ DRY"
        print(f"{img['id']:02d}. {img['name']:30s} | {flood_label} | Depth: {img['expected_depth_cm']:3d}cm")
    
    # Create manifest
    create_metadata_csv(images)
    
    print("\n" + "=" * 60)
    print("✅ ALL IMAGES READY FOR BATCH UPLOAD!")
    print("   Location: test_images/batch_upload/")
    print("   Total: 10 images with edge cases")
    print("=" * 60)
