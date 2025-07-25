import yaml

def create_fixed_yaml():
    """Create a fixed YAML with exact channel specifications."""
    
    # For scale 'l' (width=1.0), we need to ensure all channels match exactly
    # lcnet_075 outputs: P5=1024, P4=96, P3=48
    
    yaml_content = """# Ultralytics YOLOv11 🚀, GPL-3.0 license
# YOLO11-seg instance segmentation model. For Usage examples see https://docs.ultralytics.com/tasks/segment

# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolo11n-seg.yaml' will call yolo11-seg.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

# YOLOv11.0s Lightweight Mamba-assisted FPN with Receptive-field Attention Neck
# Using fixed channel specifications to avoid scaling issues

backbone:
  # [from, repeats, module, args]
  - [-1, 1, lcnet_075, [1024]]       # 0  outputs (P5, P4, P3) = (1024, 96, 48)
  - [0, 1, Index, [0]]               # 1  P5 (1024 channels)
  - [0, 1, Index, [1]]               # 2  P4 (96 channels)  
  - [0, 1, Index, [2]]               # 3  P3 (48 channels)
  - [1, 1, SPPF, [512, 5]]           # 4  P5' (1024 -> 512 channels)
  - [4, 1, C2PSA, [512]]             # 5  refined P5' (512 channels)

head:
  - [5, 1, nn.Upsample, [None, 2, "nearest"]]    # 6  upsample P5' (512 channels)
  - [[6, 2], 1, Concat, [1]]                     # 7  cat(P5', P4) = 512 + 96 = 608 channels
  - [7, 1, C3k2, [256, False]]                   # 8  P4' (608 -> 256 channels)
  
  - [8, 1, nn.Upsample, [None, 2, "nearest"]]    # 9  upsample P4' (256 channels)
  - [[9, 3], 1, Concat, [1]]                     # 10 cat(P4', P3) = 256 + 48 = 304 channels
  - [10, 1, C3k2, [128, False]]                  # 11 P3' (304 -> 128 channels)
  
  - [11, 1, nn.Upsample, [None, 2, "nearest"]]   # 12 upsample P3' (128 channels)
  - [12, 1, C3k2, [64, False]]                   # 13 P2 (128 -> 64 channels)
  - [13, 1, MAFR, [64]]                          # 14 enhanced P2 (64 channels)
  
  - [[14, 11, 8], 1, Detect, [nc]]              # 15 Detect(P2, P3', P4')
"""
    
    # Write the YAML file
    with open('yolov11-lcnet-mafrneck-fixed-v2.yaml', 'w') as f:
        f.write(yaml_content)
    
    print("Created: yolov11-lcnet-mafrneck-fixed-v2.yaml")
    
    # Also create a version with explicit scale
    yaml_dict = yaml.safe_load(yaml_content)
    yaml_dict['scale'] = 'l'  # Force scale to 'l'
    
    with open('yolov11l-lcnet-mafrneck-fixed-v2.yaml', 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
    
    print("Created: yolov11l-lcnet-mafrneck-fixed-v2.yaml")

if __name__ == "__main__":
    create_fixed_yaml()