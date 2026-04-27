# debug_data_loading.py
import sys
from pathlib import Path
from config import Config
from data_loader import EEGDataLoader

def debug_data_loading():
    print("=" * 60)
    print("Debugging Data Loading")
    print("=" * 60)
    
    # 1. 检查配置
    print(f"\n1. Configuration:")
    print(f"   DATA_ROOT: {Config.DATA_ROOT}")
    print(f"   DATA_ROOT exists: {Config.DATA_ROOT.exists()}")
    
    if not Config.DATA_ROOT.exists():
        print(f"   ERROR: Data directory not found!")
        print(f"   Please check your data path in config.py")
        return
    
    # 2. 列出所有被试
    subjects = list(Config.DATA_ROOT.glob("S*"))
    print(f"\n2. Found {len(subjects)} subjects:")
    for subj in subjects[:10]:
        print(f"   {subj.name}")
    
    if len(subjects) == 0:
        print(f"   No subjects found! Check if data files are in correct format.")
        print(f"   Expected pattern: S* (e.g., S001, S002)")
        return
    
    # 3. 检查第一个被试的文件
    first_subject = subjects[0]
    print(f"\n3. Checking first subject: {first_subject.name}")
    
    edf_files = list(first_subject.glob("*.edf"))
    print(f"   EDF files found: {len(edf_files)}")
    for f in edf_files[:5]:
        print(f"     - {f.name}")
    
    if len(edf_files) == 0:
        print(f"   No EDF files found! Check file format.")
        # 检查其他格式
        other_files = list(first_subject.glob("*"))
        print(f"   Other files in directory: {[f.name for f in other_files[:10]]}")
        return
    
    # 4. 测试加载单个文件
    print(f"\n4. Testing load_single_subject...")
    loader = EEGDataLoader()
    
    # 测试第一个被试
    X, y = loader.load_single_subject(first_subject.name, verbose=True)
    
    if X is not None:
        print(f"\n   ✓ Successfully loaded {first_subject.name}!")
        print(f"   Data shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Unique labels: {np.unique(y)}")
    else:
        print(f"\n   ✗ Failed to load {first_subject.name}")
        print(f"   Trying to check file content...")
        
        # 尝试直接读取一个文件查看事件
        test_file = edf_files[0]
        print(f"   Reading file: {test_file.name}")
        try:
            raw = mne.io.read_raw_edf(test_file, preload=False, verbose=False)
            print(f"     Channels: {len(raw.ch_names)}")
            print(f"     Duration: {raw.times[-1]:.2f} seconds")
            print(f"     Sampling rate: {raw.info['sfreq']} Hz")
            
            # 尝试读取事件
            raw_preload = mne.io.read_raw_edf(test_file, preload=True, verbose=False)
            events, event_id = mne.events_from_annotations(raw_preload, verbose=False)
            print(f"     Events found: {len(events)}")
            print(f"     Event IDs: {event_id}")
            
            if len(events) == 0:
                print(f"     WARNING: No events found in this file!")
                print(f"     Annotations: {raw_preload.annotations}")
                
        except Exception as e:
            print(f"     Error reading file: {e}")
    
    # 5. 尝试加载多个被试
    print(f"\n5. Testing load_multiple_subjects with max_subjects=1...")
    try:
        X_multi, y_multi, subject_info = loader.load_multiple_subjects(
            subjects=[first_subject.name],
            max_subjects=1
        )
        if X_multi is not None:
            print(f"   ✓ Success! Loaded {len(subject_info)} subject(s)")
            print(f"   Combined data shape: {X_multi.shape}")
        else:
            print(f"   ✗ Failed to load multiple subjects")
    except Exception as e:
        print(f"   ✗ Error: {e}")

if __name__ == "__main__":
    import numpy as np
    import mne
    debug_data_loading()