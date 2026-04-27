# data_loader.py
import numpy as np
import mne
from pathlib import Path
from tqdm import tqdm
from config import Config
import pickle
import warnings
warnings.filterwarnings('ignore')

class EEGDataLoader:
    """EEG数据加载器 - 适配BCI数据集"""
    
    def __init__(self, config=Config):
        self.config = config
        self.config.create_dirs()
        
    def load_single_subject(self, subject_id, runs=None, verbose=False):
        """
        加载单个被试的数据
        
        Parameters:
        -----------
        subject_id : str
            被试ID，如 'S001'
        runs : list, optional
            要加载的run列表，如 ['R03', 'R04']
        verbose : bool
            是否打印详细信息
        """
        if runs is None:
            runs = self.config.TASK_RUNS
            
        subject_files = self.config.get_subject_files(subject_id, runs)
        
        if not subject_files:
            print(f"Warning: No files found for {subject_id}")
            return None, None
        
        if verbose:
            print(f"Loading {subject_id}: {len(subject_files)} files")
        
        all_data = []
        all_labels = []
        
        for file_path in subject_files:
            try:
                # 读取EDF文件
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                
                # 添加被试ID信息
                #raw.info['subject_id'] = subject_id
                
                # 提取事件和标签
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                
                # 只保留运动想象相关的事件
                # 通常：T0=提示，T1=想象开始，T2=结束等
                if events.shape[0] > 0:
                    # 提取epochs
                    epochs = mne.Epochs(
                        raw, events, event_id,
                        tmin=self.config.TMIN,
                        tmax=self.config.TMAX,
                        baseline=None,
                        preload=True,
                        verbose=False
                    )
                    
                    # 获取数据
                    X = epochs.get_data()
                    y = epochs.events[:, -1]
                    
                    all_data.append(X)
                    all_labels.append(y)
                    
                    if verbose:
                        print(f"  {file_path.name}: {X.shape[0]} epochs")
                
            except Exception as e:
                if verbose:
                    print(f"  Error loading {file_path.name}: {e}")
                continue
        
        if not all_data:
            return None, None
        
        # 合并所有run的数据
        X_concat = np.vstack(all_data)
        y_concat = np.hstack(all_labels)
        
        # 重新映射标签到0,1,2,...
        unique_labels = np.unique(y_concat)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y_mapped = np.array([label_map[label] for label in y_concat])
        
        if verbose:
            print(f"  Total: {X_concat.shape[0]} trials, {X_concat.shape[1]} channels, {X_concat.shape[2]} timepoints")
            print(f"  Classes: {unique_labels} -> {list(label_map.keys())}")
        
        return X_concat, y_mapped
    
    def load_multiple_subjects(self, subjects=None, runs=None, max_subjects=None):
        """
        加载多个被试的数据
        
        Parameters:
        -----------
        subjects : list, optional
            被试列表，如 ['S001', 'S002']
        runs : list, optional
            要加载的run列表
        max_subjects : int, optional
            最大加载被试数量
        """
        if subjects is None:
            subjects = self.config.SUBJECTS
        
        if max_subjects:
            subjects = subjects[:max_subjects]
        
        all_X = []
        all_y = []
        subject_info = []
        
        for subject in tqdm(subjects, desc="Loading subjects"):
            X, y = self.load_single_subject(subject, runs, verbose=False)
            
            if X is not None and len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                subject_info.append({
                    'subject': subject,
                    'n_trials': len(X),
                    'classes': np.unique(y)
                })
        
        if not all_X:
            raise ValueError("No data loaded!")
        
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)
        
        # 打印统计信息
        print(f"\nData loading summary:")
        print(f"  Subjects: {len(subject_info)}")
        print(f"  Total trials: {len(y_combined)}")
        print(f"  Data shape: {X_combined.shape}")
        print(f"  Classes: {np.unique(y_combined)}")
        
        return X_combined, y_combined, subject_info
    
    def get_event_description(self, subject_id='S001'):
        """获取事件描述（用于理解标签含义）"""
        files = self.config.get_subject_files(subject_id, runs=['R03'])
        
        if not files:
            print(f"No files found for {subject_id}")
            return
        
        raw = mne.io.read_raw_edf(files[0], preload=True, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        print("Event descriptions:")
        print("=" * 40)
        for name, code in event_id.items():
            print(f"  {name}: {code}")
        
        return event_id
    
    def explore_dataset(self):
        """探索数据集结构"""
        print("Exploring dataset...")
        print("=" * 50)
        print(f"Data root: {self.config.DATA_ROOT}")
        print(f"Exists: {self.config.DATA_ROOT.exists()}")
        
        if not self.config.DATA_ROOT.exists():
            print(f"Error: Data directory not found!")
            return
        
        # 列出所有被试
        subjects = list(self.config.DATA_ROOT.glob("S*"))
        print(f"\nFound {len(subjects)} subjects:")
        for subj in subjects[:5]:
            print(f"  {subj.name}")
        
        # 查看第一个被试的文件
        if subjects:
            first_subject = subjects[0]
            files = list(first_subject.glob("*.edf"))
            print(f"\nFirst subject ({first_subject.name}) files:")
            for f in files[:5]:
                print(f"  {f.name}")
            
            # 加载一个示例文件查看事件
            if files:
                print(f"\nLoading sample file: {files[0].name}")
                raw = mne.io.read_raw_edf(files[0], preload=False, verbose=False)
                print(f"  Channels: {len(raw.ch_names)}")
                print(f"  Duration: {raw.times[-1]:.1f} seconds")
                print(f"  Sampling rate: {raw.info['sfreq']} Hz")

# 测试代码
if __name__ == "__main__":
    # 测试加载器
    loader = EEGDataLoader()
    
    # 探索数据集
    loader.explore_dataset()
    
    # 获取事件描述
    print("\n" + "="*50)
    loader.get_event_description('S001')
    
    # 加载第一个被试的数据
    print("\n" + "="*50)
    X, y = loader.load_single_subject('S001', verbose=True)
    
    if X is not None:
        print(f"\nSuccessfully loaded data!")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Classes: {np.unique(y)}")
    # data_loader.py (在文件末尾添加)

class DataSaver:
    """数据保存工具"""
    
    @staticmethod
    def save_processed_data(data, labels, filename):
        """保存处理后的数据"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({'data': data, 'labels': labels}, f)
        print(f"Data saved to {filename}")
    
    @staticmethod
    def load_processed_data(filename):
        """加载处理后的数据"""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data loaded from {filename}")
        return data['data'], data['labels']
    
    @staticmethod
    def save_features(features, labels, filename):
        """保存特征数据"""
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump({'features': features, 'labels': labels}, f)
        print(f"Features saved to {filename}")
    
    @staticmethod
    def load_features(filename):
        """加载特征数据"""
        import pickle
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Features loaded from {filename}")
        return data['features'], data['labels']