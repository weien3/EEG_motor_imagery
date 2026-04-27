# main.py
import numpy as np
from pathlib import Path
from config import Config
from data_loader import EEGDataLoader, DataSaver
from preprocess import EEGPreprocessor
from feature_extraction import FeatureExtractor
from train import ModelTrainer
from evaluate import Evaluator
import logging
import argparse
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(args):
    # 创建配置及目录
    Config.create_dirs()
    
    # 1. 加载数据
    logger.info("Step 1: Loading data...")
    loader = EEGDataLoader()
    
    # 探索数据集
    loader.explore_dataset()
    
    # 获取事件描述（了解标签含义）
    logger.info("Getting event descriptions...")
    event_id = loader.get_event_description('S001')
    logger.info(f"Event IDs: {event_id}")
    
    # 加载多个被试的数据
    logger.info("Loading subjects data...")
    X, y, subject_info = loader.load_multiple_subjects(
        subjects=None,  # 使用配置中的所有被试
        max_subjects=args.max_subjects if hasattr(args, 'max_subjects') else 3 
    )
    
    if X is None or len(X) == 0:
        logger.error("No data loaded!")
        return None, None
    
    logger.info(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    logger.info(f"Classes: {np.unique(y)}")
    
    # 保存原始数据（可选）
    if args.save_data:
        data_path = Path(Config.PROCESSED_DATA_PATH) / "raw_data.pkl"
        DataSaver.save_processed_data(X, y, str(data_path))
    
    # 2. 预处理
    logger.info("Step 2: Preprocessing...")
    preprocessor = EEGPreprocessor()
    
    # 数据归一化（在特征提取前）
    # X shape: (n_trials, n_channels, n_timepoints)
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        X_normalized[i] = preprocessor.normalize_data(X[i], method='zscore')
    
    logger.info(f"Normalized data shape: {X_normalized.shape}")
    
    # 3. 特征提取
    logger.info("Step 3: Feature extraction...")
    feature_extractor = FeatureExtractor()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, 
        test_size=Config.TEST_SIZE, 
        stratify=y, 
        random_state=Config.RANDOM_STATE
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 提取特征
    logger.info("Extracting features from training set...")
    X_train_feat = feature_extractor.combine_features(
        X_train, y_train, 
        use_csp=True, 
        use_band_power=True,
        use_stats=True
    )
    
    logger.info("Extracting features from test set...")
    X_test_feat = feature_extractor.combine_features(
        X_test, y_test,
        use_csp=True,
        use_band_power=True,
        use_stats=True
    )
    
    logger.info(f"Features shape: Train {X_train_feat.shape}, Test {X_test_feat.shape}")
    
    # 可选：PCA降维
    if X_train_feat.shape[1] > 50:
        logger.info(f"Applying PCA (dimensions: {X_train_feat.shape[1]} -> 30)...")
        X_train_feat = feature_extractor.apply_pca(X_train_feat, n_components=30, fit=True)
        X_test_feat = feature_extractor.apply_pca(X_test_feat, n_components=30, fit=False)
        logger.info(f"After PCA: {X_train_feat.shape[1]} dimensions")
    
    # 保存特征（可选）
    if args.save_features:
        feature_path = Path(Config.PROCESSED_DATA_PATH) / "features.pkl"
        DataSaver.save_features(X_train_feat, y_train, str(feature_path))
    
    # 4. 训练模型
    logger.info("Step 4: Training model...")
    trainer = ModelTrainer()
    
    model = None
    if Config.MODEL_TYPE == 'svm':
        logger.info("Training SVM classifier...")
        model = trainer.train_svm(X_train_feat, y_train)
    elif Config.MODEL_TYPE == 'rf':
        logger.info("Training Random Forest classifier...")
        model = trainer.train_random_forest(X_train_feat, y_train)
    elif Config.MODEL_TYPE == 'cnn':
        logger.info("Training CNN model...")
        # CNN需要原始数据，不是特征
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
        X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
        
        # 进一步划分验证集
        X_train_cnn, X_val, y_train_cnn, y_val = train_test_split(
            X_train_cnn, y_train, 
            test_size=Config.VAL_SIZE, 
            stratify=y_train, 
            random_state=Config.RANDOM_STATE
        )
        logger.info(f"CNN train: {X_train_cnn.shape}, val: {X_val.shape}")
        
        model = trainer.train_cnn(X_train_cnn, y_train_cnn, X_val, y_val)
    else:
        raise ValueError(f"Unknown model type: {Config.MODEL_TYPE}")
    
    # 保存模型
    if model is not None:
        model_path = Path(Config.MODEL_SAVE_PATH) / f"{Config.MODEL_TYPE}_model.pkl"
        trainer.model = model  # 确保trainer有model属性
        trainer.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
    
    # 5. 评估
    logger.info("Step 5: Evaluating model...")
    evaluator = Evaluator()
    
    # 评估模型
    test_results = evaluator.evaluate(model, X_test_feat, y_test)
    
    logger.info(f"Test Results: Accuracy = {test_results.get('accuracy', 'N/A'):.4f}")
    if 'classification_report' in test_results:
        logger.info(f"\n{test_results['classification_report']}")
    
    
    # 绘制混淆矩阵
    if hasattr(evaluator, 'plot_confusion_matrix'):
        cm_path = Path(Config.RESULT_PATH) / "confusion_matrix.png"
        # 检查 plot_confusion_matrix 期望的参数
        try:
            # 尝试不同参数格式
            if 'predictions' in test_results:
                evaluator.plot_confusion_matrix(
                    y_test, 
                    test_results['predictions'],
                    save_path=str(cm_path)
                )
            else:
                # 如果没有预测结果，只传真实标签
                evaluator.plot_confusion_matrix(
                    y_true=y_test,
                    save_path=str(cm_path)
                )
            logger.info(f"Confusion matrix saved to {cm_path}")
        except TypeError as e:
            logger.warning(f"Could not plot confusion matrix: {e}")
    
    # 6. 可选：交叉验证
    if args.cross_validate:
        logger.info("Performing cross-validation...")
        from sklearn.model_selection import cross_val_score
        
        if Config.MODEL_TYPE == 'svm' or Config.MODEL_TYPE == 'rf':
            cv_scores = cross_val_score(model, X_train_feat, y_train, cv=5)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        else:
            logger.warning("Cross-validation not implemented for CNN yet")
    
    # 7. 保存结果
    if args.save_results:
        import json
        results_path = Path(Config.RESULT_PATH) / "results.json"
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_serializable = {k: convert_to_serializable(v) for k, v in test_results.items()}
        
        with open(results_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        logger.info(f"Results saved to {results_path}")
    
    logger.info("Pipeline completed successfully!")
    
    return model, test_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Motor Imagery Pipeline")
    parser.add_argument("--cross_validate", action="store_true", 
                       help="Perform cross-validation")
    parser.add_argument("--save_data", action="store_true",
                       help="Save processed data")
    parser.add_argument("--save_features", action="store_true",
                       help="Save extracted features")
    parser.add_argument("--save_results", action="store_true",
                       help="Save evaluation results")
    parser.add_argument("--max_subjects", type=int, default=3,
                       help="Maximum number of subjects to load (for testing)")
    
    args = parser.parse_args()
    
    model, results = main(args)