"""
FC3D智能交互式预测系统 - 增强优化版
Author: AI Assistant
Date: 2024
Description: 基于先进AI架构的福彩3D预测系统，包含完整的超参数优化模块和增强训练策略
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import pickle
import os
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any, Optional, Union
import sys
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
import json
import hashlib

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)


# 设置日志
def setup_logging():
    """设置日志记录"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('fc3d_predictor_enhanced.log', encoding='utf-8')
        ]
    )


setup_logging()


# 常量定义
class EnhancedConstants:
    """增强版常量定义"""
    DEFAULT_SEQUENCE_LENGTH = 60
    BATCH_SIZE = 64
    MAX_EPOCHS = 200
    PATIENCE = 20
    LEARNING_RATE = 2e-4
    VALIDATION_SPLIT = 0.15
    MIN_SEQUENCE_LENGTH = 20
    OPTIMIZATION_TRIALS = 50
    OPTIMIZATION_TIMEOUT = 3600
    MIN_DATA_POINTS = 200
    WARMUP_EPOCHS = 10
    GRADIENT_ACCUMULATION_STEPS = 4
    LABEL_SMOOTHING = 0.1
    TOP_CANDIDATES = 6  # 修改为6个候选数字


# 自定义异常
class ModelValidationError(Exception):
    """模型验证错误"""
    pass


class DataValidationError(Exception):
    """数据验证错误"""
    pass


class OptimizationError(Exception):
    """超参数优化错误"""
    pass


class ConfigurationError(Exception):
    """配置错误"""
    pass


class ProgressBar:
    """自定义进度条类"""

    def __init__(self, total: int, desc: str = "Processing"):
        self.total = total
        self.desc = desc
        self.start_time = None
        self.current = 0

    def __enter__(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total, desc=self.desc,
                         bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}')
        return self

    def update(self, n: int = 1, **kwargs):
        """更新进度条"""
        self.current += n
        elapsed = time.time() - self.start_time
        if self.current > 0:
            avg_time_per_step = elapsed / self.current
            remaining_time = avg_time_per_step * (self.total - self.current)
            remaining_str = str(timedelta(seconds=int(remaining_time)))
        else:
            remaining_str = "Calculating..."

        self.pbar.set_postfix({
            **kwargs,
            'remaining': remaining_str
        })
        self.pbar.update(n)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pbar.close()


class DataQualityValidator:
    """数据质量验证器"""

    @staticmethod
    def validate_data_structure(data: pd.DataFrame) -> Dict[str, Any]:
        """验证数据结构"""
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }

        # 检查必需列
        required_columns = ['期号', '百位', '十位', '个位', '和值', '跨度', '重复数字']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"缺少必需列: {missing_columns}")

        # 检查数据类型
        if '百位' in data.columns:
            if not pd.api.types.is_numeric_dtype(data['百位']):
                validation_result['errors'].append("百位列必须为数值类型")

        # 检查数值范围
        for col in ['百位', '十位', '个位']:
            if col in data.columns:
                if data[col].min() < 0 or data[col].max() > 9:
                    validation_result['errors'].append(f"{col}数值超出范围(0-9)")

        # 检查缺失值
        missing_values = data[required_columns].isnull().sum()
        if missing_values.any():
            validation_result['warnings'].append(f"存在缺失值: {missing_values.to_dict()}")

        # 检查重复数据
        duplicates = data.duplicated().sum()
        if duplicates > 0:
            validation_result['warnings'].append(f"存在{duplicates}条重复数据")

        # 生成摘要
        validation_result['summary'] = {
            'total_records': len(data),
            'date_range': {
                'start': data['期号'].min() if '期号' in data.columns else 'N/A',
                'end': data['期号'].max() if '期号' in data.columns else 'N/A'
            },
            'value_ranges': {
                col: (data[col].min(), data[col].max())
                for col in ['百位', '十位', '个位']
                if col in data.columns
            }
        }

        return validation_result

    @staticmethod
    def generate_data_report(data: pd.DataFrame) -> str:
        """生成数据质量报告"""
        validation = DataQualityValidator.validate_data_structure(data)

        report_lines = ["=" * 60, "数据质量验证报告", "=" * 60]

        if validation['is_valid']:
            report_lines.append("✅ 数据结构验证通过")
        else:
            report_lines.append("❌ 数据结构验证失败:")
            for error in validation['errors']:
                report_lines.append(f"   - {error}")

        if validation['warnings']:
            report_lines.append("\n⚠️ 警告信息:")
            for warning in validation['warnings']:
                report_lines.append(f"   - {warning}")

        # 添加统计信息
        report_lines.append("\n📊 数据统计:")
        for key, value in validation['summary'].items():
            if isinstance(value, dict):
                report_lines.append(f"   {key}:")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"     {sub_key}: {sub_value}")
            else:
                report_lines.append(f"   {key}: {value}")

        report_lines.append("=" * 60)
        return "\n".join(report_lines)


class ConfigManager:
    """配置管理器"""

    CONFIG_FILE = "fc3d_system_config_enhanced.json"

    @classmethod
    def save_config(cls, predictor, filepath: str = None) -> bool:
        """保存系统配置"""
        try:
            if filepath is None:
                filepath = cls.CONFIG_FILE

            # 准备可序列化的优化结果
            serializable_optimization_results = {}
            for model_type, results in predictor.optimization_results.items():
                serializable_optimization_results[model_type] = {
                    'best_params': results.get('best_params', {}),
                    'best_score': results.get('best_score', 0)
                }

            config = {
                'system_info': {
                    'version': '2.0.0',
                    'save_time': datetime.now().isoformat(),
                    'data_hash': cls._calculate_data_hash(predictor.data) if predictor.data is not None else None
                },
                'feature_columns': predictor.feature_columns,
                'current_period': predictor.current_period,
                'model_status': {
                    name: {
                        'is_trained': info.get('is_trained', False),
                        'has_optimized_params': info.get('optimized_params') is not None,
                        'best_val_acc': info.get('info', {}).get('best_val_acc', 0)
                    }
                    for name, info in predictor.models.items()
                },
                'optimization_results': serializable_optimization_results,
                'data_summary': {
                    'total_records': len(predictor.data) if predictor.data is not None else 0,
                    'feature_count': len(predictor.feature_columns) if predictor.feature_columns else 0
                }
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)

            logging.info(f"系统配置已保存: {filepath}")
            return True

        except Exception as e:
            logging.error(f"保存配置失败: {e}")
            return False

    @classmethod
    def load_config(cls, predictor, filepath: str = None) -> bool:
        """加载系统配置"""
        try:
            if filepath is None:
                filepath = cls.CONFIG_FILE

            if not os.path.exists(filepath):
                logging.warning(f"配置文件不存在: {filepath}")
                return False

            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 验证配置完整性
            if not cls._validate_config(config):
                raise ConfigurationError("配置文件格式无效")

            # 应用配置
            predictor.feature_columns = config.get('feature_columns', [])
            predictor.current_period = config.get('current_period')
            predictor.optimization_results = config.get('optimization_results', {})

            # 更新模型状态
            model_status = config.get('model_status', {})
            for model_type, status in model_status.items():
                if model_type in predictor.models:
                    predictor.models[model_type]['is_trained'] = status.get('is_trained', False)

            logging.info(f"系统配置已加载: {filepath}")
            return True

        except Exception as e:
            logging.error(f"加载配置失败: {e}")
            return False

    @staticmethod
    def _calculate_data_hash(data: pd.DataFrame) -> str:
        """计算数据哈希值"""
        if data is None:
            return ""
        return hashlib.md5(pd.util.hash_pandas_object(data).values.tobytes()).hexdigest()

    @staticmethod
    def _validate_config(config: Dict) -> bool:
        """验证配置格式"""
        required_sections = ['system_info', 'model_status']
        return all(section in config for section in required_sections)


class PerformanceTracker:
    """模型性能跟踪器"""

    def __init__(self):
        self.performance_history = {}

    def track_prediction(self, model_type: str, predicted: List, actual: List, period: str):
        """跟踪预测性能"""
        if model_type not in self.performance_history:
            self.performance_history[model_type] = []

        # 计算准确率
        bai_correct = predicted[0] == actual[0]
        shi_correct = predicted[1] == actual[1]
        ge_correct = predicted[2] == actual[2]
        all_correct = bai_correct and shi_correct and ge_correct

        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'period': period,
            'predicted': predicted,
            'actual': actual,
            'accuracy': {
                'bai': bai_correct,
                'shi': shi_correct,
                'ge': ge_correct,
                'all': all_correct
            }
        }

        self.performance_history[model_type].append(performance_data)

        # 限制历史记录长度
        if len(self.performance_history[model_type]) > 1000:
            self.performance_history[model_type] = self.performance_history[model_type][-1000:]

    def get_performance_summary(self, model_type: str) -> Dict[str, Any]:
        """获取性能摘要"""
        if model_type not in self.performance_history or not self.performance_history[model_type]:
            return {}

        history = self.performance_history[model_type]
        total_predictions = len(history)

        accuracies = {
            'bai': sum(1 for h in history if h['accuracy']['bai']) / total_predictions,
            'shi': sum(1 for h in history if h['accuracy']['shi']) / total_predictions,
            'ge': sum(1 for h in history if h['accuracy']['ge']) / total_predictions,
            'all': sum(1 for h in history if h['accuracy']['all']) / total_predictions
        }

        return {
            'total_predictions': total_predictions,
            'accuracy_rates': accuracies,
            'recent_performance': history[-10:] if len(history) >= 10 else history
        }

    def save_performance_data(self, filepath: str = "performance_history_enhanced.json"):
        """保存性能数据"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
            logging.info(f"性能数据已保存: {filepath}")
        except Exception as e:
            logging.error(f"保存性能数据失败: {e}")

    def load_performance_data(self, filepath: str = "performance_history_enhanced.json"):
        """加载性能数据"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    self.performance_history = json.load(f)
                logging.info(f"性能数据已加载: {filepath}")
        except Exception as e:
            logging.error(f"加载性能数据失败: {e}")


class EnhancedFC3DDataset(Dataset):
    """增强版福彩3D数据集类"""

    def __init__(self, data: pd.DataFrame, sequence_length: int = EnhancedConstants.DEFAULT_SEQUENCE_LENGTH,
                 feature_columns: Optional[List[str]] = None,
                 fit_scaler: bool = True,
                 external_scaler: Optional[StandardScaler] = None,
                 for_prediction: bool = False):
        self.data = data.reset_index(drop=True)
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns
        self.scaler = external_scaler
        self.for_prediction = for_prediction
        self._prepare_features(fit_scaler)

    def _prepare_features(self, fit_scaler: bool = True):
        """增强特征工程"""
        df = self.data.copy()

        # 基础特征
        df['period'] = df.index

        # 更丰富的技术指标
        for window in [3, 5, 10, 15, 20]:
            # 移动平均
            for col in ['百位', '十位', '个位']:
                df[f'{col}_ma_{window}'] = df[col].rolling(window, min_periods=1).mean()
                df[f'{col}_std_{window}'] = df[col].rolling(window, min_periods=1).std()

            # 和值与跨度的技术指标
            df[f'和值_ma_{window}'] = df['和值'].rolling(window, min_periods=1).mean()
            df[f'跨度_ma_{window}'] = df['跨度'].rolling(window, min_periods=1).mean()

        # 增强滞后特征
        for lag in [1, 2, 3, 5, 7, 10, 15]:
            for col in ['百位', '十位', '个位']:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        # 周期特征
        df['day_of_week'] = df.index % 7  # 假设每天一期
        df['period_in_month'] = df.index % 30

        # 统计特征
        for col in ['百位', '十位', '个位']:
            df[f'{col}_rolling_skew_10'] = df[col].rolling(10, min_periods=1).skew()
            df[f'{col}_rolling_kurt_10'] = df[col].rolling(10, min_periods=1).kurt()

        # 组合特征
        df['百十组合'] = df['百位'] * 10 + df['十位']
        df['十个组合'] = df['十位'] * 10 + df['个位']
        df['百个组合'] = df['百位'] * 10 + df['个位']

        # 热编码重复数字
        df['重复数_0'] = (df['重复数字'] == 0).astype(int)
        df['重复数_1'] = (df['重复数字'] == 1).astype(int)
        df['重复数_2'] = (df['重复数字'] == 2).astype(int)

        # 趋势特征
        for col in ['百位', '十位', '个位']:
            df[f'{col}_trend_5'] = df[col].rolling(5, min_periods=1).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
            )

        # 波动率特征
        for col in ['百位', '十位', '个位']:
            df[f'{col}_volatility_10'] = df[col].rolling(10, min_periods=1).std() / df[col].rolling(10,
                                                                                                    min_periods=1).mean()

        # 填充NaN值 - 使用更智能的填充
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 确定特征列
        if self.feature_columns is None:
            exclude_columns = ['期号', '百位', '十位', '个位', '重复数字']
            self.feature_columns = [col for col in df.columns if col not in exclude_columns]

        # 使用RobustScaler代替StandardScaler
        if fit_scaler and self.scaler is None:
            self.scaler = RobustScaler()
            feature_data = df[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(feature_data)
            self.features = df.copy()
            self.features[self.feature_columns] = scaled_features
        elif self.scaler is not None:
            feature_data = df[self.feature_columns].values
            scaled_features = self.scaler.transform(feature_data)
            self.features = df.copy()
            self.features[self.feature_columns] = scaled_features
        else:
            self.features = df

    def get_feature_dimension(self) -> int:
        """获取特征维度"""
        return len(self.feature_columns)

    def get_feature_columns(self) -> List[str]:
        """获取特征列名"""
        return self.feature_columns.copy()

    def get_scaler(self) -> Optional[StandardScaler]:
        """获取标准化器"""
        return self.scaler

    def __len__(self):
        if self.for_prediction:
            return 1 if len(self.data) >= self.sequence_length else 0
        else:
            return max(0, len(self.data) - self.sequence_length)

    def __getitem__(self, idx):
        if self.for_prediction:
            if len(self.features) < self.sequence_length:
                raise DataValidationError("数据不足进行预测")

            start_idx = len(self.features) - self.sequence_length
            sequence_data = self.features.iloc[start_idx:start_idx + self.sequence_length]

            features = sequence_data[self.feature_columns].values.astype(np.float32)

            return (
                torch.FloatTensor(features),
                torch.LongTensor([0]),
                torch.LongTensor([0]),
                torch.LongTensor([0])
            )
        else:
            if idx + self.sequence_length >= len(self.features):
                idx = len(self.features) - self.sequence_length - 1

            if idx < 0:
                idx = 0

            sequence_data = self.features.iloc[idx:idx + self.sequence_length]
            features = sequence_data[self.feature_columns].values.astype(np.float32)

            target_period = idx + self.sequence_length
            if target_period >= len(self.data):
                target_period = len(self.data) - 1

            target_bai = self.data.iloc[target_period]['百位']
            target_shi = self.data.iloc[target_period]['十位']
            target_ge = self.data.iloc[target_period]['个位']

            return (
                torch.FloatTensor(features),
                torch.LongTensor([target_bai]),
                torch.LongTensor([target_shi]),
                torch.LongTensor([target_ge])
            )


class BaseModel(ABC, nn.Module):
    """基础模型抽象类"""

    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    @abstractmethod
    def forward(self, x):
        pass

    def predict_top6(self, x):
        """预测每个位置的前6个最可能数字"""
        with torch.no_grad():
            outputs = self.forward(x)
            if isinstance(outputs, tuple):
                bai_probs, shi_probs, ge_probs = outputs
            else:
                bai_probs, shi_probs, ge_probs = outputs.chunk(3, dim=1)

            bai_top6 = torch.topk(bai_probs, EnhancedConstants.TOP_CANDIDATES, dim=1)[1].cpu().numpy()[0]
            shi_top6 = torch.topk(shi_probs, EnhancedConstants.TOP_CANDIDATES, dim=1)[1].cpu().numpy()[0]
            ge_top6 = torch.topk(ge_probs, EnhancedConstants.TOP_CANDIDATES, dim=1)[1].cpu().numpy()[0]

            return bai_top6, shi_top6, ge_top6


class EnhancedTemporalMoE(BaseModel):
    """增强版时序混合专家模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_experts: int = 8,
                 dropout_rate: float = 0.2, expert_dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim)

        # 增强的专家网络
        expert_output_dim = hidden_dim // 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(hidden_dim, expert_output_dim),
                nn.BatchNorm1d(expert_output_dim),
                nn.GELU(),
                nn.Dropout(expert_dropout // 2)
            ) for _ in range(num_experts)
        ])

        # 增强的门控网络
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

        # 残差连接
        self.residual_linear = nn.Linear(input_dim, expert_output_dim)

        # 时序处理层
        self.temporal_processor = nn.LSTM(
            input_size=expert_output_dim,
            hidden_size=expert_output_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        # LSTM输出维度调整（双向）
        lstm_output_dim = expert_output_dim * 2

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # 层归一化
        self.layer_norm1 = nn.LayerNorm(lstm_output_dim)
        self.layer_norm2 = nn.LayerNorm(lstm_output_dim)

        # 增强的输出头
        output_head_dim = hidden_dim
        self.bai_head = self._create_enhanced_output_head(lstm_output_dim * 2)  # 多尺度池化后维度翻倍
        self.shi_head = self._create_enhanced_output_head(lstm_output_dim * 2)
        self.ge_head = self._create_enhanced_output_head(lstm_output_dim * 2)

        # 初始化权重
        self._initialize_weights()

    def _create_enhanced_output_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(input_dim // 4, 10)
        )

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 门控机制 - 使用序列平均
        gate_input = x.mean(dim=1)
        gate_weights = self.gate(gate_input)

        # 专家处理
        expert_outputs = []
        x_flat = x.reshape(-1, x.size(-1))

        for i, expert in enumerate(self.experts):
            expert_out = expert(x_flat)
            expert_out = expert_out.reshape(batch_size, seq_len, -1)
            expert_outputs.append(expert_out.unsqueeze(-1))

        # 加权组合
        expert_outputs = torch.cat(expert_outputs, dim=-1)
        weighted_experts = torch.einsum('bsde,be->bsd', expert_outputs, gate_weights)

        # 残差连接
        residual = self.residual_linear(x)
        weighted_experts = weighted_experts + residual

        # 时序处理
        temporal_out, _ = self.temporal_processor(weighted_experts)
        temporal_out = self.layer_norm1(temporal_out)

        # 注意力机制
        attn_out, _ = self.attention(temporal_out, temporal_out, temporal_out)
        combined = self.layer_norm2(temporal_out + attn_out)

        # 多尺度池化
        avg_pool = combined.mean(dim=1)
        max_pool = combined.max(dim=1)[0]
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # 输出预测
        bai_logits = self.bai_head(pooled)
        shi_logits = self.shi_head(pooled)
        ge_logits = self.ge_head(pooled)

        return (
            torch.softmax(bai_logits, dim=-1),
            torch.softmax(shi_logits, dim=-1),
            torch.softmax(ge_logits, dim=-1)
        )


class EnhancedAttentionLSTM(BaseModel):
    """增强版注意力LSTM模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 3,
                 dropout_rate: float = 0.2, lstm_dropout: float = 0.1):
        super().__init__(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        # 增强输出头
        self.bai_head = self._create_enhanced_output_head(lstm_output_dim)
        self.shi_head = self._create_enhanced_output_head(lstm_output_dim)
        self.ge_head = self._create_enhanced_output_head(lstm_output_dim)

        self._initialize_weights()

    def _create_enhanced_output_head(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.BatchNorm1d(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(input_dim // 4, 10)
        )

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(x)

        # 自注意力
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # 残差连接 + 层归一化
        normalized_out = self.layer_norm(lstm_out + attn_out)

        # 多尺度池化
        avg_pool = normalized_out.mean(dim=1)
        max_pool = normalized_out.max(dim=1)[0]
        final_out = torch.cat([avg_pool, max_pool], dim=1)

        # 输出预测
        bai_logits = self.bai_head(final_out)
        shi_logits = self.shi_head(final_out)
        ge_logits = self.ge_head(final_out)

        return (
            torch.softmax(bai_logits, dim=-1),
            torch.softmax(shi_logits, dim=-1),
            torch.softmax(ge_logits, dim=-1)
        )


class EnhancedTransformer(BaseModel):
    """增强版Transformer模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 4,
                 num_heads: int = 8, dropout_rate: float = 0.1, attention_dropout: float = 0.05):
        super().__init__(input_dim, hidden_dim)

        # 确保hidden_dim能被num_heads整除
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"

        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=1000)

        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # 增强的概率校准层
        self.calibration = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05)
        )

        # 输出头
        self.bai_head = nn.Linear(hidden_dim // 2, 10)
        self.shi_head = nn.Linear(hidden_dim // 2, 10)
        self.ge_head = nn.Linear(hidden_dim // 2, 10)

        # 温度参数用于概率校准
        self.temperature = nn.Parameter(torch.ones(1))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.TransformerEncoderLayer):
                for name, param in module.named_parameters():
                    if 'weight' in name and 'norm' not in name:
                        if 'linear' in name:
                            nn.init.xavier_uniform_(param)
                        else:
                            nn.init.xavier_uniform_(param)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 输入投影
        x_proj = self.input_projection(x)

        # 添加位置编码
        x_pos = self.pos_encoding(x_proj)

        # Transformer处理
        transformer_out = self.transformer(x_pos)

        # 多尺度池化
        avg_pool = transformer_out.mean(dim=1)
        max_pool = transformer_out.max(dim=1)[0]
        pooled = torch.cat([avg_pool, max_pool], dim=1)

        # 概率校准
        calibrated = self.calibration(pooled)

        # 温度缩放
        calibrated = calibrated / self.temperature

        # 输出预测
        bai_logits = self.bai_head(calibrated)
        shi_logits = self.shi_head(calibrated)
        ge_logits = self.ge_head(calibrated)

        return (
            torch.softmax(bai_logits, dim=-1),
            torch.softmax(shi_logits, dim=-1),
            torch.softmax(ge_logits, dim=-1)
        )


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EnhancedModelTrainer:
    """增强版模型训练器"""

    def __init__(self, model: BaseModel, model_name: str, learning_rate: float = EnhancedConstants.LEARNING_RATE):
        self.model = model
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 使用更先进的优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # 使用余弦退火学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=EnhancedConstants.WARMUP_EPOCHS,
            T_mult=2,
            eta_min=1e-6
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        # 训练状态
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        # 梯度累积
        self.gradient_accumulation_steps = EnhancedConstants.GRADIENT_ACCUMULATION_STEPS

        logging.info(f"Enhanced model trainer initialized for {model_name} on device: {self.device}")

    def cross_entropy_with_label_smoothing(self, pred, target, epsilon=EnhancedConstants.LABEL_SMOOTHING):
        """标签平滑的交叉熵损失"""
        n_classes = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - epsilon) + epsilon / n_classes
        log_prb = torch.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

    def calculate_accuracy(self, probs, targets):
        """计算准确率"""
        _, predicted = torch.max(probs, 1)
        correct = (predicted == targets).float().sum()
        return correct.item() / targets.size(0)

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """增强的训练epoch"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_batches = len(train_loader)

        if total_batches == 0:
            raise DataValidationError("训练数据为空")

        self.optimizer.zero_grad()

        with ProgressBar(total_batches, desc=f"Epoch {epoch} Training") as pbar:
            for batch_idx, (data, bai_target, shi_target, ge_target) in enumerate(train_loader):
                data = data.to(self.device)
                bai_target = bai_target.to(self.device).squeeze()
                shi_target = shi_target.to(self.device).squeeze()
                ge_target = ge_target.to(self.device).squeeze()

                bai_probs, shi_probs, ge_probs = self.model(data)

                # 使用标签平滑的损失函数
                bai_loss = self.cross_entropy_with_label_smoothing(bai_probs, bai_target)
                shi_loss = self.cross_entropy_with_label_smoothing(shi_probs, shi_target)
                ge_loss = self.cross_entropy_with_label_smoothing(ge_probs, ge_target)

                loss = (bai_loss + shi_loss + ge_loss) / 3
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step(epoch + batch_idx / total_batches)

                total_loss += loss.item() * self.gradient_accumulation_steps

                # 计算准确率
                bai_acc = self.calculate_accuracy(bai_probs, bai_target)
                shi_acc = self.calculate_accuracy(shi_probs, shi_target)
                ge_acc = self.calculate_accuracy(ge_probs, ge_target)
                batch_acc = (bai_acc + shi_acc + ge_acc) / 3
                total_acc += batch_acc

                pbar.update(1, loss=loss.item(), accuracy=f"{batch_acc:.4f}")

        # 处理剩余的梯度
        if total_batches % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_acc)
        return avg_loss, avg_acc

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_batches = len(val_loader)

        if total_batches == 0:
            raise DataValidationError("验证数据为空")

        with torch.no_grad():
            with ProgressBar(total_batches, desc="Validating") as pbar:
                for data, bai_target, shi_target, ge_target in val_loader:
                    data = data.to(self.device)
                    bai_target = bai_target.to(self.device).squeeze()
                    shi_target = shi_target.to(self.device).squeeze()
                    ge_target = ge_target.to(self.device).squeeze()

                    bai_probs, shi_probs, ge_probs = self.model(data)

                    bai_loss = self.cross_entropy_with_label_smoothing(bai_probs, bai_target)
                    shi_loss = self.cross_entropy_with_label_smoothing(shi_probs, shi_target)
                    ge_loss = self.cross_entropy_with_label_smoothing(ge_probs, ge_target)

                    loss = (bai_loss + shi_loss + ge_loss) / 3
                    total_loss += loss.item()

                    # 计算准确率
                    bai_acc = self.calculate_accuracy(bai_probs, bai_target)
                    shi_acc = self.calculate_accuracy(shi_probs, shi_target)
                    ge_acc = self.calculate_accuracy(ge_probs, ge_target)
                    batch_acc = (bai_acc + shi_acc + ge_acc) / 3
                    total_acc += batch_acc

                    pbar.update(1, loss=loss.item(), accuracy=f"{batch_acc:.4f}")

        avg_loss = total_loss / total_batches
        avg_acc = total_acc / total_batches
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_acc)
        return avg_loss, avg_acc

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = EnhancedConstants.MAX_EPOCHS,
              patience: int = EnhancedConstants.PATIENCE) -> Dict[str, Any]:
        """完整训练过程"""
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        best_model_state = None

        logging.info(f"开始训练 {self.model_name}...")
        print(f"🚀 开始训练 {self.model_name}...")
        print(f"📊 训练样本: {len(train_loader.dataset)}, 验证样本: {len(val_loader.dataset)}")
        print(f"⚙️  使用增强训练策略: 标签平滑={EnhancedConstants.LABEL_SMOOTHING}, "
              f"梯度累积={self.gradient_accumulation_steps}")

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            try:
                train_loss, train_acc = self.train_epoch(train_loader, epoch)
                val_loss, val_acc = self.validate(val_loader)

                current_lr = self.optimizer.param_groups[0]['lr']

                print(f'Epoch: {epoch}/{epochs}\t'
                      f'训练损失: {train_loss:.6f}\t训练准确率: {train_acc:.4f}\t'
                      f'验证损失: {val_loss:.6f}\t验证准确率: {val_acc:.4f}\t'
                      f'学习率: {current_lr:.2e}')

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    print(f'🎯 发现更好的模型! 验证损失: {val_loss:.6f}, 验证准确率: {val_acc:.4f}')
                else:
                    patience_counter += 1
                    print(f'⏳ 早停计数: {patience_counter}/{patience}')

                if patience_counter >= patience:
                    print(f'🛑 早停触发! 在epoch {epoch}停止训练')
                    break

            except Exception as e:
                logging.error(f"训练过程中出现错误: {e}")
                print(f"❌ 训练过程中出现错误: {e}")
                break

        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        training_time = time.time() - start_time

        result = {
            'best_val_loss': best_val_loss,
            'best_val_acc': best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'final_epoch': epoch,
            'training_time': training_time
        }

        logging.info(f"训练完成: {result}")
        return result


class EnhancedHyperparameterOptimizer:
    """增强版超参数优化器"""

    def __init__(self, predictor, model_type):
        self.predictor = predictor
        self.model_type = model_type
        self.study = None
        self.best_params = None

        # 增强的搜索空间
        self.search_spaces = {
            'temporal_moe': {
                'hidden_dim': {'type': 'int', 'low': 192, 'high': 512},
                'num_experts': {'type': 'int', 'low': 6, 'high': 16},
                'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 5e-4, 'log': True},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.4},
                'expert_dropout': {'type': 'float', 'low': 0.05, 'high': 0.2}
            },
            'attention_lstm': {
                'hidden_dim': {'type': 'int', 'low': 192, 'high': 512},
                'num_layers': {'type': 'int', 'low': 2, 'high': 5},
                'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 5e-4, 'log': True},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.4},
                'lstm_dropout': {'type': 'float', 'low': 0.05, 'high': 0.2}
            },
            'transformer': {
                'hidden_dim': {'type': 'int', 'low': 192, 'high': 512},
                'num_layers': {'type': 'int', 'low': 3, 'high': 8},
                'num_heads': {'type': 'int', 'low': 4, 'high': 12},
                'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 5e-4, 'log': True},
                'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.3},
                'attention_dropout': {'type': 'float', 'low': 0.05, 'high': 0.15}
            }
        }

    def optimize(self, n_trials: int = EnhancedConstants.OPTIMIZATION_TRIALS,
                 timeout: int = EnhancedConstants.OPTIMIZATION_TIMEOUT) -> Dict[str, Any]:
        """执行超参数优化"""
        print(f"🔍 开始增强超参数优化: {self.model_type}")
        print(f"🎯 试验次数: {n_trials}, 超时: {timeout}秒")

        # 创建Optuna研究
        self.study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        # 执行优化
        self.study.optimize(
            lambda trial: self._objective(trial),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # 保存最佳参数
        self.best_params = self.study.best_params
        self._save_optimization_results()

        return self.best_params

    def _objective(self, trial: Trial) -> float:
        """优化目标函数"""
        try:
            # 获取搜索空间
            search_space = self.search_spaces[self.model_type]
            params = {}

            # 采样超参数
            for param_name, config in search_space.items():
                if config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, config['low'], config['high'])
                elif config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, config['low'], config['high'], log=config.get('log', False)
                    )

            # 对于transformer模型，确保hidden_dim能被num_heads整除
            if self.model_type == 'transformer':
                hidden_dim = params['hidden_dim']
                num_heads = params['num_heads']
                if hidden_dim % num_heads != 0:
                    params['hidden_dim'] = (hidden_dim // num_heads) * num_heads

            # 使用时间序列交叉验证评估参数
            avg_val_acc = self._cross_validate(params)

            return avg_val_acc

        except Exception as e:
            print(f"❌ 超参数试验失败: {e}")
            return -1.0

    def _cross_validate(self, params: Dict[str, Any], n_splits: int = 3) -> float:
        """时间序列交叉验证"""
        if not self.predictor.is_loaded:
            raise DataValidationError("请先加载数据")

        sequence_length = EnhancedConstants.DEFAULT_SEQUENCE_LENGTH
        if len(self.predictor.data) < sequence_length * 2:
            sequence_length = len(self.predictor.data) // 2

        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(self.predictor.data) // sequence_length))
        val_accuracies = []

        data_array = np.arange(len(self.predictor.data))

        for train_idx, val_idx in tscv.split(data_array):
            try:
                train_data = self.predictor.data.iloc[train_idx]
                val_data = self.predictor.data.iloc[val_idx]

                if len(train_data) < sequence_length or len(val_data) < 1:
                    continue

                # 使用增强数据集
                train_dataset = EnhancedFC3DDataset(train_data, sequence_length,
                                                    feature_columns=None, fit_scaler=True)
                val_dataset = EnhancedFC3DDataset(val_data, sequence_length,
                                                  feature_columns=train_dataset.get_feature_columns(),
                                                  fit_scaler=False, external_scaler=train_dataset.get_scaler())

                train_loader = DataLoader(train_dataset, batch_size=EnhancedConstants.BATCH_SIZE,
                                          shuffle=True, num_workers=0)
                val_loader = DataLoader(val_dataset, batch_size=EnhancedConstants.BATCH_SIZE,
                                        shuffle=False, num_workers=0)

                if len(train_loader) == 0 or len(val_loader) == 0:
                    continue

                # 创建模型
                input_dim = train_dataset.get_feature_dimension()
                config = self.predictor.model_configs[self.model_type]

                model_params = {k: v for k, v in params.items()
                                if k in ['hidden_dim', 'num_layers', 'num_heads', 'num_experts',
                                         'dropout_rate', 'expert_dropout', 'lstm_dropout', 'attention_dropout']}
                # ✅ 修复：确保输入维度正确传入
if model_type == 'attention_lstm':
    model = EnhancedAttentionLSTM(
        input_dim=input_dim,
        hidden_dim=model_params.get('hidden_dim', 256),
        num_layers=model_params.get('num_layers', 3),
        dropout_rate=model_params.get('dropout_rate', 0.2),
        lstm_dropout=model_params.get('lstm_dropout', 0.1)
    )
elif model_type == 'transformer':
    model = EnhancedTransformer(
        input_dim=input_dim,
        hidden_dim=model_params.get('hidden_dim', 256),
        num_layers=model_params.get('num_layers', 4),
        num_heads=model_params.get('num_heads', 8),
        dropout_rate=model_params.get('dropout_rate', 0.1),
        attention_dropout=model_params.get('attention_dropout', 0.05)
    )
else:
    model = config['class'](input_dim=input_dim, **model_params

                # 创建增强训练器
                trainer = EnhancedModelTrainer(model, f"{self.model_type}_cv")
                if 'learning_rate' in params:
                    trainer.optimizer = optim.AdamW(
                        model.parameters(),
                        lr=params['learning_rate'],
                        weight_decay=1e-4,
                        betas=(0.9, 0.999)
                    )

                # 快速评估
                model.train()
                for epoch in range(3):
                    total_val_acc = 0
                    total_batches = 0

                    for data, bai_target, shi_target, ge_target in val_loader:
                        data = data.to(trainer.device)
                        bai_target = bai_target.to(trainer.device).squeeze()
                        shi_target = shi_target.to(trainer.device).squeeze()
                        ge_target = ge_target.to(trainer.device).squeeze()

                        bai_probs, shi_probs, ge_probs = model(data)

                        bai_acc = trainer.calculate_accuracy(bai_probs, bai_target)
                        shi_acc = trainer.calculate_accuracy(shi_probs, shi_target)
                        ge_acc = trainer.calculate_accuracy(ge_probs, ge_target)

                        batch_acc = (bai_acc + shi_acc + ge_acc) / 3
                        total_val_acc += batch_acc
                        total_batches += 1

                    if total_batches > 0:
                        avg_val_acc = total_val_acc / total_batches
                    else:
                        avg_val_acc = 0

                val_accuracies.append(avg_val_acc)

            except Exception as e:
                print(f"⚠️ 交叉验证折叠失败: {e}")
                continue

        if val_accuracies:
            return np.mean(val_accuracies)
        else:
            return 0.0

    def _save_optimization_results(self):
        """保存优化结果"""
        if self.study and self.best_params:
            results = {
                'model_type': self.model_type,
                'best_params': self.best_params,
                'best_value': self.study.best_value,
                'completed_trials': len(self.study.trials),
                'timestamp': datetime.now().isoformat(),
                'trials_summary': [
                    {
                        'number': trial.number,
                        'value': trial.value,
                        'params': trial.params,
                        'state': str(trial.state)
                    }
                    for trial in self.study.trials
                ]
            }

            filename = f"{self.model_type}_enhanced_optimization_results.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

            print(f"💾 优化结果已保存: {filename}")
            self._display_optimization_summary()

    def _display_optimization_summary(self):
        """显示优化总结"""
        print(f"\n{'=' * 80}")
        print(f"🎉 {self.model_type.upper()} 增强超参数优化完成!")
        print(f"{'=' * 80}")
        print(f"🏆 最佳验证准确率: {self.study.best_value:.4f}")
        print(f"🔧 最佳超参数:")
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        print(f"📊 完成试验数: {len(self.study.trials)}")
        print(f"{'=' * 80}")


class EnhancedFC3DPredictor:
    """增强版FC3D预测系统主类"""

    def __init__(self):
        self.data = None
        self.models = {}
        self.current_period = None
        self.is_loaded = False
        self.feature_columns = None
        self.optimization_results = {}
        self.current_scaler = None
        self.performance_tracker = PerformanceTracker()

        # 初始化模型状态
        self._initialize_models()

        # 增强模型配置
        self.model_configs = {
            'temporal_moe': {
                'class': EnhancedTemporalMoE,
                'hidden_dim': 256,
                'num_experts': 8,
                'dropout_rate': 0.2,
                'expert_dropout': 0.1
            },
            'attention_lstm': {
                'class': EnhancedAttentionLSTM,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout_rate': 0.2,
                'lstm_dropout': 0.1
            },
            'transformer': {
                'class': EnhancedTransformer,
                'hidden_dim': 256,
                'num_layers': 4,
                'num_heads': 8,
                'dropout_rate': 0.1,
                'attention_dropout': 0.05
            }
        }

        # 加载配置和性能数据
        try:
            ConfigManager.load_config(self)
            self.performance_tracker.load_performance_data()
        except Exception as e:
            logging.warning(f"加载配置或性能数据失败: {e}")

        logging.info("增强版FC3D预测系统初始化完成")

    def _initialize_models(self):
        """初始化模型状态"""
        self.models = {}
        for model_type in ['temporal_moe', 'attention_lstm', 'transformer']:
            self.models[model_type] = {
                'model': None,
                'trainer': None,
                'info': {},
                'is_trained': False,
                'optimized_params': None
            }

    def load_data(self, file_path: str) -> bool:
        """加载数据"""
        try:
            logging.info(f"正在加载数据: {file_path}")
            print(f"📂 正在加载数据: {file_path}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"数据文件不存在: {file_path}")

            self.data = pd.read_csv(file_path)

            # 数据验证
            validation_result = DataQualityValidator.validate_data_structure(self.data)

            if not validation_result['is_valid']:
                error_msg = "数据验证失败:\n" + "\n".join(validation_result['errors'])
                raise DataValidationError(error_msg)

            # 显示数据质量报告
            report = DataQualityValidator.generate_data_report(self.data)
            print(report)

            # 数据排序
            self.data = self.data.sort_values('期号').reset_index(drop=True)

            # 检查数据量
            if len(self.data) < EnhancedConstants.MIN_DATA_POINTS:
                print(
                    f"⚠️  警告: 数据量较少({len(self.data)}期)，建议至少{EnhancedConstants.MIN_DATA_POINTS}期数据以获得更好效果")

            # 设置当前期号
            self.current_period = self.data['期号'].max()

            # 初始化特征列 - 使用增强数据集
            temp_dataset = EnhancedFC3DDataset(self.data, sequence_length=10)
            self.feature_columns = temp_dataset.get_feature_columns()

            # 重置scaler
            self.current_scaler = None

            print(f"✅ 数据加载成功! 共{len(self.data)}期数据, 最新期号: {self.current_period}")
            print(f"📊 增强特征数量: {len(self.feature_columns)}")
            self.is_loaded = True

            # 保存配置
            ConfigManager.save_config(self)

            logging.info(f"数据加载成功: {len(self.data)}行数据")
            return True

        except Exception as e:
            error_msg = f"数据加载失败: {e}"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            return False

    def prepare_datasets(self, sequence_length: int = EnhancedConstants.DEFAULT_SEQUENCE_LENGTH) -> Tuple[
        EnhancedFC3DDataset, EnhancedFC3DDataset]:
        """准备训练和验证数据集"""
        if not self.is_loaded:
            raise DataValidationError("请先加载数据")

        if len(self.data) < sequence_length + EnhancedConstants.MIN_SEQUENCE_LENGTH:
            raise DataValidationError(
                f"数据量不足，至少需要{sequence_length + EnhancedConstants.MIN_SEQUENCE_LENGTH}期数据")

        # 划分训练集和验证集
        split_idx = int((1 - EnhancedConstants.VALIDATION_SPLIT) * len(self.data))
        train_data = self.data.iloc[:split_idx]
        val_data = self.data.iloc[split_idx:]

        # 使用增强数据集
        train_dataset = EnhancedFC3DDataset(train_data, sequence_length, self.feature_columns, fit_scaler=True)
        val_dataset = EnhancedFC3DDataset(val_data, sequence_length, self.feature_columns, fit_scaler=False,
                                          external_scaler=train_dataset.get_scaler())

        # 保存scaler状态
        self.current_scaler = train_dataset.get_scaler()

        print(f"📊 增强数据集划分: 训练集{len(train_dataset)}样本, 验证集{len(val_dataset)}样本")
        logging.info(f"增强数据集划分完成: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}")

        return train_dataset, val_dataset

    def train_model(self, model_type: str, use_optimized_params: bool = False) -> bool:
        """训练指定类型的模型"""
        if not self.is_loaded:
            print("❌ 请先加载数据")
            return False

        if model_type not in self.model_configs:
            print(f"❌ 未知的模型类型: {model_type}")
            return False

        try:
            logging.info(f"开始训练增强模型: {model_type}")

            # 准备数据
            train_dataset, val_dataset = self.prepare_datasets()
            train_loader = DataLoader(train_dataset, batch_size=EnhancedConstants.BATCH_SIZE, shuffle=True,
                                      num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=EnhancedConstants.BATCH_SIZE, shuffle=False, num_workers=0)

            # 获取输入维度
            input_dim = train_dataset.get_feature_dimension()

            # 创建模型
            config = self.model_configs[model_type]

            # 检查是否使用优化参数
            if use_optimized_params and model_type in self.optimization_results:
                optimized_params = self.optimization_results[model_type]['best_params']
                model_params = {k: v for k, v in optimized_params.items()
                                if k in ['hidden_dim', 'num_layers', 'num_heads', 'num_experts',
                                         'dropout_rate', 'expert_dropout', 'lstm_dropout', 'attention_dropout']}
                learning_rate = optimized_params.get('learning_rate', EnhancedConstants.LEARNING_RATE)
                print(f"🎯 使用优化参数训练增强模型: {model_params}")
            else:
                model_params = {k: v for k, v in config.items() if k != 'class'}
                learning_rate = EnhancedConstants.LEARNING_RATE

            model = config['class'](input_dim=input_dim, **model_params)

            # 使用增强训练器训练模型
            trainer = EnhancedModelTrainer(model, model_type, learning_rate=learning_rate)
            training_info = trainer.train(train_loader, val_loader)

            # 保存模型
            self.models[model_type] = {
                'model': model,
                'trainer': trainer,
                'info': training_info,
                'is_trained': True,
                'optimized_params': model_params if use_optimized_params else None
            }

            # 显示详细训练信息
            self._display_training_summary(model_type, training_info, use_optimized_params)

            # 保存模型到文件
            self._save_model(model_type, input_dim)

            # 保存配置
            ConfigManager.save_config(self)

            logging.info(f"增强模型训练完成: {model_type}")
            return True

        except Exception as e:
            error_msg = f"增强模型训练失败: {e}"
            logging.error(error_msg)
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return False

    def _display_training_summary(self, model_type: str, training_info: Dict, use_optimized_params: bool):
        """显示训练总结"""
        best_val_loss = training_info['best_val_loss']
        best_val_acc = training_info['best_val_acc']
        training_time = training_info['training_time']

        print(f"\n{'=' * 60}")
        if use_optimized_params:
            print(f"🎉 {model_type.upper()} 增强模型(优化参数)训练完成!")
        else:
            print(f"🎉 {model_type.upper()} 增强模型训练完成!")
        print(f"{'=' * 60}")
        print(f"📈 最佳验证损失: {best_val_loss:.6f}")
        print(f"🎯 最佳验证准确率: {best_val_acc:.4f}")
        print(f"⏱️  总训练时间: {training_time:.2f}秒")
        print(f"🔄 训练轮次: {training_info['final_epoch']}")

        if training_info['train_losses']:
            print(f"📉 最终训练损失: {training_info['train_losses'][-1]:.6f}")
        else:
            print(f"📉 最终训练损失: N/A")

        if training_info['train_accuracies']:
            print(f"📊 最终训练准确率: {training_info['train_accuracies'][-1]:.4f}")
        else:
            print(f"📊 最终训练准确率: N/A")

        print(f"⚙️  训练策略: 标签平滑={EnhancedConstants.LABEL_SMOOTHING}, "
              f"梯度累积={EnhancedConstants.GRADIENT_ACCUMULATION_STEPS}")
        print(f"{'=' * 60}")

    def _save_model(self, model_type: str, input_dim: int):
        """保存模型到文件"""
        if model_type in self.models and self.models[model_type]['is_trained']:
            filename = f"{model_type}_enhanced_model.pth"
            torch.save({
                'model_state_dict': self.models[model_type]['model'].state_dict(),
                'training_info': self.models[model_type]['info'],
                'input_dim': input_dim,
                'feature_columns': self.feature_columns,
                'timestamp': datetime.now().isoformat(),
                'model_type': model_type,
                'optimized_params': self.models[model_type].get('optimized_params')
            }, filename)
            print(f"💾 增强模型已保存: {filename}")
            logging.info(f"增强模型保存成功: {filename}")

    def load_existing_models(self) -> int:
        """加载已有模型，返回成功加载的模型数量"""
        if not self.is_loaded:
            print("❌ 请先加载数据")
            return 0

        model_files = {
            'temporal_moe': 'temporal_moe_enhanced_model.pth',
            'attention_lstm': 'attention_lstm_enhanced_model.pth',
            'transformer': 'transformer_enhanced_model.pth'
        }

        loaded_count = 0

        # 先准备数据集获取特征维度
        try:
            train_dataset, _ = self.prepare_datasets()
            input_dim = train_dataset.get_feature_dimension()
            logging.info(f"获取特征维度: {input_dim}")
        except Exception as e:
            print(f"❌ 无法获取特征维度: {e}")
            return 0

        for model_type, filename in model_files.items():
            if os.path.exists(filename):
                try:
                    print(f"🔄 正在加载 {model_type} 增强模型...")

                    # 检查文件完整性
                    file_size = os.path.getsize(filename)
                    if file_size == 0:
                        print(f"⚠️ 模型文件为空: {filename}")
                        continue

                    # 创建模型架构
                    config = self.model_configs[model_type]
                    model = config['class'](
                        input_dim=input_dim,
                        **{k: v for k, v in config.items() if k != 'class'}
                    )

                    # 加载权重
                    checkpoint = torch.load(filename, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])

                    # 更新模型状态
                    self.models[model_type] = {
                        'model': model,
                        'info': checkpoint.get('training_info', {}),
                        'is_trained': True,
                        'optimized_params': checkpoint.get('optimized_params')
                    }

                    print(f"✅ 成功加载 {model_type} 增强模型")
                    loaded_count += 1
                    logging.info(f"增强模型加载成功: {model_type}")

                except Exception as e:
                    error_msg = f"加载 {model_type} 增强模型失败: {e}"
                    print(f"❌ {error_msg}")
                    logging.error(error_msg)
                    # 重置模型状态
                    self.models[model_type] = {
                        'model': None,
                        'info': {},
                        'is_trained': False,
                        'optimized_params': None
                    }
            else:
                print(f"⚠️  {model_type} 增强模型文件不存在: {filename}")
                if model_type not in self.models:
                    self.models[model_type] = {
                        'model': None,
                        'info': {},
                        'is_trained': False,
                        'optimized_params': None
                    }

        # 保存配置
        ConfigManager.save_config(self)

        print(f"\n📥 总共成功加载了 {loaded_count} 个增强模型")
        return loaded_count

    def optimize_hyperparameters(self, model_type: str,
                                 n_trials: int = EnhancedConstants.OPTIMIZATION_TRIALS,
                                 timeout: int = EnhancedConstants.OPTIMIZATION_TIMEOUT) -> bool:
        """执行增强超参数优化"""
        if model_type not in self.model_configs:
            print(f"❌ 未知的模型类型: {model_type}")
            return False

        if not self.is_loaded:
            print("❌ 请先加载数据")
            return False

        try:
            print(f"🔬 开始增强超参数优化: {model_type}")
            print(f"🎯 目标: 通过 {n_trials} 次试验找到最优超参数")

            # 创建增强优化器
            optimizer = EnhancedHyperparameterOptimizer(self, model_type)

            # 执行优化
            best_params = optimizer.optimize(n_trials=n_trials, timeout=timeout)

            # 保存优化结果
            self.optimization_results[model_type] = {
                'best_params': best_params,
                'best_score': optimizer.study.best_value if optimizer.study else 0
            }

            # 使用最优参数训练最终模型
            print(f"🚀 使用最优参数训练最终增强模型...")
            success = self.train_model(model_type, use_optimized_params=True)

            # 保存配置
            ConfigManager.save_config(self)

            return success

        except Exception as e:
            error_msg = f"增强超参数优化失败: {e}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            import traceback
            traceback.print_exc()
            return False

    def predict_next_period(self):
        """预测下一期号码"""
        if not self.is_loaded:
            print("❌ 请先加载数据")
            return

        # 检查是否有训练好的模型
        trained_models = [name for name, info in self.models.items()
                          if info.get('is_trained', False)]

        if not trained_models:
            print("❌ 没有可用的训练好的增强模型，请先训练或加载模型")
            return

        try:
            # 准备最新数据
            sequence_length = EnhancedConstants.DEFAULT_SEQUENCE_LENGTH
            if len(self.data) < sequence_length:
                sequence_length = len(self.data)
                if sequence_length < EnhancedConstants.MIN_SEQUENCE_LENGTH:
                    print(f"❌ 数据不足，至少需要{EnhancedConstants.MIN_SEQUENCE_LENGTH}期数据")
                    return

            # 取最后sequence_length条数据
            latest_data = self.data.tail(sequence_length)

            # 确保有scaler
            if self.current_scaler is None:
                print("⚠️  没有找到scaler，重新创建...")
                temp_dataset = EnhancedFC3DDataset(latest_data, sequence_length=sequence_length,
                                                   feature_columns=self.feature_columns, fit_scaler=True)
                self.current_scaler = temp_dataset.get_scaler()

            # 创建预测数据集
            dataset = EnhancedFC3DDataset(latest_data, sequence_length=sequence_length,
                                          feature_columns=self.feature_columns, fit_scaler=False,
                                          external_scaler=self.current_scaler,
                                          for_prediction=True)

            # 检查数据集是否为空
            if len(dataset) == 0:
                print("❌ 数据不足，无法创建预测序列")
                return

            # 获取预测数据
            features, _, _, _ = dataset[0]
            features = features.unsqueeze(0)

            # 计算下一期期号
            next_period = self.current_period + 1

            print(f"\n🎯 增强模型预测下一期号码 (期号: {next_period})")
            print("=" * 50)

            predictions = {}
            for model_type in trained_models:
                model_info = self.models[model_type]
                model = model_info['model']
                model.eval()

                with torch.no_grad():
                    bai_top6, shi_top6, ge_top6 = model.predict_top6(features)

                predictions[model_type] = {
                    '百位': bai_top6,
                    '十位': shi_top6,
                    '个位': ge_top6
                }

                # 显示模型类型和是否使用优化参数
                model_desc = model_type.upper()
                if model_info.get('optimized_params'):
                    model_desc += " (优化参数)"

                print(f"\n📊 {model_desc} 增强模型预测:")
                print(f"   百位候选: {bai_top6.tolist()}")
                print(f"   十位候选: {shi_top6.tolist()}")
                print(f"   个位候选: {ge_top6.tolist()}")

                # 显示推荐组合
                print("   🎲 推荐组合:")
                for i in range(min(6, len(bai_top6))):
                    combo = f"{bai_top6[i]}{shi_top6[i]}{ge_top6[i]}"
                    print(f"      {combo}")

            # 综合推荐
            print(f"\n🌟 综合推荐号码:")
            self._generate_comprehensive_recommendation(predictions)

            logging.info("增强模型预测完成")

        except Exception as e:
            error_msg = f"增强模型预测失败: {e}"
            print(f"❌ {error_msg}")
            logging.error(error_msg)
            import traceback
            traceback.print_exc()

    def _generate_comprehensive_recommendation(self, predictions: Dict):
        """生成综合推荐号码"""
        all_bai = []
        all_shi = []
        all_ge = []

        for model_pred in predictions.values():
            all_bai.extend(model_pred['百位'])
            all_shi.extend(model_pred['十位'])
            all_ge.extend(model_pred['个位'])

        # 统计频率
        bai_freq = pd.Series(all_bai).value_counts()
        shi_freq = pd.Series(all_shi).value_counts()
        ge_freq = pd.Series(all_ge).value_counts()

        print("   百位高频: ", bai_freq.head(3).index.tolist())
        print("   十位高频: ", shi_freq.head(3).index.tolist())
        print("   个位高频: ", ge_freq.head(3).index.tolist())

        # 推荐组合
        top_bai = bai_freq.index[0] if len(bai_freq) > 0 else 0
        top_shi = shi_freq.index[0] if len(shi_freq) > 0 else 0
        top_ge = ge_freq.index[0] if len(ge_freq) > 0 else 0

        print(f"   💫 最优推荐: {top_bai}{top_shi}{top_ge}")

        # 显示其他推荐组合
        print("   🎯 其他推荐:")
        for i in range(min(3, len(bai_freq), len(shi_freq), len(ge_freq))):
            bai = bai_freq.index[i] if i < len(bai_freq) else bai_freq.index[0]
            shi = shi_freq.index[i] if i < len(shi_freq) else shi_freq.index[0]
            ge = ge_freq.index[i] if i < len(ge_freq) else ge_freq.index[0]
            print(f"      {bai}{shi}{ge}")

    def calculate_position_accuracy(self, model, data_loader):
        """计算每个位置的准确率和三个位置同时命中率"""
        model.eval()
        device = next(model.parameters()).device

        bai_correct = 0
        shi_correct = 0
        ge_correct = 0
        all_correct = 0
        total_samples = 0

        with torch.no_grad():
            with ProgressBar(len(data_loader), desc="Calculating Accuracy") as pbar:
                for data, bai_target, shi_target, ge_target in data_loader:
                    data = data.to(device)
                    bai_target = bai_target.to(device).squeeze()
                    shi_target = shi_target.to(device).squeeze()
                    ge_target = ge_target.to(device).squeeze()

                    bai_probs, shi_probs, ge_probs = model(data)

                    # 获取预测结果
                    _, bai_pred = torch.max(bai_probs, 1)
                    _, shi_pred = torch.max(shi_probs, 1)
                    _, ge_pred = torch.max(ge_probs, 1)

                    # 计算每个位置的正确数
                    bai_correct += (bai_pred == bai_target).sum().item()
                    shi_correct += (shi_pred == shi_target).sum().item()
                    ge_correct += (ge_pred == ge_target).sum().item()

                    # 计算三个位置同时正确的数量
                    all_correct += ((bai_pred == bai_target) &
                                    (shi_pred == shi_target) &
                                    (ge_pred == ge_target)).sum().item()

                    total_samples += bai_target.size(0)
                    pbar.update(1)

        return {
            'bai_accuracy': bai_correct / total_samples if total_samples > 0 else 0,
            'shi_accuracy': shi_correct / total_samples if total_samples > 0 else 0,
            'ge_accuracy': ge_correct / total_samples if total_samples > 0 else 0,
            'all_accuracy': all_correct / total_samples if total_samples > 0 else 0,
            'total_samples': total_samples
        }

    def run_backtest(self):
        """运行增强模型回测"""
        if not self.models:
            print("❌ 请先训练或加载增强模型")
            return

        print("\n📊 开始增强模型回测...")
        logging.info("开始增强模型回测")

        # 使用后20%数据作为测试集
        test_data = self.data.iloc[int((1 - EnhancedConstants.VALIDATION_SPLIT) * len(self.data)):]
        if len(test_data) < EnhancedConstants.DEFAULT_SEQUENCE_LENGTH + 1:
            print("❌ 测试数据不足，无法进行回测")
            return

        test_dataset = EnhancedFC3DDataset(test_data, feature_columns=self.feature_columns, fit_scaler=False,
                                           external_scaler=self.current_scaler)
        test_loader = DataLoader(test_dataset, batch_size=EnhancedConstants.BATCH_SIZE, shuffle=False, num_workers=0)

        results = {}

        trained_models = [name for name, info in self.models.items()
                          if info.get('is_trained', False)]

        if not trained_models:
            print("❌ 没有可用的训练好的增强模型")
            return

        for model_type in trained_models:
            print(f"\n🔍 测试 {model_type} 增强模型...")
            model = self.models[model_type]['model']

            accuracy_info = self.calculate_position_accuracy(model, test_loader)
            results[model_type] = accuracy_info

            # 显示模型类型和是否使用优化参数
            model_desc = model_type.upper()
            if self.models[model_type].get('optimized_params'):
                model_desc += " (优化参数)"

            print(f"   ✅ {model_desc} 回测结果:")
            print(f"      百位准确率: {accuracy_info['bai_accuracy']:.4f}")
            print(f"      十位准确率: {accuracy_info['shi_accuracy']:.4f}")
            print(f"      个位准确率: {accuracy_info['ge_accuracy']:.4f}")
            print(f"      三位置同时命中率: {accuracy_info['all_accuracy']:.4f}")
            print(f"      测试样本数: {accuracy_info['total_samples']}")

        # 显示详细回测总结
        self._display_backtest_summary(results)

        logging.info("增强模型回测完成")
        return results

    def _display_backtest_summary(self, results: Dict):
        """显示回测详细总结"""
        print(f"\n{'=' * 80}")
        print("🎯 增强模型回测详细总结")
        print(f"{'=' * 80}")

        # 创建总结表格
        summary_data = []
        for model_type, acc_info in results.items():
            model_desc = model_type.upper()
            if self.models[model_type].get('optimized_params'):
                model_desc += " (优化)"

            summary_data.append({
                '模型': model_desc,
                '百位准确率': f"{acc_info['bai_accuracy']:.4f}",
                '十位准确率': f"{acc_info['shi_accuracy']:.4f}",
                '个位准确率': f"{acc_info['ge_accuracy']:.4f}",
                '三位置同时命中率': f"{acc_info['all_accuracy']:.4f}",
                '测试样本': acc_info['total_samples']
            })

        # 打印表格
        for data in summary_data:
            print(f"📊 {data['模型']}:")
            print(f"   百位: {data['百位准确率']} | 十位: {data['十位准确率']} | "
                  f"个位: {data['个位准确率']} | 三位置同时: {data['三位置同时命中率']}")

        # 找出最佳模型
        if results:
            best_model = max(results.items(), key=lambda x: x[1]['all_accuracy'])
            model_desc = best_model[0].upper()
            if self.models[best_model[0]].get('optimized_params'):
                model_desc += " (优化)"

            print(f"\n🏆 最佳表现增强模型: {model_desc}")
            print(f"   三位置同时命中率: {best_model[1]['all_accuracy']:.4f}")
        print(f"{'=' * 80}")

    def show_model_info(self):
        """显示增强模型信息"""
        if not self.models:
            print("❌ 没有可用的增强模型")
            return

        trained_models = [name for name, info in self.models.items()
                          if info.get('is_trained', False)]

        if not trained_models:
            print("❌ 没有训练好的增强模型")
            return

        print("\n📈 增强模型详细信息:")
        print("=" * 60)

        for model_type in trained_models:
            model_info = self.models[model_type]
            model_desc = model_type.upper()
            if model_info.get('optimized_params'):
                model_desc += " (优化参数)"

            print(f"\n🔧 {model_desc} 增强模型:")
            if 'info' in model_info:
                info = model_info['info']
                print(f"   最佳验证损失: {info.get('best_val_loss', 'N/A'):.6f}")
                print(f"   最佳验证准确率: {info.get('best_val_acc', 'N/A'):.4f}")
                print(f"   训练轮次: {info.get('final_epoch', 'N/A')}")
                print(f"   训练时间: {info.get('training_time', 'N/A'):.2f}秒")

            # 显示模型参数数量
            model = model_info['model']
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   总参数量: {total_params:,}")
            print(f"   可训练参数: {trainable_params:,}")
            print(f"   训练状态: {'✅ 已训练' if model_info.get('is_trained', False) else '❌ 未训练'}")

            # 显示优化参数信息
            if model_info.get('optimized_params'):
                print(f"   优化参数: {model_info['optimized_params']}")

    def show_optimization_results(self):
        """显示优化结果"""
        if not self.optimization_results:
            print("❌ 没有找到优化结果")
            return

        print(f"\n{'=' * 80}")
        print("📊 增强超参数优化结果总结")
        print(f"{'=' * 80}")

        for model_type, results in self.optimization_results.items():
            best_params = results.get('best_params', {})
            best_score = results.get('best_score', 0)

            print(f"\n🔧 {model_type.upper()}:")
            print(f"   最佳准确率: {best_score:.4f}")
            print(f"   最优参数:")
            for param, value in best_params.items():
                print(f"     {param}: {value}")

        print(f"{'=' * 80}")

    def show_performance_history(self):
        """显示性能历史"""
        print(f"\n{'=' * 80}")
        print("📈 增强模型性能历史")
        print(f"{'=' * 80}")

        for model_type in self.models.keys():
            summary = self.performance_tracker.get_performance_summary(model_type)
            if summary:
                model_desc = model_type.upper()
                if self.models[model_type].get('optimized_params'):
                    model_desc += " (优化)"

                print(f"\n🔍 {model_desc}:")
                print(f"   总预测次数: {summary['total_predictions']}")
                print(f"   百位命中率: {summary['accuracy_rates']['bai']:.4f}")
                print(f"   十位命中率: {summary['accuracy_rates']['shi']:.4f}")
                print(f"   个位命中率: {summary['accuracy_rates']['ge']:.4f}")
                print(f"   三位置同时命中率: {summary['accuracy_rates']['all']:.4f}")

        print(f"{'=' * 80}")

    def _run_hyperparameter_optimization_menu(self):
        """增强超参数优化子菜单"""
        opt_menu = """
⚙️  增强超参数优化菜单

1. 🔬 优化增强时序混合模型
2. 🧠 优化增强LSTM模型  
3. 🔗 优化增强Transformer模型
4. 📊 查看增强优化结果
5. ↩️  返回主菜单

请选择 (1-5): """

        while True:
            choice = input(opt_menu).strip()

            if choice == '1':
                self.optimize_hyperparameters('temporal_moe', n_trials=20, timeout=1800)
            elif choice == '2':
                self.optimize_hyperparameters('attention_lstm', n_trials=20, timeout=1800)
            elif choice == '3':
                self.optimize_hyperparameters('transformer', n_trials=20, timeout=1800)
            elif choice == '4':
                self.show_optimization_results()
            elif choice == '5':
                break
            else:
                print("❌ 无效选择")

            input("\n按Enter键继续...")

    def run_interactive_menu(self):
        """运行交互式菜单"""
        menu = """
🎰 FC3D智能交互式预测系统 - 增强优化版 🎰

1. 📂 加载数据
2. 🚀 训练增强时序混合模型  
3. 🧠 训练增强LSTM模型
4. 🔗 训练增强Transformer模型
5. 📥 加载已有增强模型
6. 🔮 预测下一期号码
7. 📊 运行各个增强模型回测
8. 📈 查看增强模型信息
9. ⚙️  增强超参数优化（提升性能）
10. 📋 查看性能历史
11. 💾 保存系统状态
12. 🚪 退出系统

请选择操作 (1-12): """

        while True:
            try:
                choice = input(menu).strip()

                if choice == '1':
                    file_path = "/Users/uajxjd/Desktop/UAFC3D.csv"
                    self.load_data(file_path)

                elif choice == '2':
                    self.train_model('temporal_moe')

                elif choice == '3':
                    self.train_model('attention_lstm')

                elif choice == '4':
                    self.train_model('transformer')

                elif choice == '5':
                    loaded_count = self.load_existing_models()
                    if loaded_count > 0:
                        print(f"✅ 成功加载 {loaded_count} 个增强模型")
                    else:
                        print("❌ 没有找到可加载的增强模型文件")

                elif choice == '6':
                    self.predict_next_period()

                elif choice == '7':
                    self.run_backtest()

                elif choice == '8':
                    self.show_model_info()

                elif choice == '9':
                    self._run_hyperparameter_optimization_menu()

                elif choice == '10':
                    self.show_performance_history()

                elif choice == '11':
                    if ConfigManager.save_config(self):
                        print("✅ 系统状态已保存")
                    else:
                        print("❌ 系统状态保存失败")

                elif choice == '12':
                    # 保存性能数据和配置
                    self.performance_tracker.save_performance_data()
                    ConfigManager.save_config(self)
                    print("👋 感谢使用增强版FC3D预测系统!")
                    logging.info("增强版系统正常退出")
                    break

                else:
                    print("❌ 无效选择，请重新输入")

                input("\n按Enter键继续...")

            except KeyboardInterrupt:
                print("\n👋 用户中断，退出系统")
                self.performance_tracker.save_performance_data()
                ConfigManager.save_config(self)
                logging.info("增强版系统被用户中断")
                break
            except Exception as e:
                error_msg = f"发生错误: {e}"
                print(f"❌ {error_msg}")
                logging.error(error_msg)
                import traceback
                traceback.print_exc()


def main():
    """主函数"""
    print("=" * 80)
    print("           FC3D智能交互式预测系统 - 增强优化版")
    print("       基于先进AI架构和增强训练策略开发")
    print("=" * 80)

    # 检查依赖
    try:
        import tqdm
        import sklearn
        import optuna
        print("✅ 所有依赖已就绪")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install tqdm matplotlib scikit-learn optuna")
        return

    predictor = EnhancedFC3DPredictor()
    predictor.run_interactive_menu()


if __name__ == "__main__":
    main()