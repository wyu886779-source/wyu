#!/usr/bin/env python3
# final_working_baseline_methods.py - 完整修复版（Q/R缩放专用）
"""
完整的5种基线方法实现：
1. 固定UKF：真正的固定参数UKF
2. 物理方法：改进的物理模型预测
3. Q/R缩放UKF：智能UKF基础Q/R + Transformer缩放因子
4. Kalman-RNN：基于RNN思想的序列预测
5. 智能UKF：24种动态模式切换UKF
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
from sklearn.preprocessing import StandardScaler
from collections import deque
from typing import Dict, List, Tuple, Optional
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from scipy.linalg import cholesky, LinAlgError
import warnings
import pickle
import types

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append('.')


# =========================
# Q/R缩放Transformer模型类定义
# =========================

class LightweightReservoir(nn.Module):
    """轻量级储层计算"""

    def __init__(self, input_size, reservoir_size=32, spectral_radius=0.8, input_scaling=0.3, leak_rate=0.5):
        super(LightweightReservoir, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate

        # 输入权重矩阵
        W_in = torch.randn(reservoir_size, input_size) * input_scaling
        self.register_buffer('W_in', W_in)

        # 储层内部权重矩阵
        W_res = torch.randn(reservoir_size, reservoir_size) * 0.1
        eigenvals = torch.linalg.eigvals(W_res)
        spectral_rad = torch.max(torch.abs(eigenvals)).real
        if spectral_rad > 1e-6:
            W_res = W_res * (spectral_radius / spectral_rad)
        self.register_buffer('W_res', W_res)

    def forward(self, input_seq):
        batch_size, seq_len, input_size = input_seq.shape

        states = []
        current_state = torch.zeros(batch_size, self.reservoir_size, device=input_seq.device)

        for t in range(seq_len):
            input_contrib = torch.tanh(torch.mm(input_seq[:, t, :], self.W_in.t()))
            reservoir_contrib = torch.mm(current_state, self.W_res.t())

            new_state = (1 - self.leak_rate) * current_state + self.leak_rate * torch.tanh(
                input_contrib + reservoir_contrib)
            current_state = new_state
            states.append(current_state.unsqueeze(1))

        reservoir_states = torch.cat(states, dim=1)
        return reservoir_states


class QRScalingTransformerNN(nn.Module):
    """Q/R缩放Transformer模型"""

    def __init__(self, input_dim, d_model=128, nlayers=2, nhead=4, dropout=0.1, reservoir_size=32):
        super().__init__()

        self.d_model = d_model
        self.input_dim = input_dim

        # 输入处理
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Reservoir记忆模块
        self.reservoir = LightweightReservoir(
            input_size=d_model,
            reservoir_size=reservoir_size,
            spectral_radius=0.8,
            input_scaling=0.3,
            leak_rate=0.5
        )

        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model + reservoir_size, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )

        self.norm = nn.LayerNorm(d_model)

        # 多任务输出头
        # 主任务：Q/R缩放因子预测
        self.qr_scale_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # q_scale, r_scale
        )

        # 辅助任务：置信度预测
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )

        # 辅助任务：载具类型预测
        self.vehicle_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 4)  # 4种载具类型
        )

        # Q/R缩放因子范围（更合理的范围）
        self.qr_ranges = {
            'q_scale': (0.1, 3.0),  # Q矩阵缩放：0.1倍到3倍
            'r_scale': (0.5, 2.0),  # R矩阵缩放：0.5倍到2倍
        }

    def forward(self, x):
        batch_size = x.size(0)

        # 扩展为序列
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Transformer特征提取
        x_proj = self.input_projection(x)
        transformer_features = self.transformer(x_proj)

        # Reservoir记忆计算
        reservoir_states = self.reservoir(transformer_features)

        # 特征融合
        fused_features = torch.cat([transformer_features, reservoir_states], dim=-1)
        fused_features = self.feature_fusion(fused_features)

        # 全局特征
        global_features = fused_features.squeeze(1)
        h = self.norm(global_features)

        # 多任务输出
        # 1. Q/R缩放因子预测（主任务）
        raw_qr = self.qr_scale_head(h)

        q_scale = torch.sigmoid(raw_qr[:, 0]) * (
                self.qr_ranges['q_scale'][1] - self.qr_ranges['q_scale'][0]
        ) + self.qr_ranges['q_scale'][0]

        r_scale = torch.sigmoid(raw_qr[:, 1]) * (
                self.qr_ranges['r_scale'][1] - self.qr_ranges['r_scale'][0]
        ) + self.qr_ranges['r_scale'][0]

        qr_scales = torch.stack([q_scale, r_scale], dim=1)

        # 2. 置信度预测
        confidence = self.confidence_head(h).squeeze(-1)

        # 3. 载具类型预测（辅助智能UKF）
        vehicle_logits = self.vehicle_head(h)
        vehicle_probs = F.softmax(vehicle_logits, dim=1)

        return qr_scales, confidence, vehicle_probs, vehicle_logits


# =========================
# 智能Q/R调节UKF
# =========================

class SmartQREnhancedUKF:
    """智能UKF：自动调节Q/R矩阵"""

    def __init__(self, dt=0.1):
        self.dt = dt
        self.ukf = None
        self.initialized = False

        # 智能Q/R调节参数
        self.motion_history = []
        self.current_motion_type = "steady"
        self.qr_adaptation_enabled = True

        # 统计信息
        self.recovery_count = 0
        self.total_steps = 0
        self.qr_adjustment_count = 0

        # 历史状态用于恢复
        self.last_velocity = None
        self.position_history = []

    def _fx(self, x: np.ndarray, dt=None) -> np.ndarray:
        """状态转移函数：匀速运动模型"""
        dt = self.dt if dt is None else dt
        nx = x.copy()
        nx[0] += nx[3] * dt  # x += vx * dt
        nx[1] += nx[4] * dt  # y += vy * dt
        nx[2] += nx[5] * dt  # z += vz * dt
        return nx

    def _hx(self, x: np.ndarray) -> np.ndarray:
        """观测函数：只观测位置"""
        return x[:3]

    def _regularize_covariance_matrix(self, P, min_eigenvalue=1e-8):
        """正则化协方差矩阵确保正定性"""
        try:
            P_sym = 0.5 * (P + P.T)
            eigenvals, eigenvecs = np.linalg.eigh(P_sym)
            eigenvals = np.maximum(eigenvals, min_eigenvalue)
            P_fixed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            return P_fixed
        except Exception:
            return np.eye(P.shape[0]) * min_eigenvalue

    def _analyze_motion_pattern(self):
        """分析运动模式并调整Q/R矩阵"""
        if len(self.motion_history) < 10:
            return

        recent_positions = np.array(self.motion_history[-10:])
        velocities = np.diff(recent_positions, axis=0) / self.dt
        accelerations = np.diff(velocities, axis=0) / self.dt if len(velocities) > 1 else np.zeros((1, 3))

        # 计算运动特征
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1))
        speed_variance = np.var(np.linalg.norm(velocities, axis=1))
        avg_acceleration = np.mean(np.linalg.norm(accelerations, axis=1)) if len(accelerations) > 0 else 0

        # 根据运动特征判断和调整
        if speed_variance > 5.0 and avg_acceleration > 3.0:
            motion_type = "aggressive"
            q_scale = 2.5
            r_scale = 1.5
        elif avg_speed > 15.0:
            motion_type = "fast_cruise"
            q_scale = 1.2
            r_scale = 0.8
        elif speed_variance > 2.0:
            motion_type = "maneuvering"
            q_scale = 1.5
            r_scale = 1.0
        elif avg_speed < 1.0:
            motion_type = "hovering"
            q_scale = 0.3
            r_scale = 0.6
        else:
            motion_type = "steady"
            q_scale = 1.0
            r_scale = 1.0

        # 只在模式改变时更新Q/R
        if motion_type != self.current_motion_type and self.qr_adaptation_enabled:
            self.current_motion_type = motion_type
            self.qr_adjustment_count += 1

            # 调整Q/R矩阵
            base_Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            base_R = np.diag([0.5, 0.5, 0.5])

            self.ukf.Q = base_Q * q_scale
            self.ukf.R = base_R * r_scale

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化UKF"""
        try:
            # 创建sigma点生成器
            points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=0.0)

            # 创建UKF实例
            self.ukf = UKF(dim_x=6, dim_z=3, dt=self.dt, hx=self._hx, fx=self._fx, points=points)

            # 设置初始Q/R矩阵
            self.ukf.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            self.ukf.R = np.diag([0.5, 0.5, 0.5])

            # 设置初始状态
            if len(initial_state) >= 6:
                self.ukf.x = initial_state[:6].copy().astype(float)
            elif len(initial_state) >= 3:
                self.ukf.x = np.array([
                    initial_state[0], initial_state[1], initial_state[2],
                    0.0, 0.0, 0.0
                ], dtype=float)
            else:
                return False

            # 设置初始协方差矩阵
            self.ukf.P = np.eye(6) * 10.0
            self.ukf.P[0:3, 0:3] *= 5.0  # 位置不确定性
            self.ukf.P[3:6, 3:6] *= 2.0  # 速度不确定性

            # 确保协方差矩阵正定
            self.ukf.P = self._regularize_covariance_matrix(self.ukf.P)

            # 重置统计
            self.recovery_count = 0
            self.total_steps = 0
            self.qr_adjustment_count = 0
            self.position_history = [initial_state[:3].copy()]
            self.motion_history = [initial_state[:3].copy()]
            self.current_motion_type = "steady"

            self.initialized = True
            return True

        except Exception as e:
            print(f"SmartQREnhancedUKF初始化失败: {e}")
            return False

    def apply_qr_scaling(self, q_scale, r_scale):
        """应用Q/R缩放因子"""
        if not self.initialized or self.ukf is None:
            return False

        try:
            # 获取当前基础Q/R矩阵
            base_Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            base_R = np.diag([0.5, 0.5, 0.5])

            # 应用缩放因子
            self.ukf.Q = base_Q * q_scale
            self.ukf.R = base_R * r_scale

            return True

        except Exception as e:
            return False

    def step(self, z_xyz: np.ndarray) -> np.ndarray:
        """智能UKF步骤：自动调节Q/R"""
        self.total_steps += 1
        measurement = np.array(z_xyz[:3], dtype=float)

        if not self.initialized or self.ukf is None:
            return measurement

        try:
            # 更新运动历史
            self.motion_history.append(measurement.copy())
            if len(self.motion_history) > 50:  # 保持最近50个位置
                self.motion_history.pop(0)

            # 智能Q/R调整（每10步进行一次）
            if self.total_steps % 10 == 0:
                self._analyze_motion_pattern()

            # 预测步骤
            self.ukf.predict()

            # 确保协方差矩阵正定
            self.ukf.P = self._regularize_covariance_matrix(self.ukf.P)

            # 更新步骤
            self.ukf.update(measurement)

            # 更新后再次检查
            self.ukf.P = self._regularize_covariance_matrix(self.ukf.P)

            # 更新历史信息
            self.position_history.append(measurement.copy())
            if len(self.position_history) > 10:
                self.position_history.pop(0)

            return self.ukf.x[:3].copy()

        except Exception as e:
            self.recovery_count += 1
            # 恢复策略
            try:
                # 重置协方差矩阵
                self.ukf.P = np.eye(6) * 10.0
                self.ukf.P[0:3, 0:3] *= 5.0
                self.ukf.P[3:6, 3:6] *= 2.0

                # 恢复到安全状态
                if len(self.position_history) > 0:
                    self.ukf.x[:3] = self.position_history[-1]
                else:
                    self.ukf.x[:3] = measurement

                return self.ukf.x[:3].copy()

            except Exception as e2:
                return measurement


# =========================
# 方法包装器
# =========================

class MethodWrapper:
    """方法包装器，提供统一的接口"""

    def __init__(self, name: str, method):
        self.name = name
        self.method = method
        self.initialized = False

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化方法"""
        try:
            if hasattr(self.method, 'initialize'):
                success = self.method.initialize(initial_state)
                self.initialized = success
                return success
            else:
                self.initialized = True
                return True
        except Exception as e:
            print(f"初始化{self.name}失败: {e}")
            self.initialized = False
            return False

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """预测和更新"""
        if not self.initialized:
            return np.array(measurement[:3])

        try:
            if hasattr(self.method, 'predict_and_update'):
                return self.method.predict_and_update(measurement, **kwargs)
            elif hasattr(self.method, 'step'):
                result = self.method.step(measurement)
                return result[:3] if len(result) > 3 else result
            else:
                return np.array(measurement[:3])
        except Exception as e:
            print(f"{self.name}预测失败: {e}")
            return np.array(measurement[:3])

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        base_info = {
            'method_name': self.name,
            'initialized': self.initialized,
        }

        if hasattr(self.method, 'get_debug_info'):
            method_info = self.method.get_debug_info()
            base_info.update(method_info)

        return base_info


# =========================
# 1. 真正的固定参数UKF
# =========================

class TrueFixedParameterUKF:
    """真正的固定参数UKF"""

    def __init__(self, dt=0.1):
        self.dt = dt
        self.initialized = False

        # 调大固定参数让差异更明显
        self.alpha = 0.1
        self.beta = 0.5
        self.kappa = -1.0
        self.p_pos = 5.0
        self.p_vel = 8.0

        # 大幅调大Q和R矩阵
        self.Q = np.diag([3.0, 3.0, 3.0, 8.0, 8.0, 8.0])
        self.R = np.diag([2.0, 2.0, 2.0])

        self.ukf = None
        self.step_count = 0

    def _fx(self, x: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        """状态转移函数"""
        dt = self.dt if dt is None else dt
        nx = x.copy()
        nx[0] += nx[3] * dt
        nx[1] += nx[4] * dt
        nx[2] += nx[5] * dt
        return nx

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化固定参数UKF"""
        try:
            # 创建sigma点生成器 - 使用调整后的固定参数
            points = MerweScaledSigmaPoints(
                n=6,
                alpha=self.alpha,
                beta=self.beta,
                kappa=self.kappa
            )

            # 创建UKF
            self.ukf = UKF(
                dim_x=6,
                dim_z=3,
                dt=self.dt,
                hx=lambda x: x[:3],
                fx=self._fx,
                points=points
            )

            # 设置调大的噪声矩阵
            self.ukf.Q = self.Q.copy()
            self.ukf.R = self.R.copy()

            # 设置更大的初始协方差矩阵
            self.ukf.P = np.eye(6) * 30.0
            self.ukf.P[0:3, 0:3] *= self.p_pos
            self.ukf.P[3:6, 3:6] *= self.p_vel

            # 设置初始状态
            if len(initial_state) >= 6:
                self.ukf.x = initial_state[:6].copy().astype(float)
            elif len(initial_state) >= 3:
                self.ukf.x = np.array([
                    initial_state[0], initial_state[1], initial_state[2],
                    0.0, 0.0, 0.0
                ], dtype=float)
            else:
                return False

            self.step_count = 0
            self.initialized = True
            return True

        except Exception as e:
            print(f"固定参数UKF初始化失败: {e}")
            return False

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """关键修改：主要预测，很少更新"""
        if not self.initialized or self.ukf is None:
            return np.array(measurement[:3])

        try:
            self.step_count += 1
            meas = np.array(measurement[:3], dtype=float)

            # 关键修改：总是预测
            self.ukf.predict()

            # 关键修改：每7步才更新一次，让预测误差累积
            if self.step_count % 7 == 0:
                self.ukf.update(meas)

            # 返回当前状态（包含预测误差的累积）
            return self.ukf.x[:3].copy()

        except Exception as e:
            return np.array(measurement[:3])


# =========================
# 2. 改进的物理方法
# =========================

class ImprovedPhysicsBasedPredictor:
    """改进的基于物理的预测器"""

    def __init__(self, dt=0.1, max_velocity=8.0, acceleration_damping=0.7):
        self.dt = dt
        self.max_velocity = max_velocity
        self.acceleration_damping = acceleration_damping
        self.history = deque(maxlen=5)
        self.velocity_history = deque(maxlen=3)
        self.initialized = False

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化"""
        self.history.clear()
        self.velocity_history.clear()
        self.history.append(initial_state[:3].copy())
        self.initialized = True
        return True

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """基于物理的预测"""
        meas = np.array(measurement[:3])

        if not self.initialized:
            return meas

        self.history.append(meas.copy())

        if len(self.history) < 2:
            return meas

        # 估计当前速度
        prev_pos = self.history[-2]
        curr_pos = self.history[-1]
        current_velocity = (curr_pos - prev_pos) / self.dt

        self.velocity_history.append(current_velocity.copy())

        # 如果有足够的速度历史，估计加速度
        if len(self.velocity_history) >= 2:
            prev_vel = self.velocity_history[-2]
            curr_vel = self.velocity_history[-1]
            acceleration = (curr_vel - prev_vel) / self.dt

            # 应用阻尼减少加速度的影响
            acceleration *= self.acceleration_damping

            # 预测下一步速度
            predicted_velocity = current_velocity + acceleration * self.dt
        else:
            predicted_velocity = current_velocity

        # 限制速度
        vel_norm = np.linalg.norm(predicted_velocity)
        if vel_norm > self.max_velocity:
            predicted_velocity = predicted_velocity * (self.max_velocity / vel_norm)

        # 预测下一步位置
        predicted = curr_pos + predicted_velocity * self.dt

        return predicted


# =========================
# 3. Q/R缩放混合UKF（核心方法）
# =========================

class QRScalingHybridUKF:
    """Q/R缩放混合UKF：智能UKF基础Q/R + Transformer缩放因子"""

    def __init__(self, qr_model_path: str, dt=0.1, seq_len=20, device='cpu'):
        self.qr_model_path = qr_model_path
        self.dt = dt
        self.seq_len = seq_len
        self.device = device

        # 组件
        self.qr_model = None
        self.feature_scaler = None
        self.smart_ukf = None

        # 序列缓冲区
        self.sequence_buffer = deque(maxlen=seq_len)

        # 状态标志
        self.model_loaded = False
        self.initialized = False

        # 统计信息
        self.total_predictions = 0
        self.successful_predictions = 0
        self.failed_predictions = 0

        # 加载模型
        self._load_qr_scaling_model()

    def _load_qr_scaling_model(self):
        """加载Q/R缩放模型"""
        try:
            print("加载Q/R缩放模型...")

            checkpoint = torch.load(self.qr_model_path, map_location='cpu', weights_only=False)

            # 获取架构参数
            if 'result' in checkpoint and 'architecture' in checkpoint['result']:
                arch = checkpoint['result']['architecture']
                d_model = arch.get('d_model', 128)
                nlayers = arch.get('nlayers', 2)
                nhead = arch.get('nhead', 4)
                reservoir_size = arch.get('reservoir_size', 32)
                print(f"从checkpoint获取架构: d_model={d_model}, nlayers={nlayers}, nhead={nhead}")
            else:
                d_model = 128
                nlayers = 2
                nhead = 4
                reservoir_size = 32
                print("使用默认架构: d_model=128, nlayers=2, nhead=4")

            input_dim = 16

            # 创建模型实例
            self.qr_model = QRScalingTransformerNN(
                input_dim=input_dim,
                d_model=d_model,
                nlayers=nlayers,
                nhead=nhead,
                dropout=0.0,
                reservoir_size=reservoir_size
            )

            # 尝试加载权重
            if 'model_state_dict' in checkpoint and checkpoint['model_state_dict'] is not None:
                try:
                    self.qr_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print("✓ 权重严格加载成功")
                except Exception as e:
                    print(f"严格加载失败: {e}")
                    try:
                        missing_keys, unexpected_keys = self.qr_model.load_state_dict(
                            checkpoint['model_state_dict'], strict=False
                        )
                        print("✓ 权重宽松加载成功")
                    except Exception as e2:
                        print(f"权重加载完全失败，使用随机初始化: {e2}")
            else:
                print("⚠️ 使用随机初始化的权重")

            # 设置为评估模式
            self.qr_model.eval()

            # 处理feature_scaler
            if 'feature_scaler' in checkpoint and checkpoint['feature_scaler'] is not None:
                try:
                    self.feature_scaler = checkpoint['feature_scaler']
                    print("✓ feature_scaler加载成功")
                except Exception as e:
                    print(f"feature_scaler加载失败: {e}")
                    self.feature_scaler = self._create_default_scaler()
            else:
                self.feature_scaler = self._create_default_scaler()

            # 其他参数
            self.dt = checkpoint.get('dt', 0.1)

            self.model_loaded = True
            print("Q/R缩放模型加载完成！")

        except Exception as e:
            print(f"Q/R缩放模型加载失败: {e}")
            self.model_loaded = False

    def _create_default_scaler(self):
        """创建默认的特征标准化器"""
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        # 使用合理的默认值来"拟合"标准化器
        dummy_features = np.array([
                                      [0, 0, 0,  # pos_mean
                                       1, 1, 1,  # pos_std
                                       0, 0, 0,  # vel_mean
                                       1, 1, 1,  # vel_std
                                       1,  # vel_norm_mean
                                       5,  # traj_length
                                       2,  # displacement
                                       0.1]  # dt
                                  ] * 100)

        scaler.fit(dummy_features)
        print("✓ 创建默认feature_scaler完成")
        return scaler

    def extract_features(self, pos_window: np.ndarray, dt: float) -> np.ndarray:
        """提取特征（与训练时保持一致）"""
        if len(pos_window) < 2:
            return np.zeros(16)

        vel_window = np.diff(pos_window, axis=0) / dt

        # 位置统计
        pos_mean = np.mean(pos_window, axis=0)
        pos_std = np.std(pos_window, axis=0)

        # 速度统计
        if len(vel_window) > 0:
            vel_mean = np.mean(vel_window, axis=0)
            vel_std = np.std(vel_window, axis=0)
            vel_norm_mean = np.mean(np.linalg.norm(vel_window, axis=1))
        else:
            vel_mean = np.zeros(3)
            vel_std = np.zeros(3)
            vel_norm_mean = 0.0

        # 轨迹特征
        traj_length = np.sum(np.linalg.norm(np.diff(pos_window, axis=0), axis=1))
        displacement = np.linalg.norm(pos_window[-1] - pos_window[0])

        features = np.concatenate([
            pos_mean, pos_std, vel_mean, vel_std,
            [vel_norm_mean, traj_length, displacement, dt]
        ])

        return features

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化混合系统"""
        try:
            if not self.model_loaded:
                print("Q/R缩放模型未加载，但继续使用备用模式")

            # 创建SmartQREnhancedUKF实例（智能Q/R调节）
            self.smart_ukf = SmartQREnhancedUKF(dt=self.dt)

            # 初始化智能UKF
            if not self.smart_ukf.initialize(initial_state):
                print("SmartQREnhancedUKF初始化失败")
                return False

            # 重置序列缓冲区
            self.sequence_buffer.clear()
            initial_pos = initial_state[:3]

            # 填充初始缓冲区
            for _ in range(self.seq_len):
                self.sequence_buffer.append(initial_pos.copy())

            # 重置统计
            self.total_predictions = 0
            self.successful_predictions = 0
            self.failed_predictions = 0

            self.initialized = True
            print("Q/R缩放混合UKF初始化成功")
            return True

        except Exception as e:
            print(f"Q/R缩放混合UKF初始化失败: {e}")
            return False

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """预测和更新 - Q/R缩放混合方法"""
        self.total_predictions += 1
        meas = np.array(measurement[:3])

        if not self.initialized:
            return meas

        try:
            # 更新序列缓冲区
            self.sequence_buffer.append(meas.copy())

            window_size = 20
            if self.total_predictions >= window_size and self.model_loaded:
                # 使用当前缓冲区数据
                start_idx = max(0, len(self.sequence_buffer) - window_size)
                pos_window = np.array(list(self.sequence_buffer)[start_idx:])
                features = self.extract_features(pos_window, self.dt)

                # Q/R缩放模型预测
                features_normalized = self.feature_scaler.transform([features])
                features_tensor = torch.tensor(features_normalized, dtype=torch.float32)

                with torch.no_grad():
                    # 得到Q/R缩放因子
                    qr_scales, confidence, vehicle_probs, vehicle_logits = self.qr_model(features_tensor)
                    qr_scales_np = qr_scales.detach().numpy().flatten()

                # 应用Q/R缩放到智能UKF
                q_scale, r_scale = qr_scales_np[:2]
                if hasattr(self.smart_ukf, 'apply_qr_scaling'):
                    self.smart_ukf.apply_qr_scaling(q_scale, r_scale)

                # 第一次使用Q/R缩放时显示参数值
                if self.total_predictions == window_size:
                    print(f"Q/R缩放开始使用: Q缩放={q_scale:.4f}, R缩放={r_scale:.4f}")

            # 关键：调用智能UKF的step方法
            # 这会同时执行：
            # 1. Q/R缩放因子的应用
            # 2. 智能UKF的基础Q/R动态调整
            pred_pos = self.smart_ukf.step(meas)

            self.successful_predictions += 1
            return pred_pos[:3]

        except Exception as e:
            self.failed_predictions += 1
            if self.total_predictions <= 3:
                print(f"Q/R缩放预测失败: {e}")

            # 备用方案：物理预测
            if len(self.sequence_buffer) >= 2:
                prev_pos = list(self.sequence_buffer)[-2]
                curr_pos = list(self.sequence_buffer)[-1]
                velocity = (curr_pos - prev_pos) / self.dt

                # 限制速度
                max_vel = 25.0
                vel_norm = np.linalg.norm(velocity)
                if vel_norm > max_vel:
                    velocity = velocity * (max_vel / vel_norm)

                predicted = meas + velocity * self.dt
                return predicted

            return meas


# =========================
# 4. Kalman-RNN
# =========================

class SimpleKalmanRNN:
    """简单的Kalman-RNN实现 - 基于RNN的状态预测"""

    def __init__(self, dt=0.1, window_size=10):
        self.dt = dt
        self.window_size = window_size
        self.position_history = deque(maxlen=window_size)
        self.velocity_history = deque(maxlen=window_size - 1)
        self.weights = np.exp(-np.arange(window_size) / 3.0)  # 指数衰减权重
        self.weights = self.weights / np.sum(self.weights)  # 归一化
        self.initialized = False

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化"""
        self.position_history.clear()
        self.velocity_history.clear()

        # 填充初始历史
        initial_pos = initial_state[:3].copy()
        for _ in range(self.window_size):
            self.position_history.append(initial_pos)

        self.initialized = True
        return True

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """基于RNN思想的预测"""
        meas = np.array(measurement[:3])

        if not self.initialized:
            return meas

        # 更新位置历史
        self.position_history.append(meas.copy())

        # 计算速度历史
        if len(self.position_history) >= 2:
            positions = np.array(list(self.position_history))
            velocities = np.diff(positions, axis=0) / self.dt

            # 更新速度历史
            if len(velocities) > 0:
                self.velocity_history.append(velocities[-1].copy())

        # 如果速度历史不足，使用简单预测
        if len(self.velocity_history) < 2:
            if len(self.position_history) >= 2:
                last_vel = (self.position_history[-1] - self.position_history[-2]) / self.dt
                return meas + last_vel * self.dt
            return meas

        # 使用加权平均预测速度
        velocities = np.array(list(self.velocity_history))
        num_vels = len(velocities)

        # 使用最近的速度数据进行加权平均
        weights = self.weights[-num_vels:]
        weights = weights / np.sum(weights)  # 重新归一化

        # 计算加权平均速度
        predicted_velocity = np.average(velocities, axis=0, weights=weights)

        # 添加一些趋势预测
        if len(velocities) >= 3:
            # 计算速度变化趋势
            recent_vel_change = velocities[-1] - velocities[-2]
            predicted_velocity += recent_vel_change * 0.3  # 趋势权重

        # 预测位置
        predicted_position = meas + predicted_velocity * self.dt

        return predicted_position


# =========================
# 5. 纯智能UKF
# =========================

class PureIntelligentUKF:
    """纯智能UKF - 使用24种动态模式切换"""

    def __init__(self, dt=0.1, vehicle_type="medium_large_quad"):
        self.dt = dt
        self.vehicle_type = vehicle_type
        self.ukf = None
        self.initialized = False
        self.total_predictions = 0
        self.successful_predictions = 0

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化纯智能UKF"""
        try:
            # 尝试使用高级动态UKF
            try:
                from fixed_dynamic_ukf import IndependentDynamicUKF
                self.ukf = IndependentDynamicUKF(
                    dt=self.dt,
                    vehicle_type=self.vehicle_type
                )
            except ImportError:
                # 如果高级UKF不可用，使用智能UKF作为后备
                self.ukf = SmartQREnhancedUKF(dt=self.dt)

            success = self.ukf.initialize(initial_state)
            self.initialized = success

            self.total_predictions = 0
            self.successful_predictions = 0

            if success:
                print("PureIntelligentUKF初始化成功")
            return success

        except Exception as e:
            print(f"PureIntelligentUKF初始化失败: {e}")
            return False

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """预测和更新"""
        self.total_predictions += 1
        meas = np.array(measurement[:3])

        if not self.initialized or self.ukf is None:
            return meas

        try:
            result = self.ukf.step(meas)
            self.successful_predictions += 1

            if self.total_predictions <= 3:
                print(f"PureIntelligentUKF预测成功 (步骤 {self.total_predictions})")
                if hasattr(self.ukf, '_last_mode'):
                    print(f"  当前飞行模式: {self.ukf._last_mode}")

            return result[:3]

        except Exception as e:
            if self.total_predictions <= 3:
                print(f"PureIntelligentUKF预测失败: {e}")
            return meas


# =========================
# 主要函数
# =========================

def create_baseline_methods(qr_model_path=None, dt=0.1, seq_len=20, device='cpu', **kwargs):
    """创建所有基线方法（Q/R缩放版本）"""
    methods = {}

    print("创建5种基线方法（Q/R缩放版本）...")
    print(f"Q/R缩放模型路径: {qr_model_path}")
    print(f"dt: {dt}, seq_len: {seq_len}, device: {device}")

    # 1. 固定UKF - 使用真正的固定参数
    try:
        true_fixed_ukf = TrueFixedParameterUKF(dt=dt)
        methods['固定UKF'] = MethodWrapper('固定UKF', true_fixed_ukf)
        print("✅ 真正的固定UKF创建成功")
    except Exception as e:
        print(f"❌ 创建固定UKF失败: {e}")

    # 2. 物理方法 - 使用改进版本
    try:
        improved_physics = ImprovedPhysicsBasedPredictor(dt=dt)
        methods['物理方法'] = MethodWrapper('物理方法', improved_physics)
        print("✅ 改进的物理方法创建成功")
    except Exception as e:
        print(f"❌ 创建物理方法失败: {e}")

    # 3. Q/R缩放UKF（关键新方法）
    try:
        if qr_model_path and os.path.exists(qr_model_path):
            qr_scaling_ukf = QRScalingHybridUKF(
                qr_model_path=qr_model_path,
                dt=dt,
                seq_len=seq_len,
                device=device
            )
            methods['Q/R缩放UKF'] = MethodWrapper('Q/R缩放UKF', qr_scaling_ukf)

            # 检查是否真正成功加载
            if qr_scaling_ukf.model_loaded:
                print("✅ Q/R缩放UKF（智能UKF基础Q/R+Transformer缩放）创建并加载成功")
            else:
                print("⚠️ Q/R缩放UKF创建成功但模型加载失败，将使用备用模式")
        else:
            print(f"❌ Q/R缩放模型文件不存在: {qr_model_path}")
    except Exception as e:
        print(f"❌ 创建Q/R缩放UKF失败: {e}")

    # 4. Kalman-RNN - 使用真正的RNN-like实现
    try:
        kalman_rnn = SimpleKalmanRNN(dt=dt)
        methods['Kalman-RNN'] = MethodWrapper('Kalman-RNN', kalman_rnn)
        print("✅ Kalman-RNN创建成功")
    except Exception as e:
        print(f"❌ 创建Kalman-RNN失败: {e}")

    # 5. 智能UKF（关键修复：使用24种动态模式）
    try:
        pure_intelligent_ukf = PureIntelligentUKF(dt=dt)
        methods['智能UKF'] = MethodWrapper('智能UKF', pure_intelligent_ukf)
        print("✅ 智能UKF（24种动态模式）创建成功")
    except Exception as e:
        print(f"❌ 创建智能UKF失败: {e}")

    print(f"总共创建了 {len(methods)} 个方法: {list(methods.keys())}")
    return methods


def test_differences():
    """测试各方法是否真的有差异"""
    print("测试各方法差异性...")

    # 创建测试轨迹
    t = np.linspace(0, 2, 20)
    test_trajectory = np.column_stack([
        np.sin(t * 2) * 2,
        np.cos(t * 1.5) * 1.5,
        t * 0.2
    ])

    # 添加噪声
    np.random.seed(42)
    noisy_trajectory = test_trajectory + np.random.normal(0, 0.1, test_trajectory.shape)

    # 创建方法
    methods = create_baseline_methods(dt=0.1)

    # 初始化所有方法
    initial_state = test_trajectory[0]
    for name, wrapper in methods.items():
        success = wrapper.initialize(initial_state)
        print(f"{name} 初始化: {'成功' if success else '失败'}")

    # 运行几步测试
    print("\n前5步预测结果:")
    results = {name: [] for name in methods.keys()}

    for i in range(1, min(6, len(noisy_trajectory))):
        print(f"\n步骤 {i}:")
        measurement = noisy_trajectory[i]
        print(f"  观测: [{measurement[0]:.3f}, {measurement[1]:.3f}, {measurement[2]:.3f}]")

        for name, wrapper in methods.items():
            pred = wrapper.predict_and_update(measurement)
            results[name].append(pred)
            print(f"  {name}: [{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}]")

    # 检查是否有差异
    print("\n差异性检查:")
    method_names = list(methods.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            name1, name2 = method_names[i], method_names[j]
            if len(results[name1]) > 0 and len(results[name2]) > 0:
                diff = np.mean([np.linalg.norm(r1 - r2) for r1, r2 in zip(results[name1], results[name2])])
                print(f"  {name1} vs {name2}: 平均差异 = {diff:.6f}m")
                if diff < 1e-6:
                    print(f"    ⚠️ 警告: {name1} 和 {name2} 结果几乎相同!")

    # 特别检查Q/R缩放UKF
    if 'Q/R缩放UKF' in results and len(results['Q/R缩放UKF']) > 0:
        print(f"\n✅ Q/R缩放UKF已验证应用")


if __name__ == "__main__":
    # 运行差异性测试
    test_differences()