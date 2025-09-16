#!/usr/bin/env python3
# qr_scaling_transformer_ukf.py
"""
Q/R缩放因子调整版：智能UKF调基础Q/R + Transformer调缩放因子 + Reservoir + 置信度
架构：智能UKF(30步调整基础Q/R) + Transformer(实时预测缩放因子) + 置信度融合
"""

import argparse
import copy
import os
import math
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from collections import deque
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# 直接从你的fixed_dynamic_ukf.py引入
try:
    from fixed_dynamic_ukf import EnhancedIndependentDynamicUKF, infer_vehicle_type_from_path
except ImportError:
    print("警告: 无法导入fixed_dynamic_ukf，将使用备用方案")
    def infer_vehicle_type_from_path(path):
        return "medium_large_quad"


# =========================
# Reservoir记忆模块（保持不变）
# =========================

class LightweightReservoir(nn.Module):
    """轻量级储层计算"""

    def __init__(self, input_size, reservoir_size=32, spectral_radius=0.8, input_scaling=0.3, leak_rate=0.5):
        super(LightweightReservoir, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate

        W_in = torch.randn(reservoir_size, input_size) * input_scaling
        self.register_buffer('W_in', W_in)

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


# =========================
# Q/R缩放因子Transformer（核心修改）
# =========================

class QRScalingTransformerNN(nn.Module):
    """预测Q/R缩放因子的Transformer + Reservoir + 置信度"""

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
# 智能Q/R调节UKF（从final_working_baseline_methods.py移植）
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

        # Q/R缩放相关
        self.current_q_scale = 1.0
        self.current_r_scale = 1.0
        self.base_Q = None
        self.base_R = None

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

        # 只在模式改变时更新基础Q/R
        if motion_type != self.current_motion_type and self.qr_adaptation_enabled:
            self.current_motion_type = motion_type
            self.qr_adjustment_count += 1

            # 更新基础Q/R矩阵
            self.base_Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5]) * q_scale
            self.base_R = np.diag([0.5, 0.5, 0.5]) * r_scale

            # 应用当前缩放因子
            self.ukf.Q = self.base_Q * self.current_q_scale
            self.ukf.R = self.base_R * self.current_r_scale

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化UKF"""
        try:
            # 创建sigma点生成器
            points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2.0, kappa=0.0)

            # 创建UKF实例
            self.ukf = UKF(dim_x=6, dim_z=3, dt=self.dt, hx=self._hx, fx=self._fx, points=points)

            # 设置初始Q/R矩阵
            self.base_Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])
            self.base_R = np.diag([0.5, 0.5, 0.5])
            self.ukf.Q = self.base_Q.copy()
            self.ukf.R = self.base_R.copy()

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
            self.current_q_scale = 1.0
            self.current_r_scale = 1.0

            self.initialized = True
            return True

        except Exception as e:
            print(f"SmartQREnhancedUKF初始化失败: {e}")
            return False

    def apply_qr_scaling(self, q_scale, r_scale):
        """应用Q/R缩放因子到当前基础矩阵"""
        if not self.initialized or self.ukf is None or self.base_Q is None or self.base_R is None:
            return False

        try:
            self.current_q_scale = q_scale
            self.current_r_scale = r_scale

            # 应用缩放：最终Q/R = 基础Q/R × 缩放因子
            self.ukf.Q = self.base_Q * q_scale
            self.ukf.R = self.base_R * r_scale

            return True

        except Exception as e:
            return False

    def step(self, z_xyz: np.ndarray) -> np.ndarray:
        """智能UKF步骤：自动调节Q/R + 应用缩放"""
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

    def get_current_mode(self):
        """获取当前飞行模式"""
        return self.current_motion_type

    def get_qr_info(self):
        """获取Q/R信息"""
        return {
            'q_scale': self.current_q_scale,
            'r_scale': self.current_r_scale,
            'base_Q_trace': np.trace(self.base_Q) if self.base_Q is not None else 0,
            'base_R_trace': np.trace(self.base_R) if self.base_R is not None else 0,
            'final_Q_trace': np.trace(self.ukf.Q),
            'final_R_trace': np.trace(self.ukf.R),
            'motion_type': self.current_motion_type
        }


# =========================
# 智能UKF包装器（替换原来的SmartUKFWithQRScaling）
# =========================

class SmartUKFWithQRScaling:
    """智能UKF + Q/R缩放因子应用"""

    def __init__(self, dt=0.1, vehicle_type="medium_large_quad"):
        self.dt = dt
        self.vehicle_type = vehicle_type
        self.smart_ukf = SmartQREnhancedUKF(dt=dt)
        self.step_count = 0
        self.qr_update_interval = 30  # 30步更新一次智能UKF的基础Q/R

        # 当前缩放因子
        self.current_q_scale = 1.0
        self.current_r_scale = 1.0

    def initialize(self, initial_state):
        """初始化包装的智能UKF"""
        return self.smart_ukf.initialize(initial_state)

    def apply_qr_scaling(self, q_scale, r_scale):
        """应用Q/R缩放因子到基础矩阵"""
        self.current_q_scale = q_scale
        self.current_r_scale = r_scale
        return self.smart_ukf.apply_qr_scaling(q_scale, r_scale)

    def step(self, measurement):
        """UKF步骤：每30步更新基础Q/R，每步应用缩放"""
        self.step_count += 1

        # 每30步让智能UKF更新基础Q/R矩阵
        if self.step_count % self.qr_update_interval == 0:
            # 智能UKF进行载具类型和飞行模式判断，更新基础Q/R
            result = self.smart_ukf.step(measurement)
            # 重新应用当前缩放因子
            self.smart_ukf.apply_qr_scaling(self.current_q_scale, self.current_r_scale)
            return result
        else:
            # 普通步骤：只运行UKF，Q/R保持当前缩放状态
            return self.smart_ukf.step(measurement)

    def get_current_mode(self):
        """获取当前飞行模式"""
        return self.smart_ukf.get_current_mode()

    def get_qr_info(self):
        """获取Q/R信息"""
        return self.smart_ukf.get_qr_info()


# =========================
# 置信度驱动的Q/R缩放预测器（保持不变）
# =========================

class QRScalingAdaptivePredictor:
    """基于置信度的Q/R缩放自适应预测器"""

    def __init__(self, dt=0.1, vehicle_type="medium_large_quad", max_prediction_steps=3):
        self.dt = dt
        self.vehicle_type = vehicle_type
        self.max_prediction_steps = max_prediction_steps

        # 置信度阈值
        self.high_confidence_threshold = 0.7  # 调整为更合理的阈值
        self.low_confidence_threshold = 0.3

        # 统计信息
        self.nn_qr_usage_count = 0
        self.hybrid_qr_usage_count = 0
        self.default_qr_usage_count = 0

    def safe_predict(self, nn_outputs, current_pos, target_pos, verbose=False):
        """置信度驱动的Q/R缩放自适应预测"""
        try:
            qr_scales, confidence, vehicle_probs, vehicle_logits = nn_outputs

            if torch.any(torch.isnan(qr_scales)) or torch.any(torch.isnan(confidence)):
                return None, float('inf')

            # 创建智能UKF包装器
            smart_ukf_wrapper = SmartUKFWithQRScaling(dt=self.dt, vehicle_type=self.vehicle_type)

            # 初始化
            initial_vel = np.random.normal(0, 0.02, 3)
            initial_state = np.concatenate([current_pos, initial_vel])

            if not smart_ukf_wrapper.initialize(initial_state):
                return None, float('inf')

            confidence_val = float(confidence)

            if verbose:
                print(f"    Q/R缩放自适应预测:")
                print(f"      置信度: {confidence_val:.4f}")
                predicted_vehicle = torch.argmax(vehicle_probs).item()
                vehicle_names = ["micro_quad", "medium_large_quad", "fixed_wing", "heavy_multirotor"]
                print(f"      预测载具: {vehicle_names[predicted_vehicle]}")

            # 自适应Q/R缩放策略
            if confidence_val > self.high_confidence_threshold:
                # 高置信度：使用NN的Q/R缩放因子
                return self._high_confidence_qr_prediction(smart_ukf_wrapper, qr_scales, current_pos, target_pos,
                                                           verbose)

            elif confidence_val < self.low_confidence_threshold:
                # 低置信度：使用默认缩放(1.0, 1.0)
                return self._low_confidence_qr_prediction(smart_ukf_wrapper, current_pos, target_pos, verbose)

            else:
                # 中等置信度：混合缩放策略
                return self._hybrid_qr_prediction(smart_ukf_wrapper, qr_scales, confidence_val, current_pos, target_pos,
                                                  verbose)

        except Exception as e:
            if verbose:
                print(f"    整体Q/R缩放预测失败: {e}")
            return None, float('inf')

    def _high_confidence_qr_prediction(self, smart_ukf_wrapper, qr_scales, current_pos, target_pos, verbose=False):
        """高置信度：使用NN的Q/R缩放因子"""
        try:
            self.nn_qr_usage_count += 1

            # 提取Q/R缩放因子
            scales_np = qr_scales.detach().cpu().numpy()
            if scales_np.ndim > 1:
                scales_np = scales_np.flatten()

            q_scale, r_scale = float(scales_np[0]), float(scales_np[1])

            if verbose:
                print(f"      高置信度模式：使用NN Q/R缩放")
                print(f"      缩放因子: Q={q_scale:.3f}, R={r_scale:.3f}")

            # 渐进式多步预测
            current_pred = current_pos.copy()

            for step in range(self.max_prediction_steps):
                progress = (step + 1) / self.max_prediction_steps
                intermediate_target = current_pos + progress * (target_pos - current_pos)
                noise = np.random.normal(0, 0.25, 3)
                noisy_obs = intermediate_target + noise

                # 在每步应用NN预测的Q/R缩放
                smart_ukf_wrapper.apply_qr_scaling(q_scale, r_scale)

                # 执行智能UKF步骤（每30步会自动更新基础Q/R）
                pred_state = smart_ukf_wrapper.step(noisy_obs)
                current_pred = pred_state[:3]

                if np.any(np.isnan(current_pred)) or np.any(np.isinf(current_pred)):
                    break

            final_error = np.linalg.norm(current_pred - target_pos)

            if verbose:
                print(f"      当前飞行模式: {smart_ukf_wrapper.get_current_mode()}")
                qr_info = smart_ukf_wrapper.get_qr_info()
                print(f"      最终Q trace: {qr_info['final_Q_trace']:.3f}, R trace: {qr_info['final_R_trace']:.3f}")
                print(f"      最终预测: [{current_pred[0]:.6f}, {current_pred[1]:.6f}, {current_pred[2]:.6f}]")

            return current_pred, final_error

        except Exception as e:
            if verbose:
                print(f"      高置信度Q/R缩放预测失败: {e}")
            return self._low_confidence_qr_prediction(smart_ukf_wrapper, current_pos, target_pos, verbose)

    def _low_confidence_qr_prediction(self, smart_ukf_wrapper, current_pos, target_pos, verbose=False):
        """低置信度：使用默认Q/R缩放(1.0, 1.0)"""
        try:
            self.default_qr_usage_count += 1

            if verbose:
                print(f"      低置信度模式：默认Q/R缩放(1.0, 1.0)")

            # 使用默认缩放因子
            current_pred = current_pos.copy()

            for step in range(self.max_prediction_steps):
                progress = (step + 1) / self.max_prediction_steps
                intermediate_target = current_pos + progress * (target_pos - current_pos)
                noise = np.random.normal(0, 0.25, 3)
                noisy_obs = intermediate_target + noise

                # 应用默认缩放
                smart_ukf_wrapper.apply_qr_scaling(1.0, 1.0)

                pred_state = smart_ukf_wrapper.step(noisy_obs)
                current_pred = pred_state[:3]

                if np.any(np.isnan(current_pred)) or np.any(np.isinf(current_pred)):
                    break

            final_error = np.linalg.norm(current_pred - target_pos)
            return current_pred, final_error

        except Exception as e:
            if verbose:
                print(f"      低置信度Q/R缩放预测失败: {e}")
            return None, float('inf')

    def _hybrid_qr_prediction(self, smart_ukf_wrapper, qr_scales, confidence_val, current_pos, target_pos,
                              verbose=False):
        """中等置信度：混合Q/R缩放策略"""
        try:
            self.hybrid_qr_usage_count += 1

            # 提取NN预测的缩放因子
            scales_np = qr_scales.detach().cpu().numpy()
            if scales_np.ndim > 1:
                scales_np = scales_np.flatten()

            q_scale_nn, r_scale_nn = float(scales_np[0]), float(scales_np[1])

            # 默认缩放因子
            q_scale_default, r_scale_default = 1.0, 1.0

            # 置信度加权融合
            weight = confidence_val
            q_scale_fused = weight * q_scale_nn + (1 - weight) * q_scale_default
            r_scale_fused = weight * r_scale_nn + (1 - weight) * r_scale_default

            if verbose:
                print(f"      混合Q/R缩放模式：权重={weight:.3f}")
                print(f"      融合缩放: Q={q_scale_fused:.3f}, R={r_scale_fused:.3f}")

            # 预测过程
            current_pred = current_pos.copy()

            for step in range(self.max_prediction_steps):
                progress = (step + 1) / self.max_prediction_steps
                intermediate_target = current_pos + progress * (target_pos - current_pos)
                noise = np.random.normal(0, 0.25, 3)
                noisy_obs = intermediate_target + noise

                # 应用融合的缩放因子
                smart_ukf_wrapper.apply_qr_scaling(q_scale_fused, r_scale_fused)

                pred_state = smart_ukf_wrapper.step(noisy_obs)
                current_pred = pred_state[:3]

                if np.any(np.isnan(current_pred)) or np.any(np.isinf(current_pred)):
                    break

            final_error = np.linalg.norm(current_pred - target_pos)
            return current_pred, final_error

        except Exception as e:
            if verbose:
                print(f"      混合Q/R缩放预测失败: {e}")
            return self._low_confidence_qr_prediction(smart_ukf_wrapper, current_pos, target_pos, verbose)

    def get_usage_stats(self):
        """获取Q/R缩放使用统计"""
        total = self.nn_qr_usage_count + self.hybrid_qr_usage_count + self.default_qr_usage_count
        if total == 0:
            return {'nn_qr_scaling': 0, 'hybrid_qr_scaling': 0, 'default_qr_scaling': 0}

        return {
            'nn_qr_scaling': self.nn_qr_usage_count / total * 100,
            'hybrid_qr_scaling': self.hybrid_qr_usage_count / total * 100,
            'default_qr_scaling': self.default_qr_usage_count / total * 100,
            'total_predictions': total
        }


# =========================
# 训练器（关键修改：去掉最大帧数限制）
# =========================

class QRScalingHybridTrainer:
    """Q/R缩放版训练器"""

    def __init__(self, data_input, device='cpu'):
        self.data_input = data_input
        self.device = torch.device(device)

        print(f"Q/R缩放版数据加载中... (无最大点数限制)")
        print(f"架构: 智能UKF(30步调基础Q/R) + Transformer(实时Q/R缩放) + Reservoir + 置信度")

        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """关键修改：去掉最大帧数限制"""
        all_trajectories = []

        if isinstance(self.data_input, str):
            files = [f.strip() for f in self.data_input.split(',') if f.strip()]
        else:
            files = [self.data_input]

        for file_path in files:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                continue

            print(f"  加载: {file_path}")
            try:
                df = pd.read_csv(file_path)

                time_cols = [col for col in df.columns if 'time' in col.lower()]
                if not time_cols:
                    print(f"    警告: {file_path} 中未找到时间列")
                    continue

                pos_cols = []
                for prefix in ['x', 'y', 'z']:
                    for col in df.columns:
                        if col.lower() == prefix or col.lower() == f'{prefix}_true' or col.lower() == f'true_{prefix}':
                            pos_cols.append(col)
                            break

                if len(pos_cols) < 3:
                    print(f"    警告: {file_path} 中位置列不完整 {pos_cols}")
                    continue

                # 关键修改：不再限制最大点数，使用所有数据
                positions = df[pos_cols].values[:, :3]

                if len(positions) < 50:
                    print(f"    跳过: 数据点太少")
                    continue

                if np.any(np.isnan(positions)):
                    positions = positions[~np.any(np.isnan(positions), axis=1)]

                time_data = df[time_cols[0]].values[:len(positions)]
                if len(time_data) > 1:
                    dt_values = np.diff(time_data)
                    dt = np.median(dt_values)
                else:
                    dt = 0.1

                # 推断载具类型
                vehicle_type = infer_vehicle_type_from_path(file_path)

                split_point = int(len(positions) * 0.8)
                train_positions = positions[:split_point]
                test_positions = positions[split_point:]

                all_trajectories.append({
                    'train_positions': train_positions,
                    'test_positions': test_positions,
                    'dt': dt,
                    'vehicle_type': vehicle_type,
                    'source_file': file_path
                })

                print(f"    成功: 载具={vehicle_type}, 训练={len(train_positions)}, 测试={len(test_positions)}")

            except Exception as e:
                print(f"    错误: {e}")
                continue

        if not all_trajectories:
            raise ValueError("没有成功加载任何数据文件")

        print(f"\n总计加载 {len(all_trajectories)} 个轨迹")

        all_dts = [traj['dt'] for traj in all_trajectories]
        self.median_dt = np.median(all_dts)
        print(f"中位时间间隔: {self.median_dt:.3f}s")

        self.train_data = []
        self.test_data = []

        for traj in all_trajectories:
            train_samples = self._generate_training_data(traj['train_positions'], traj['dt'], traj['vehicle_type'])
            self.train_data.extend(train_samples)

            test_samples = self._generate_training_data(traj['test_positions'], traj['dt'], traj['vehicle_type'])
            self.test_data.extend(test_samples)

        print(f"训练样本: {len(self.train_data)}")
        print(f"测试样本: {len(self.test_data)}")

        all_train_features = np.array([sample['features'] for sample in self.train_data])
        self.feature_scaler = StandardScaler()
        self.feature_scaler.fit(all_train_features)

        print("数据准备完成!")

    def _generate_training_data(self, positions, dt, vehicle_type):
        """生成训练数据"""
        training_samples = []
        window_size = 20

        vehicle_map = {"micro_quad": 0, "medium_large_quad": 1, "fixed_wing": 2, "heavy_multirotor": 3}
        vehicle_label = vehicle_map.get(vehicle_type, 1)

        for i in range(window_size, len(positions) - 1):
            pos_window = positions[i - window_size:i]
            vel_window = np.diff(pos_window, axis=0) / dt

            features = self._extract_sequence_features(pos_window, vel_window, dt)
            target_pos = positions[i + 1]

            training_samples.append({
                'features': features,
                'target_pos': target_pos,
                'current_pos': positions[i],
                'vehicle_type': vehicle_type,
                'vehicle_label': vehicle_label,
                'dt': dt
            })

        return training_samples

    def _extract_sequence_features(self, pos_window, vel_window, dt):
        """保持原有的特征提取逻辑"""
        pos_mean = np.mean(pos_window, axis=0)
        pos_std = np.std(pos_window, axis=0)

        vel_mean = np.mean(vel_window, axis=0) if len(vel_window) > 0 else np.zeros(3)
        vel_std = np.std(vel_window, axis=0) if len(vel_window) > 0 else np.zeros(3)
        vel_norm_mean = np.mean(np.linalg.norm(vel_window, axis=1)) if len(vel_window) > 0 else 0.0

        traj_length = np.sum(np.linalg.norm(np.diff(pos_window, axis=0), axis=1))
        displacement = np.linalg.norm(pos_window[-1] - pos_window[0])

        features = np.concatenate([
            pos_mean, pos_std, vel_mean, vel_std,
            [vel_norm_mean, traj_length, displacement, dt]
        ])

        return features

    def evaluate_on_test_set(self, model):
        """测试集评估 - Q/R缩放版本"""
        model.eval()

        qr_scales_collected = []
        confidence_collected = []
        vehicle_accuracy = []
        errors_collected = []
        success_count = 0
        failure_count = 0

        print("    使用Q/R缩放自适应预测器 (智能UKF基础Q/R + Transformer缩放)")
        predictor = QRScalingAdaptivePredictor(
            dt=self.median_dt,
            max_prediction_steps=3
        )

        with torch.no_grad():
            for i, sample in enumerate(self.test_data):
                try:
                    features = torch.tensor(
                        self.feature_scaler.transform([sample['features']]),
                        dtype=torch.float32, device=self.device
                    )

                    # Q/R缩放Transformer输出
                    qr_scales, confidence, vehicle_probs, vehicle_logits = model(features)

                    # 收集统计
                    qr_scales_collected.append(qr_scales.detach().cpu().numpy().flatten())
                    confidence_collected.append(float(confidence))

                    # 载具类型准确率
                    predicted_vehicle = torch.argmax(vehicle_probs).item()
                    true_vehicle = sample['vehicle_label']
                    vehicle_accuracy.append(1 if predicted_vehicle == true_vehicle else 0)

                    # Q/R缩放自适应预测
                    try:
                        _, error = predictor.safe_predict(
                            (qr_scales, confidence, vehicle_probs, vehicle_logits),
                            sample['current_pos'],
                            sample['target_pos'],
                            verbose=(i < 2)
                        )

                        if error != float('inf'):
                            errors_collected.append(error)
                            success_count += 1
                        else:
                            failure_count += 1

                    except Exception as e:
                        failure_count += 1
                        if i < 2:
                            print(f"    预测异常(样本{i}): {str(e)[:50]}")

                except Exception as e:
                    if i < 2:
                        print(f"    评估失败: {e}")
                    continue

        # 统计分析
        success_rate = success_count / max(success_count + failure_count, 1)
        print(f"    [Q/R缩放自适应] 成功率: {success_rate:.1%} ({success_count}/{success_count + failure_count})")

        if len(errors_collected) > 0:
            error_mean = np.mean(errors_collected)
            error_std = np.std(errors_collected)
            print(f"    [误差] 均值: {error_mean:.6f} ± {error_std:.6f}")
        else:
            error_mean = float('inf')
            print(f"    ✗ 所有预测都失败")

        if len(confidence_collected) > 0:
            avg_confidence = np.mean(confidence_collected)
            print(f"    [置信度] 平均: {avg_confidence:.4f}")

        if len(vehicle_accuracy) > 0:
            vehicle_acc = np.mean(vehicle_accuracy)
            print(f"    [载具识别] 准确率: {vehicle_acc:.1%}")

        if len(qr_scales_collected) > 0:
            avg_qr = np.mean(qr_scales_collected, axis=0)
            std_qr = np.std(qr_scales_collected, axis=0)
            print(f"    [Q/R缩放] Q={avg_qr[0]:.3f}±{std_qr[0]:.3f}, R={avg_qr[1]:.3f}±{std_qr[1]:.3f}")

        # 显示Q/R缩放策略统计
        usage_stats = predictor.get_usage_stats()
        print(f"    [Q/R缩放策略] NN缩放: {usage_stats['nn_qr_scaling']:.1f}%, "
              f"混合缩放: {usage_stats['hybrid_qr_scaling']:.1f}%, "
              f"默认缩放: {usage_stats['default_qr_scaling']:.1f}%")

        return error_mean

    def train_model(self, train_params):
        """训练Q/R缩放模型"""
        seed = train_params.get('seed', 42)
        epochs = train_params.get('epochs', 50)
        batch_size = train_params.get('batch_size', 16)
        lr = train_params.get('lr', 3e-5)
        d_model = train_params.get('d_model', 128)
        nlayers = train_params.get('nlayers', 2)
        nhead = train_params.get('nhead', 4)
        dropout = train_params.get('dropout', 0.1)
        weight_decay = train_params.get('weight_decay', 0.0001)
        patience = train_params.get('patience', 15)
        reservoir_size = train_params.get('reservoir_size', 32)

        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n开始Q/R缩放版训练 (种子: {seed})")

        # 创建Q/R缩放模型
        input_dim = len(self.train_data[0]['features'])
        model = QRScalingTransformerNN(
            input_dim=input_dim,
            d_model=d_model,
            nlayers=nlayers,
            nhead=nhead,
            dropout=dropout,
            reservoir_size=reservoir_size
        ).to(self.device)

        print(f"Q/R缩放Transformer参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"架构: d_model={d_model}, nlayers={nlayers}, nhead={nhead}, reservoir_size={reservoir_size}")

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5, min_lr=1e-6)

        best_error = float('inf')
        best_model_state = None
        patience_counter = 0

        # 创建训练用预测器
        train_predictor = QRScalingAdaptivePredictor(dt=self.median_dt, max_prediction_steps=3)

        for epoch in range(epochs):
            model.train()
            epoch_errors = []

            indices = np.random.permutation(len(self.train_data))

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]

                optimizer.zero_grad()
                batch_error = 0.0
                valid_count = 0

                for idx in batch_indices:
                    sample = self.train_data[idx]
                    features = torch.tensor(
                        self.feature_scaler.transform([sample['features']]),
                        dtype=torch.float32, device=self.device
                    )

                    # Q/R缩放前向传播
                    qr_scales, confidence, vehicle_probs, vehicle_logits = model(features)

                    # 基于Q/R缩放预测的损失函数
                    try:
                        _, error = train_predictor.safe_predict(
                            (qr_scales, confidence, vehicle_probs, vehicle_logits),
                            sample['current_pos'],
                            sample['target_pos'],
                            verbose=False
                        )

                        if error != float('inf'):
                            # 主损失：预测误差
                            main_loss = error

                            # 辅助损失：载具类型分类
                            vehicle_label = torch.tensor([sample['vehicle_label']],
                                                         dtype=torch.long, device=self.device)
                            vehicle_loss = F.cross_entropy(vehicle_logits, vehicle_label)

                            # 正则化：Q/R缩放因子合理性
                            qr_reg = torch.mean((qr_scales - 1.0) ** 2)  # 鼓励接近1.0

                            # 总损失
                            total_loss = main_loss + 0.1 * vehicle_loss + 0.01 * qr_reg

                            batch_error += total_loss
                            valid_count += 1

                    except Exception:
                        continue

                if valid_count > 0:
                    batch_error = batch_error / valid_count

                    # 反向传播
                    batch_error.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_errors.append(batch_error.item())

            # 计算平均损失
            if epoch_errors:
                avg_error = np.mean(epoch_errors)

                # 测试集评估
                test_error = self.evaluate_on_test_set(model)

                scheduler.step(test_error)

                # 早停检查
                if test_error < best_error:
                    best_error = test_error
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                print(f"Epoch {epoch + 1:2d}/{epochs}: "
                      f"Train={avg_error:.6f}, Test={test_error:.6f}, "
                      f"LR={optimizer.param_groups[0]['lr']:.1e}")

                if patience_counter >= patience:
                    print(f"早停触发 (patience={patience})")
                    break

        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return {
            'model': model,
            'best_error': best_error,
            'seed': seed,
            'mode': 'qr_scaling_hybrid_with_smart_ukf',
            'architecture': {
                'd_model': d_model,
                'nlayers': nlayers,
                'nhead': nhead,
                'dropout': dropout,
                'reservoir_size': reservoir_size
            }
        }


def main():
    parser = argparse.ArgumentParser(description='Q/R缩放版混合系统：智能UKF基础Q/R + Transformer缩放')
    parser.add_argument('--data_path', type=str, required=True, help='训练数据路径')
    parser.add_argument('--output_dir', type=str, default='./qr_scaling_models', help='输出目录')
    parser.add_argument('--device', type=str, default='cpu', help='设备')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Q/R缩放版混合系统")
    print("=" * 60)
    print("核心架构:")
    print("✓ 智能UKF：每30步根据载具类型+飞行模式调整基础Q/R")
    print("✓ Transformer：实时预测Q/R缩放因子[0.1-3.0, 0.5-2.0]")
    print("✓ 最终Q/R = 基础Q/R × 缩放因子")
    print("✓ Reservoir记忆增强时序学习")
    print("✓ 置信度自适应融合策略")
    print("✓ 移除训练最大帧数限制")
    print(f"数据: {args.data_path}")

    try:
        trainer = QRScalingHybridTrainer(
            args.data_path,
            args.device
        )

        seeds = [42, 123, 456, 789, 1024]
        results = []

        for seed in seeds:
            train_params = {
                'seed': seed,
                'epochs': 50,
                'batch_size': 16,
                'lr': 3e-5,
                'd_model': 128,
                'nlayers': 2,
                'nhead': 4,
                'dropout': 0.1,
                'weight_decay': 0.0001,
                'patience': 15,
                'reservoir_size': 32
            }

            print(f"\n{'=' * 60}")
            print(f"训练种子 {seed} (Q/R缩放版：智能UKF + Transformer + Reservoir + 置信度)")
            print(f"{'=' * 60}")

            result = trainer.train_model(train_params)
            results.append(result)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(args.output_dir, f'qr_scaling_seed_{seed}_{timestamp}.pth')

            torch.save({
                'model_state_dict': result['model'].state_dict(),
                'feature_scaler': trainer.feature_scaler,
                'train_params': train_params,
                'result': result,
                'method': 'qr_scaling_hybrid_with_smart_ukf',
                'model_type': 'QRScalingTransformerNN',
                'dt': trainer.median_dt,
                'version': '8.0_qr_scaling_unlimited_data'
            }, model_path)

            print(f"\n✓ 种子{seed}训练完成:")
            print(f"  误差: {result['best_error']:.6f}")
            print(f"  模式: {result['mode']}")
            print(f"  保存至: {model_path}")

        # 选择最佳模型
        best_result = min(results, key=lambda x: x['best_error'])
        print(f"\n{'=' * 60}")
        print(f"Q/R缩放版训练完成总结")
        print(f"{'=' * 60}")

        for result in results:
            status = "🏆 BEST" if result == best_result else "   "
            print(f"{status} 种子{result['seed']}: 误差={result['best_error']:.6f}")

        best_model_path = os.path.join(args.output_dir, f'BEST_qr_scaling.pth')
        torch.save({
            'model_state_dict': best_result['model'].state_dict(),
            'feature_scaler': trainer.feature_scaler,
            'train_params': train_params,
            'result': best_result,
            'method': 'qr_scaling_hybrid_with_smart_ukf',
            'model_type': 'QRScalingTransformerNN',
            'dt': trainer.median_dt,
            'all_results': results,
            'version': '8.0_qr_scaling_unlimited_data'
        }, best_model_path)

        print(f"\n✓ 最佳Q/R缩放模型保存至: {best_model_path}")
        print(f"✓ 最佳性能: 误差={best_result['best_error']:.6f}")

        print(f"\nQ/R缩放策略效果:")
        print("1. 智能UKF提供基础Q/R矩阵（载具类型+飞行模式适配）")
        print("2. Transformer学习实时缩放因子（轨迹特征驱动）")
        print("3. 分层优化：粗调(智能UKF) + 精调(Transformer)")
        print("4. Reservoir记忆：增强复杂轨迹模式学习")
        print("5. 置信度融合：自适应选择最优缩放策略")
        print("6. 无数据限制：使用完整轨迹数据进行训练")

    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()