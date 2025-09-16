#!/usr/bin/env python3
# simplified_run_experiments.py - 完整集成实验（Q/R缩放专用版）
"""
集成7种高质量基线方法对比实验：
- 5种核心方法（固定UKF、物理方法、Q/R缩放UKF、Kalman-RNN、智能UKF）
- VECTOR方法（使用训练好的vector_best_model.pth）
- 微分平坦性物理预测器（针对4种载具类型）
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import time
from typing import Optional, Dict, Any, List, Tuple
from scipy.optimize import minimize, least_squares

warnings.filterwarnings("ignore", category=UserWarning)

# 导入核心基线方法
sys.path.append('.')
try:
    from final_working_baseline_methods import create_baseline_methods, MethodWrapper

    print("✅ 成功导入核心基线方法")
except ImportError as e:
    print(f"❌ 导入核心基线方法失败: {e}")
    sys.exit(1)

# 导入新轨迹预测方法（仅用于Trajectron++）
try:
    from new_trajectory_methods import (
        create_optimized_trajectron_plus_plus,
        get_available_methods,
    )

    NEW_METHODS_AVAILABLE = True
    print("✅ 成功导入Trajectron++相关方法")
except ImportError as e:
    NEW_METHODS_AVAILABLE = False
    print(f"⚠️ Trajectron++方法导入失败: {e}")
    print("   将只使用核心方法、VECTOR和物理方法")
# 在现有导入之后添加
try:
    from final_working_baseline_methods import (
        QRScalingTransformerNN,
        LightweightReservoir,
        SmartQREnhancedUKF
    )
    print("✅ 成功导入Q/R缩放相关类定义")
except ImportError as e:
    print(f"⚠️ Q/R缩放类导入失败: {e}")

# =========================
# 微分平坦性物理预测器
# =========================
import numpy as np
from collections import defaultdict


def print_final_results_with_stats(all_runs_results, num_runs=5):
    """输出多次运行的统计结果（均值±标准差）"""
    print(f"\n{'=' * 60}")
    print(f"轨迹预测方法对比结果 (基于{num_runs}次独立运行)")
    print("=" * 60)

    # 收集所有方法在所有运行中的指标
    method_stats = defaultdict(lambda: {
        'ADE': [], 'FDE': [], 'Success_Rate': [], 'Processing_Time': []
    })

    # 遍历所有运行结果
    for run_results in all_runs_results:
        for result in run_results:
            for method_name, metrics in result.items():
                if 'dataset_info' in method_name:
                    continue

                if metrics['ADE'] != float('inf'):
                    method_stats[method_name]['ADE'].append(metrics['ADE'])
                    method_stats[method_name]['FDE'].append(metrics['FDE'])
                    method_stats[method_name]['Success_Rate'].append(metrics['Success_Rate'])
                    method_stats[method_name]['Processing_Time'].append(metrics['Processing_Time'])

    # 计算每个方法的统计量
    final_stats = []
    for method_name, stats in method_stats.items():
        if len(stats['ADE']) > 0:
            ade_mean = np.mean(stats['ADE'])
            ade_std = np.std(stats['ADE'])
            fde_mean = np.mean(stats['FDE'])
            fde_std = np.std(stats['FDE'])
            success_mean = np.mean(stats['Success_Rate'])
            success_std = np.std(stats['Success_Rate'])
            time_mean = np.mean(stats['Processing_Time'])
            time_std = np.std(stats['Processing_Time'])

            final_stats.append((
                method_name,
                ade_mean, ade_std,
                fde_mean, fde_std,
                success_mean, success_std,
                time_mean, time_std,
                len(stats['ADE'])  # 有效运行次数
            ))

    # 按ADE均值排序
    final_stats.sort(key=lambda x: x[1])

    print("\n总体性能排名 (均值±标准差):")
    print("-" * 80)
    for rank, (method_name, ade_mean, ade_std, fde_mean, fde_std,
               success_mean, success_std, time_mean, time_std, valid_runs) in enumerate(final_stats, 1):
        method_type = "🧠" if any(x in method_name for x in ['Q/R', 'VECTOR', 'Transformer', 'Trajectron']) else "⚙️"

        print(f"#{rank} {method_type} {method_name} ({valid_runs}/{num_runs} runs):")
        print(f"   ADE: {ade_mean:.6f}±{ade_std:.6f}m")
        print(f"   FDE: {fde_mean:.6f}±{fde_std:.6f}m")
        print(f"   Success Rate: {success_mean:.1f}±{success_std:.1f}%")
        print(f"   Processing Time: {time_mean:.3f}±{time_std:.3f}ms")
        print()

    # 论文格式输出
    print("论文表格格式:")
    print("-" * 80)
    print("Method | ADE (m) | FDE (m) | Success Rate (%) | Time (ms)")
    print("-" * 80)
    for method_name, ade_mean, ade_std, fde_mean, fde_std, success_mean, success_std, time_mean, time_std, valid_runs in final_stats:
        print(
            f"{method_name} | {ade_mean:.6f}±{ade_std:.6f} | {fde_mean:.6f}±{fde_std:.6f} | {success_mean:.1f}±{success_std:.1f} | {time_mean:.3f}±{time_std:.3f}")

    # LaTeX表格格式
    print("\nLaTeX表格格式:")
    print("-" * 80)
    for method_name, ade_mean, ade_std, fde_mean, fde_std, success_mean, success_std, time_mean, time_std, valid_runs in final_stats:
        latex_name = method_name.replace('_', '\\_')
        print(
            f"{latex_name} & ${ade_mean:.6f} \\pm {ade_std:.6f}$ & ${fde_mean:.6f} \\pm {fde_std:.6f}$ & ${success_mean:.1f} \\pm {success_std:.1f}$ & ${time_mean:.3f} \\pm {time_std:.3f}$ \\\\")

    # Q/R缩放方法特别分析
    print(f"\n{'=' * 60}")
    print("Q/R缩放方法统计分析:")
    print("=" * 60)

    qr_found = False
    for method_name, ade_mean, ade_std, fde_mean, fde_std, success_mean, success_std, time_mean, time_std, valid_runs in final_stats:
        if 'Q/R缩放' in method_name:
            rank = [x[0] for x in final_stats].index(method_name) + 1
            print(f"🎯 {method_name}:")
            print(f"   排名: #{rank}/{len(final_stats)}")
            print(f"   ADE稳定性: {ade_mean:.6f}±{ade_std:.6f}m (变异系数: {ade_std / ade_mean * 100:.2f}%)")
            print(f"   FDE稳定性: {fde_mean:.6f}±{fde_std:.6f}m (变异系数: {fde_std / fde_mean * 100:.2f}%)")
            print(
                f"   成功率稳定性: {success_mean:.1f}±{success_std:.1f}% (变异系数: {success_std / success_mean * 100:.2f}%)")
            print(f"   计算效率: {time_mean:.3f}±{time_std:.3f}ms")
            print(f"   有效运行: {valid_runs}/{num_runs}")

            # 性能评价
            cv_ade = ade_std / ade_mean * 100
            performance_level = "优秀" if rank <= 2 else "良好" if rank <= 4 else "中等"
            stability_level = "高" if cv_ade < 10 else "中" if cv_ade < 20 else "低"

            print(f"   性能评价: {performance_level}")
            print(f"   稳定性: {stability_level} (ADE变异系数 {cv_ade:.2f}%)")
            print(f"   实时性: {'满足' if time_mean < 10.0 else '不满足'} (<10ms要求)")
            qr_found = True
            break

    if not qr_found:
        print("❌ Q/R缩放方法未参与评估")

    return final_stats
class DifferentialFlatnessTrajectoryPredictor:
    """基于微分平坦性的UAV轨迹预测器"""

    def __init__(self, dt=0.1, vehicle_type="medium_large_quad"):
        self.dt = dt
        self.vehicle_type = vehicle_type

        # 载具物理参数
        self.params = self._get_vehicle_parameters(vehicle_type)

        # 状态历史缓冲区
        self.position_history = deque(maxlen=15)
        self.velocity_history = deque(maxlen=10)
        self.acceleration_history = deque(maxlen=8)

        # 微分平坦性相关
        self.polynomial_order = 7  # 用于轨迹拟合的多项式阶数
        self.optimization_horizon = 5  # 优化时域步数

        # 约束参数
        self.constraints = self._get_dynamic_constraints()

        # 参数估计
        self.mass_estimator = AdaptiveMassEstimator(self.params['nominal_mass'])
        self.drag_estimator = DragCoefficientEstimator()

        # 统计信息
        self.prediction_count = 0
        self.constraint_violations = 0
        self.optimization_failures = 0

        self.initialized = False

    def _get_vehicle_parameters(self, vehicle_type: str) -> Dict:
        """获取载具特定的物理参数"""
        params_db = {
            "micro_quad": {
                "nominal_mass": 0.20,  # kg
                "max_thrust": 51.9,  # N
                "max_tilt_angle": np.deg2rad(60),
                "max_angular_velocity": np.deg2rad(600),
                "drag_coefficient": 0.2351,
                "moment_arm": 0.1,  # m
                "base_max_snap": 200.0,  # m/s⁴
            },
            "medium_large_quad": {
                "nominal_mass": 1.50,  # kg
                "max_thrust": 79.2,  # N
                "max_tilt_angle": np.deg2rad(50),
                "max_angular_velocity": np.deg2rad(140),
                "drag_coefficient": 0.0182,
                "moment_arm": 0.25,  # m
                "base_max_snap": 760.4,  # m/s⁴
            },
            "fixed_wing": {
                "nominal_mass": 2.60,  # kg
                "max_thrust": 50.0,  # N
                "max_bank_angle": np.deg2rad(60),
                "max_load_factor": 3.0,
                "min_velocity": 16.2,  # m/s
                "drag_coefficient": 0.0250,
                "lift_coefficient": 0.8,
                "wing_area": 0.5,  # m^2
                "base_max_snap": 10.1,  # m/s⁴
            },
            "heavy_multirotor": {
                "nominal_mass": 4.00,  # kg
                "max_thrust": 251.0,  # N
                "max_tilt_angle": np.deg2rad(45),
                "max_angular_velocity": np.deg2rad(173.9),
                "drag_coefficient": 0.0569,
                "moment_arm": 0.4,  # m
                "base_max_snap": 1856.4,  # m/s⁴
            }
        }

        return params_db.get(vehicle_type, params_db["medium_large_quad"])

    def _get_dynamic_constraints(self) -> Dict:
        """获取动力学约束"""
        if self.vehicle_type == "fixed_wing":
            return {
                "max_acceleration": 17.3,  # m/s²
                "max_velocity": 42.1,  # m/s
                "min_velocity": 16.2,  # m/s
                "max_climb_rate": 7.2,  # m/s
                "max_turn_rate": np.deg2rad(20.0),  # rad/s
            }
        else:  # 多旋翼类型
            constraints_db = {
                "micro_quad": {
                    "max_acceleration": min(227.1, 100.0),  # m/s²
                    "max_velocity": 32.4,  # m/s
                    "max_vertical_velocity": 11.4,  # m/s
                    "max_tilt_angle": np.deg2rad(60),
                    "max_angular_velocity": np.deg2rad(600),
                },
                "medium_large_quad": {
                    "max_acceleration": 38.7,  # m/s²
                    "max_velocity": 35.1,  # m/s
                    "max_vertical_velocity": 17.3,  # m/s
                    "max_tilt_angle": np.deg2rad(50),
                    "max_angular_velocity": np.deg2rad(140),
                },
                "heavy_multirotor": {
                    "max_acceleration": 47.6,  # m/s²
                    "max_velocity": 24.2,  # m/s
                    "max_vertical_velocity": 7.2,  # m/s
                    "max_tilt_angle": np.deg2rad(45),
                    "max_angular_velocity": np.deg2rad(173.9),
                }
            }
            return constraints_db.get(self.vehicle_type, constraints_db["medium_large_quad"])

    def _get_max_snap(self) -> float:
        """获取最大允许snap值"""
        snap_db = {
            "micro_quad": 200.0,  # m/s^4
            "medium_large_quad": 760.4,  # m/s^4
            "fixed_wing": 10.1,  # m/s^4
            "heavy_multirotor": 1856.4  # m/s^4
        }
        return snap_db.get(self.vehicle_type, 30.0)

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化预测器"""
        try:
            self.position_history.clear()
            self.velocity_history.clear()
            self.acceleration_history.clear()

            # 初始化状态
            initial_pos = initial_state[:3]
            self.position_history.append(initial_pos.copy())
            self.velocity_history.append(np.zeros(3))
            self.acceleration_history.append(np.zeros(3))

            # 重置估计器
            self.mass_estimator.reset(self.params['nominal_mass'])
            self.drag_estimator.reset()

            # 重置统计
            self.prediction_count = 0
            self.constraint_violations = 0
            self.optimization_failures = 0

            self.initialized = True
            return True

        except Exception as e:
            print(f"微分平坦性预测器初始化失败: {e}")
            return False

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """基于微分平坦性的轨迹预测"""
        self.prediction_count += 1
        current_pos = measurement[:3]

        if not self.initialized:
            return current_pos

        try:
            # 更新状态历史
            self.position_history.append(current_pos.copy())

            # 计算速度和加速度
            if len(self.position_history) >= 2:
                velocity = self._estimate_velocity()
                self.velocity_history.append(velocity.copy())

                if len(self.velocity_history) >= 2:
                    acceleration = self._estimate_acceleration()
                    self.acceleration_history.append(acceleration.copy())

            # 执行微分平坦性预测
            if len(self.position_history) >= 5:
                predicted_pos = self._differential_flatness_prediction()

                # 验证动力学可行性
                if self._validate_dynamics(predicted_pos):
                    return predicted_pos
                else:
                    self.constraint_violations += 1
                    return self._constrained_prediction()
            else:
                # 数据不足时使用简单外推
                return self._simple_extrapolation()

        except Exception as e:
            self.optimization_failures += 1
            if self.prediction_count <= 3:
                print(f"微分平坦性预测失败: {e}")
            return self._fallback_prediction(current_pos)

    def _estimate_velocity(self) -> np.ndarray:
        """估计当前速度"""
        if len(self.position_history) >= 3:
            # 使用3点数值微分
            positions = np.array(list(self.position_history)[-3:])
            velocity = (-3 * positions[0] + 4 * positions[1] - positions[2]) / (2 * self.dt)
        else:
            positions = np.array(list(self.position_history)[-2:])
            velocity = (positions[1] - positions[0]) / self.dt

        return velocity

    def _estimate_acceleration(self) -> np.ndarray:
        """估计当前加速度"""
        if len(self.velocity_history) >= 3:
            velocities = np.array(list(self.velocity_history)[-3:])
            acceleration = (-3 * velocities[0] + 4 * velocities[1] - velocities[2]) / (2 * self.dt)
        else:
            velocities = np.array(list(self.velocity_history)[-2:])
            acceleration = (velocities[1] - velocities[0]) / self.dt

        return acceleration

    def _differential_flatness_prediction(self) -> np.ndarray:
        """核心：基于微分平坦性的轨迹预测"""
        # 获取历史轨迹点
        positions = np.array(list(self.position_history)[-8:])  # 使用最近8个点
        t_history = np.arange(len(positions)) * (-self.dt)  # 时间向量（负数表示过去）

        # 为每个坐标轴拟合多项式
        predicted_pos = np.zeros(3)

        for axis in range(3):
            # 拟合7阶多项式（确保snap连续性）
            try:
                coeffs = np.polyfit(t_history, positions[:, axis], min(self.polynomial_order, len(positions) - 1))
                poly = np.poly1d(coeffs)

                # 预测下一时刻 (t = dt)
                predicted_pos[axis] = poly(self.dt)

                # 检查snap（四阶导数）约束
                snap = self._compute_polynomial_snap(coeffs, self.dt)
                if abs(snap) > self._get_max_snap():
                    # Snap过大，使用约束优化
                    predicted_pos[axis] = self._constrained_polynomial_prediction(
                        t_history, positions[:, axis], axis
                    )

            except (np.linalg.LinAlgError, np.RankWarning):
                # 多项式拟合失败，降阶重试
                try:
                    coeffs = np.polyfit(t_history, positions[:, axis], 3)
                    poly = np.poly1d(coeffs)
                    predicted_pos[axis] = poly(self.dt)
                except:
                    # 再次失败，使用线性外推
                    predicted_pos[axis] = positions[-1, axis] + (positions[-1, axis] - positions[-2, axis])

        return predicted_pos

    def _compute_polynomial_snap(self, coeffs: np.ndarray, t: float) -> float:
        """计算多项式的snap（四阶导数）"""
        if len(coeffs) < 5:
            return 0.0

        # 四阶导数的系数
        snap_coeffs = []
        for i in range(len(coeffs) - 4):
            coeff = coeffs[i]
            for j in range(4):
                coeff *= (len(coeffs) - 1 - i - j)
            snap_coeffs.append(coeff)

        if len(snap_coeffs) == 0:
            return 0.0

        # 在时刻t计算snap
        snap = 0.0
        for i, coeff in enumerate(snap_coeffs):
            snap += coeff * (t ** (len(snap_coeffs) - 1 - i))

        return snap

    def _constrained_polynomial_prediction(self, t_history: np.ndarray,
                                           position_history: np.ndarray, axis: int) -> float:
        """约束优化的多项式预测"""
        try:
            # 简化的约束预测：使用低阶多项式
            coeffs = np.polyfit(t_history, position_history, 3)
            poly = np.poly1d(coeffs)
            return poly(self.dt)
        except:
            # 异常情况，使用线性外推
            return position_history[-1] + (position_history[-1] - position_history[-2])

    def _validate_dynamics(self, predicted_pos: np.ndarray) -> bool:
        """验证预测结果的动力学可行性"""
        current_pos = np.array(list(self.position_history)[-1])

        # 检查速度约束
        predicted_velocity = (predicted_pos - current_pos) / self.dt
        velocity_magnitude = np.linalg.norm(predicted_velocity)

        if velocity_magnitude > self.constraints['max_velocity']:
            return False

        # 对于固定翼，检查最小速度
        if self.vehicle_type == "fixed_wing":
            if velocity_magnitude < self.constraints['min_velocity']:
                return False

        # 检查加速度约束
        if len(self.velocity_history) > 0:
            current_velocity = self.velocity_history[-1]
            predicted_acceleration = (predicted_velocity - current_velocity) / self.dt
            accel_magnitude = np.linalg.norm(predicted_acceleration)

            if accel_magnitude > self.constraints['max_acceleration']:
                return False

        return True

    def _constrained_prediction(self) -> np.ndarray:
        """约束优化的预测"""
        current_pos = np.array(list(self.position_history)[-1])
        current_vel = self.velocity_history[-1] if self.velocity_history else np.zeros(3)

        # 简单的约束处理：限制速度
        simple_pred = self._simple_extrapolation()
        pred_velocity = (simple_pred - current_pos) / self.dt

        # 限制速度
        velocity_magnitude = np.linalg.norm(pred_velocity)
        if velocity_magnitude > self.constraints['max_velocity']:
            pred_velocity = pred_velocity * (self.constraints['max_velocity'] / velocity_magnitude)

        return current_pos + pred_velocity * self.dt

    def _simple_extrapolation(self) -> np.ndarray:
        """简单线性外推"""
        if len(self.position_history) >= 2:
            positions = np.array(list(self.position_history)[-2:])
            velocity = (positions[1] - positions[0]) / self.dt

            # 速度限制
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > self.constraints['max_velocity']:
                velocity = velocity * (self.constraints['max_velocity'] / velocity_magnitude)

            return positions[1] + velocity * self.dt
        else:
            return np.array(list(self.position_history)[-1])

    def _fallback_prediction(self, current_pos: np.ndarray) -> np.ndarray:
        """应急预测方案"""
        if len(self.position_history) >= 2:
            prev_pos = np.array(list(self.position_history)[-2])
            safe_velocity = (current_pos - prev_pos) / self.dt

            # 保守的速度限制
            max_safe_velocity = self.constraints['max_velocity'] * 0.5
            velocity_magnitude = np.linalg.norm(safe_velocity)
            if velocity_magnitude > max_safe_velocity:
                safe_velocity = safe_velocity * (max_safe_velocity / velocity_magnitude)

            return current_pos + safe_velocity * self.dt
        else:
            return current_pos

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        success_rate = ((self.prediction_count - self.optimization_failures) /
                        max(self.prediction_count, 1)) * 100

        return {
            'method': 'Differential Flatness Physics',
            'vehicle_type': self.vehicle_type,
            'predictions': self.prediction_count,
            'success_rate': f"{success_rate:.1f}%",
            'constraint_violations': self.constraint_violations,
            'optimization_failures': self.optimization_failures,
            'current_mass_estimate': f"{self.mass_estimator.get_current_estimate():.2f}kg",
            'polynomial_order': self.polynomial_order
        }


class AdaptiveMassEstimator:
    """自适应质量估计器"""

    def __init__(self, initial_mass: float):
        self.initial_mass = initial_mass
        self.current_estimate = initial_mass
        self.measurements = deque(maxlen=20)

    def reset(self, mass: float):
        self.current_estimate = mass
        self.measurements.clear()

    def get_current_estimate(self) -> float:
        return self.current_estimate


class DragCoefficientEstimator:
    """阻力系数估计器"""

    def __init__(self):
        self.drag_estimates = deque(maxlen=15)
        self.current_drag = 0.02

    def reset(self):
        self.drag_estimates.clear()
        self.current_drag = 0.02

    def get_current_drag(self) -> float:
        return self.current_drag


# =========================
# VECTOR相关类定义
# =========================

class VectorGRU(nn.Module):
    """VECTOR GRU模型"""

    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2,
                 output_dim=3, dropout=0.5, sequence_length=20):
        super(VectorGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # GRU层处理速度序列
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # 输出层：从GRU隐状态到位置预测
        self.position_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        # 速度预测分支
        self.velocity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Tanh()
        )

    def forward(self, velocity_sequence, last_position=None):
        """前向传播"""
        # GRU处理速度序列
        gru_out, hidden = self.gru(velocity_sequence)

        # 使用最后时刻的隐状态
        last_hidden = gru_out[:, -1, :]

        # 预测下一步速度
        predicted_velocity = self.velocity_predictor(last_hidden)

        # 如果提供了最后位置，通过积分得到下一步位置
        if last_position is not None:
            predicted_position = last_position + predicted_velocity * 0.1  # dt=0.1
        else:
            position_delta = self.position_predictor(last_hidden)
            predicted_position = position_delta

        return predicted_position, predicted_velocity


class VectorPredictor:
    """VECTOR预测器"""

    def __init__(self, model_path=None, sequence_length=20, dt=0.1, device='cpu'):
        self.sequence_length = sequence_length
        self.dt = dt
        self.device = device

        # 速度历史缓冲区
        self.velocity_buffer = deque(maxlen=sequence_length)
        self.position_history = deque(maxlen=10)

        # 模型和标准化器
        self.model = None
        self.position_scaler = None
        self.velocity_scaler = None
        self.model_loaded = False

        # 统计信息
        self.total_predictions = 0
        self.successful_predictions = 0

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        """加载训练好的VECTOR模型和标准化器"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # 创建模型实例
            self.model = VectorGRU(
                input_dim=3,
                hidden_dim=checkpoint.get('hidden_dim', 64),
                num_layers=checkpoint.get('num_layers', 2),
                output_dim=3,
                dropout=0.0,  # 推理时不使用dropout
                sequence_length=self.sequence_length
            )

            # 加载权重
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            # 加载标准化器
            if 'position_scaler' in checkpoint and 'velocity_scaler' in checkpoint:
                self.position_scaler = checkpoint['position_scaler']
                self.velocity_scaler = checkpoint['velocity_scaler']
                print("VECTOR标准化器加载成功")
            else:
                print("警告: VECTOR检查点中没有标准化器")
                self.position_scaler = None
                self.velocity_scaler = None

            self.model.eval()
            self.model_loaded = True
            print("VECTOR模型加载成功")

        except Exception as e:
            print(f"VECTOR模型加载失败: {e}")
            self.model_loaded = False

    def _compute_velocity(self, pos_history):
        """计算当前速度"""
        if len(pos_history) < 2:
            return np.zeros(3)

        recent_positions = np.array(list(pos_history)[-2:])
        velocity = (recent_positions[-1] - recent_positions[-2]) / self.dt
        return velocity

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化预测器"""
        try:
            initial_pos = initial_state[:3]

            # 清空缓冲区
            self.velocity_buffer.clear()
            self.position_history.clear()

            # 填充初始历史
            for _ in range(self.sequence_length):
                self.velocity_buffer.append(np.zeros(3))

            self.position_history.append(initial_pos.copy())

            # 重置统计
            self.total_predictions = 0
            self.successful_predictions = 0

            return True

        except Exception as e:
            print(f"VECTOR预测器初始化失败: {e}")
            return False

    def predict_and_update(self, measurement: np.ndarray, **kwargs) -> np.ndarray:
        """单步预测方法"""
        self.total_predictions += 1
        current_pos = np.array(measurement[:3])

        # 更新位置历史
        self.position_history.append(current_pos.copy())

        # 计算当前速度并更新速度缓冲区
        current_velocity = self._compute_velocity(self.position_history)
        self.velocity_buffer.append(current_velocity.copy())

        if not self.model_loaded or self.model is None:
            # 回退到简单的物理预测
            return self._physics_fallback(current_pos, current_velocity)

        try:
            # 使用VECTOR模型预测
            with torch.no_grad():
                # 准备输入数据并标准化
                vel_sequence_raw = np.array(list(self.velocity_buffer))
                current_pos_raw = current_pos.copy()

                # 标准化输入
                if self.velocity_scaler is not None:
                    vel_sequence_scaled = self.velocity_scaler.transform(vel_sequence_raw)
                else:
                    vel_sequence_scaled = vel_sequence_raw

                if self.position_scaler is not None:
                    current_pos_scaled = self.position_scaler.transform(current_pos_raw.reshape(1, -1)).squeeze()
                else:
                    current_pos_scaled = current_pos_raw

                # 转换为张量
                vel_sequence = torch.FloatTensor(vel_sequence_scaled).unsqueeze(0)  # (1, seq_len, 3)
                last_pos = torch.FloatTensor(current_pos_scaled).unsqueeze(0)  # (1, 3)

                # 模型预测（在标准化空间中）
                predicted_pos_scaled, predicted_vel_scaled = self.model(vel_sequence, last_pos)

                # 转换为numpy
                predicted_pos_scaled = predicted_pos_scaled.squeeze(0).numpy()

                # 反标准化到原始空间
                if self.position_scaler is not None:
                    predicted_position = self.position_scaler.inverse_transform(
                        predicted_pos_scaled.reshape(1, -1)
                    ).squeeze()
                else:
                    predicted_position = predicted_pos_scaled

                self.successful_predictions += 1

                # 调试信息（仅前几步）
                if self.total_predictions <= 3:
                    print(f"VECTOR预测步骤 {self.total_predictions}: 成功")

                return predicted_position

        except Exception as e:
            if self.total_predictions <= 3:
                print(f"VECTOR预测失败，使用物理回退: {e}")

            return self._physics_fallback(current_pos, current_velocity)

    def _physics_fallback(self, current_pos, current_velocity):
        """物理模型回退方案"""
        # 限制速度
        max_velocity = 25.0  # m/s
        vel_norm = np.linalg.norm(current_velocity)
        if vel_norm > max_velocity:
            current_velocity = current_velocity * (max_velocity / vel_norm)

        # 简单的匀速预测
        predicted_pos = current_pos + current_velocity * self.dt
        return predicted_pos


# =========================
# 评估指标
# =========================

def calculate_trajectory_metrics(predictions, targets, threshold=2.0):
    """计算轨迹预测的标准评估指标"""
    predictions = np.array(predictions)
    targets = np.array(targets)

    if predictions.ndim == 2:  # 单步预测
        errors = np.linalg.norm(predictions - targets, axis=1)
        ade = np.mean(errors)
        fde = ade  # 单步情况下ADE=FDE
        success_rate = np.mean(errors < threshold) * 100

    else:  # 多步预测
        all_errors = np.linalg.norm(predictions - targets, axis=2)  # [N, T]
        ade = np.mean(all_errors)
        final_errors = all_errors[:, -1]  # [N]
        fde = np.mean(final_errors)
        success_rate = np.mean(final_errors < threshold) * 100

    return {
        'ADE': ade,
        'FDE': fde,
        'Success_Rate': success_rate
    }


# =========================
# 方法创建函数
# =========================

def create_integrated_baseline_methods_complete(qr_model_path=None, vector_model_path=None, dt=0.1, seq_len=20,
                                                device='cpu', **kwargs):
    """创建集成的7种基线方法（核心5种 + VECTOR + 物理方法）"""

    print("=" * 60)
    print("创建集成的7种高质量基线方法对比实验（Q/R缩放专用版）")
    print("=" * 60)
    print(f"参数配置:")
    print(f"  Q/R缩放模型: {qr_model_path}")
    print(f"  VECTOR模型: {vector_model_path}")
    print(f"  时间间隔dt: {dt}")
    print(f"  序列长度: {seq_len}")
    print(f"  计算设备: {device}")
    print("=" * 60)

    methods = {}

    # 第一部分：创建核心5种方法
    print("\n【第一部分：核心5种基线方法】")
    try:
        core_methods = create_baseline_methods(
            qr_model_path=qr_model_path,
            dt=dt,
            seq_len=seq_len,
            device=device,
            **kwargs
        )
        methods.update(core_methods)
        print(f"✅ 成功创建 {len(core_methods)} 种核心方法: {list(core_methods.keys())}")
    except Exception as e:
        print(f"❌ 核心方法创建失败: {e}")

    # 第二部分：创建VECTOR方法
    print("\n【第二部分：VECTOR方法】")

    if vector_model_path and os.path.exists(vector_model_path):
        try:
            vector_predictor = VectorPredictor(
                model_path=vector_model_path,
                sequence_length=20,
                dt=dt,
                device=device
            )
            methods['VECTOR'] = MethodWrapper('VECTOR', vector_predictor)

            # 添加状态检查
            if vector_predictor.model_loaded:
                print("✅ VECTOR方法创建成功，模型正确加载")
                print(f"  位置标准化器: {'已加载' if vector_predictor.position_scaler is not None else '未加载'}")
                print(f"  速度标准化器: {'已加载' if vector_predictor.velocity_scaler is not None else '未加载'}")
            else:
                print("⚠️ VECTOR方法创建成功，但模型加载失败，将使用物理回退")

        except Exception as e:
            print(f"❌ 创建VECTOR失败: {e}")
    else:
        print(f"❌ VECTOR模型文件不存在: {vector_model_path}")

    # 第三部分：尝试创建优化的Trajectron++（如果可用）
    print("\n【第三部分：优化Trajectron++方法】")

    if NEW_METHODS_AVAILABLE:
        try:
            optimized_trajectron = create_optimized_trajectron_plus_plus(
                dt=dt,
                seq_len=30,  # 最佳参数
                device=device,
                hidden_dim=64,  # 最佳参数
                num_layers=2,  # 最佳参数
                dropout=0.1  # 最佳参数
            )

            methods['Trajectron++'] = MethodWrapper('Trajectron++', optimized_trajectron)
            print("✅ Trajectron++创建成功")

        except Exception as e:
            print(f"❌ 创建优化Trajectron++失败: {e}")
    else:
        print("❌ Trajectron++方法模块不可用，跳过创建")

    # 第四部分：创建微分平坦性物理预测器
    print("\n【第四部分：微分平坦性物理预测器】")

    # 根据数据集推断载具类型（在实际评估时会动态设置）
    try:
        physics_predictor = DifferentialFlatnessTrajectoryPredictor(dt=dt, vehicle_type="medium_large_quad")
        methods['Physics-DF'] = MethodWrapper('Physics-DF', physics_predictor)
        print("✅ 微分平坦性物理预测器创建成功")
    except Exception as e:
        print(f"❌ 创建微分平坦性物理预测器失败: {e}")

    print(f"\n总共成功创建 {len(methods)} 种方法:")
    for i, method_name in enumerate(methods.keys(), 1):
        method_type = "🧠" if any(
            x in method_name for x in ['Q/R', 'VECTOR', 'Transformer', 'Trajectron']) else "⚙️"
        print(f"  {i}. {method_type} {method_name}")

    return methods


# =========================
# 数据处理
# =========================

def extract_sequence_features(pos_window, vel_window, dt):
    """提取序列特征 - 训练代码原版（16维）"""
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


def generate_continuous_samples(positions, dt):
    """生成连续样本 - 更接近训练环境"""
    continuous_samples = []
    window_size = 20

    for i in range(window_size, len(positions) - 1):
        pos_window = positions[i - window_size:i]
        vel_window = np.diff(pos_window, axis=0) / dt

        features = extract_sequence_features(pos_window, vel_window, dt)
        target_pos = positions[i + 1]

        continuous_samples.append({
            'features': features,
            'target_pos': target_pos,
            'current_pos': positions[i],
            'dt': dt,
            'step_index': i
        })

    return continuous_samples


def infer_vehicle_type_from_filename(filename):
    """从文件名推断载具类型"""
    filename_lower = filename.lower()
    if "enhanced_drone_data" in filename_lower or "drone_dt01" in filename_lower:
        return "medium_large_quad"
    elif "drone_flight_data" in filename_lower:
        return "micro_quad"
    elif "fixed_wing" in filename_lower:
        return "fixed_wing"
    elif "heavy_multirotor" in filename_lower:
        return "heavy_multirotor"
    else:
        return "medium_large_quad"


def load_dataset_continuous_style(file_path, max_points=700):
    """按连续预测的方式加载数据集"""
    print(f"加载数据集: {file_path}")

    try:
        df = pd.read_csv(file_path)

        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if not time_cols:
            print(f"未找到时间列")
            return None

        pos_cols = []
        for prefix in ['x', 'y', 'z']:
            for col in df.columns:
                if col.lower() == prefix or col.lower() == f'{prefix}_true' or col.lower() == f'true_{prefix}':
                    pos_cols.append(col)
                    break

        if len(pos_cols) < 3:
            print(f"位置列不完整: {pos_cols}")
            return None

        positions_full = df[pos_cols].values[:, :3]
        print(f"  原始数据: {len(positions_full)} 帧")

        positions = positions_full[:max_points]
        print(f"  限制后数据: {len(positions)} 帧 (前{max_points}帧)")

        if len(positions) < 50:
            print(f"数据点太少: {len(positions)}")
            return None

        if np.any(np.isnan(positions)):
            print(f"  处理NaN值...")
            valid_mask = ~np.any(np.isnan(positions), axis=1)
            positions = positions[valid_mask]

        if len(positions) < 50:
            print(f"清理后数据点太少: {len(positions)}")
            return None

        time_data = df[time_cols[0]].values[:len(positions)]
        if len(time_data) > 1:
            dt_values = np.diff(time_data)
            dt = np.median(dt_values)
        else:
            dt = 0.1

        print(f"  时间间隔: {dt:.3f}s")

        # 使用相同的80/20划分
        split_point = int(len(positions) * 0.8)
        train_positions = positions[:split_point]
        test_positions = positions[split_point:]

        print(f"  训练数据: {len(train_positions)} 帧")
        print(f"  测试数据: {len(test_positions)} 帧")

        # 生成连续样本
        test_samples = generate_continuous_samples(test_positions, dt)
        print(f"  连续测试样本: {len(test_samples)} 个")

        # 推断载具类型
        vehicle_type = infer_vehicle_type_from_filename(os.path.basename(file_path))
        print(f"  载具类型: {vehicle_type}")

        return {
            'test_samples': test_samples,
            'test_positions': test_positions,
            'dt': dt,
            'filename': os.path.basename(file_path),
            'vehicle_type': vehicle_type,
            'split_info': {
                'total_frames': len(positions),
                'train_frames': len(train_positions),
                'test_frames': len(test_positions),
                'test_samples': len(test_samples)
            }
        }

    except Exception as e:
        print(f"加载数据集失败: {e}")
        return None


# =========================
# 评估器
# =========================

class ImprovedBaselineMethodsEvaluator:
    """改进的基线方法评估器 - 支持载具类型动态配置"""

    def __init__(self, qr_model_path=None, vector_model_path=None, device='cpu'):
        self.qr_model_path = qr_model_path
        self.vector_model_path = vector_model_path
        self.device = device
        self.methods = {}
        self.method_states = {}

    def initialize_methods(self, dt=0.1, vehicle_type="medium_large_quad"):
        """初始化所有基线方法（7种）"""
        print(f"初始化7种集成基线方法...")
        print(f"载具类型: {vehicle_type}")

        # 使用新的集成方法创建函数
        self.methods = create_integrated_baseline_methods_complete(
            qr_model_path=self.qr_model_path,
            vector_model_path=self.vector_model_path,
            dt=dt,
            device=self.device
        )

        # 为微分平坦性物理预测器重新设置载具类型
        if 'Physics-DF' in self.methods:
            try:
                physics_predictor = DifferentialFlatnessTrajectoryPredictor(dt=dt, vehicle_type=vehicle_type)
                self.methods['Physics-DF'] = MethodWrapper('Physics-DF', physics_predictor)
                print(f"✅ 微分平坦性物理预测器已配置为 {vehicle_type}")
            except Exception as e:
                print(f"❌ 重新配置物理预测器失败: {e}")

        print(f"成功初始化 {len(self.methods)} 种方法: {list(self.methods.keys())}")

        # 检查Q/R缩放UKF的状态
        if 'Q/R缩放UKF' in self.methods:
            self._check_qr_scaling_model_status()

        # 检查VECTOR的状态
        if 'VECTOR' in self.methods:
            self._check_vector_model_status()

        return len(self.methods) > 0

    def _check_qr_scaling_model_status(self):
        """检查Q/R缩放模型的加载状态"""
        print("\n=== Q/R缩放模型状态检查 ===")
        try:
            qr_method = self.methods['Q/R缩放UKF'].method

            print(f"模型加载状态: {qr_method.model_loaded}")

            if hasattr(qr_method, 'qr_model') and qr_method.qr_model is not None:
                print("✓ Q/R缩放神经网络模型已加载")
                test_input = torch.randn(1, 16)
                with torch.no_grad():
                    qr_scales, confidence, vehicle_probs, vehicle_logits = qr_method.qr_model(test_input)
                print(f"模型测试输出: Q/R缩放={qr_scales.flatten().numpy()}")
            else:
                print("✗ Q/R缩放神经网络模型未加载")

            print("=== 状态检查完成 ===\n")

        except Exception as e:
            print(f"状态检查失败: {e}")

    def _check_vector_model_status(self):
        """检查VECTOR模型的加载状态"""
        print("\n=== VECTOR模型状态检查 ===")
        try:
            vector_method = self.methods['VECTOR'].method

            print(f"模型加载状态: {vector_method.model_loaded}")

            if vector_method.model_loaded:
                print("✓ VECTOR神经网络模型已加载")
                print(f"  位置标准化器: {'已加载' if vector_method.position_scaler is not None else '未加载'}")
                print(f"  速度标准化器: {'已加载' if vector_method.velocity_scaler is not None else '未加载'}")

                # 测试模型推理
                try:
                    test_vel_seq = torch.randn(1, 20, 3)
                    test_pos = torch.randn(1, 3)
                    with torch.no_grad():
                        pred_pos, pred_vel = vector_method.model(test_vel_seq, test_pos)
                    print(f"  模型推理测试: 成功")
                except Exception as e:
                    print(f"  模型推理测试: 失败 - {e}")
            else:
                print("✗ VECTOR神经网络模型未加载，将使用物理回退")

            print("=== 状态检查完成 ===\n")

        except Exception as e:
            print(f"状态检查失败: {e}")

    def _evaluate_single_step_continuous_with_metrics(self, sample, method_wrapper, verbose=False):
        """单步连续评估 - 增加标准轨迹预测指标和精确时间测量"""
        try:
            current_pos = sample['current_pos']
            target_pos = sample['target_pos']

            # 使用更精确的时间测量
            step_start_time = time.perf_counter()

            # 3步连续预测过程
            max_steps = 3
            current_pred = current_pos.copy()
            trajectory_predictions = [current_pos.copy()]  # 记录预测轨迹
            trajectory_targets = [current_pos.copy()]  # 记录目标轨迹

            for step in range(max_steps):
                progress = (step + 1) / max_steps
                intermediate_target = current_pos + progress * (target_pos - current_pos)
                trajectory_targets.append(intermediate_target.copy())

                # 添加0.25m噪声
                noise = np.random.normal(0, 0.25, 3)
                noisy_obs = intermediate_target + noise

                pred_result = method_wrapper.predict_and_update(noisy_obs)
                current_pred = pred_result[:3] if len(pred_result) > 3 else pred_result
                trajectory_predictions.append(current_pred.copy())

            # 计算单步时间
            step_time = (time.perf_counter() - step_start_time) * 1000  # 转换为毫秒

            # 计算标准轨迹预测指标
            pred_array = np.array(trajectory_predictions[1:])  # 去掉初始位置
            target_array = np.array(trajectory_targets[1:])  # 去掉初始位置

            metrics = calculate_trajectory_metrics(
                pred_array.reshape(1, -1, 3),
                target_array.reshape(1, -1, 3)
            )

            # 添加时间信息
            metrics['Processing_Time'] = step_time

            if verbose:
                print(f"    ADE: {metrics['ADE']:.6f}m")
                print(f"    FDE: {metrics['FDE']:.6f}m")
                print(f"    Time: {step_time:.3f}ms")
                print(f"    Success: {'Yes' if metrics['FDE'] < 2.0 else 'No'}")

            return metrics

        except Exception as e:
            if verbose:
                print(f"    预测失败: {e}")
            return {'ADE': float('inf'), 'FDE': float('inf'), 'Success_Rate': 0.0, 'Processing_Time': float('inf')}

    def evaluate_continuous_trajectory_with_metrics(self, dataset, max_samples_per_method=100):
        """连续轨迹评估 - 输出标准轨迹预测指标和精确时间统计"""
        print(f"\n连续轨迹评估: {dataset['filename']}")
        print(f"数据信息: {dataset['split_info']}")
        print(f"载具类型: {dataset['vehicle_type']}")

        test_samples = dataset['test_samples']
        test_positions = dataset['test_positions']

        if max_samples_per_method and max_samples_per_method > 0:
            eval_samples = min(len(test_samples), max_samples_per_method)
            test_samples = test_samples[:eval_samples]
            print(f"评估 {eval_samples} 个连续样本 (限制为{max_samples_per_method})...")
        else:
            eval_samples = len(test_samples)
            print(f"评估全部 {eval_samples} 个连续样本...")

        results = {}

        for method_name, method_wrapper in self.methods.items():
            print(f"\n--- 连续评估 {method_name} ---")

            # 初始化
            initial_pos = test_positions[20]
            initial_state = np.concatenate([initial_pos, [0, 0, 0]])

            if not method_wrapper.initialize(initial_state):
                print(f"  {method_name} 初始化失败")
                continue

            print(f"  {method_name} 初始化成功")

            # 收集所有样本的指标
            all_ade = []
            all_fde = []
            all_success = []
            all_processing_times = []

            for i, sample in enumerate(test_samples):
                np.random.seed(42 + i)  # 确保可重复性

                metrics = self._evaluate_single_step_continuous_with_metrics(
                    sample, method_wrapper, verbose=(i < 3)
                )

                if metrics['ADE'] != float('inf'):
                    all_ade.append(metrics['ADE'])
                    all_fde.append(metrics['FDE'])
                    all_success.append(1 if metrics['FDE'] < 2.0 else 0)
                    all_processing_times.append(metrics['Processing_Time'])

            # 计算平均指标
            if len(all_ade) > 0:
                avg_ade = np.mean(all_ade)
                avg_fde = np.mean(all_fde)
                success_rate = np.mean(all_success) * 100
                avg_processing_time = np.mean(all_processing_times)

                results[method_name] = {
                    'ADE': avg_ade,
                    'FDE': avg_fde,
                    'Success_Rate': success_rate,
                    'Processing_Time': avg_processing_time,
                    'samples_evaluated': len(all_ade),
                    'total_samples': len(test_samples)
                }

                print(
                    f"  {method_name}: ADE={avg_ade:.6f}m, FDE={avg_fde:.6f}m, 成功率={success_rate:.1f}%, 处理时间={avg_processing_time:.3f}ms")
            else:
                print(f"  {method_name}: 所有预测都失败了!")
                results[method_name] = {
                    'ADE': float('inf'), 'FDE': float('inf'), 'Success_Rate': 0.0,
                    'Processing_Time': float('inf'),
                    'samples_evaluated': 0, 'total_samples': len(test_samples)
                }

        return results


def print_final_results_with_metrics(all_results):
    """输出标准轨迹预测指标的最终结果"""
    print(f"\n{'=' * 60}")
    print("轨迹预测方法对比结果（标准指标）")
    print("=" * 60)

    # 汇总所有方法的平均指标
    method_metrics = {}
    for result in all_results:
        for method_name, metrics in result.items():
            if 'dataset_info' in method_name:
                continue

            if method_name not in method_metrics:
                method_metrics[method_name] = {
                    'ADE': [], 'FDE': [], 'Success_Rate': [], 'Processing_Time': []
                }

            if metrics['ADE'] != float('inf'):
                method_metrics[method_name]['ADE'].append(metrics['ADE'])
                method_metrics[method_name]['FDE'].append(metrics['FDE'])
                method_metrics[method_name]['Success_Rate'].append(metrics['Success_Rate'])
                method_metrics[method_name]['Processing_Time'].append(metrics['Processing_Time'])

    # 计算平均值并排序
    final_results = []
    for method_name, metrics in method_metrics.items():
        if len(metrics['ADE']) > 0:
            avg_ade = np.mean(metrics['ADE'])
            avg_fde = np.mean(metrics['FDE'])
            avg_success = np.mean(metrics['Success_Rate'])
            avg_time = np.mean(metrics['Processing_Time'])
            final_results.append((method_name, avg_ade, avg_fde, avg_success, avg_time))

    # 按ADE排序
    final_results.sort(key=lambda x: x[1])

    print("\n总体性能排名:")
    for rank, (method_name, ade, fde, success_rate, proc_time) in enumerate(final_results, 1):
        method_type = "🧠" if any(
            x in method_name for x in ['Q/R', 'VECTOR', 'Transformer', 'Trajectron']) else "⚙️"
        print(
            f"  #{rank} {method_type} {method_name}: ADE={ade:.6f}m, FDE={fde:.6f}m, 成功率={success_rate:.1f}%, 处理时间={proc_time:.3f}ms")

    # 分类分析
    print(f"\n方法分类分析:")
    neural_methods = [r for r in final_results if
                      any(x in r[0] for x in ['Q/R', 'VECTOR', 'Transformer', 'Trajectron'])]
    physics_methods = [r for r in final_results if any(x in r[0] for x in ['UKF', 'Kalman', 'Physics', '物理'])]

    if neural_methods:
        best_neural = min(neural_methods, key=lambda x: x[1])
        print(f"  最佳神经网络方法: {best_neural[0]} (ADE={best_neural[1]:.6f}m)")

    if physics_methods:
        best_physics = min(physics_methods, key=lambda x: x[1])
        print(f"  最佳物理方法: {best_physics[0]} (ADE={best_physics[1]:.6f}m)")

    # 计算效率分析
    print(f"\n计算效率分析:")
    fast_methods = [r for r in final_results if r[4] < 1.0]  # <1ms的方法
    medium_methods = [r for r in final_results if 1.0 <= r[4] < 10.0]  # 1-10ms的方法
    slow_methods = [r for r in final_results if r[4] >= 10.0]  # >10ms的方法

    print(f"  实时方法 (<1ms): {len(fast_methods)} 种")
    print(f"  准实时方法 (1-10ms): {len(medium_methods)} 种")
    print(f"  慢速方法 (>10ms): {len(slow_methods)} 种")

    # 输出论文格式结果
    print(f"\n论文格式结果:")
    for rank, (method_name, ade, fde, success_rate, proc_time) in enumerate(final_results, 1):
        print(f"{method_name}: ADE={ade:.6f}m, FDE={fde:.6f}m, 成功率={success_rate:.1f}%, 时间={proc_time:.3f}ms")

    return final_results


def main():
    """修改后的主函数 - 支持多次运行"""
    parser = argparse.ArgumentParser(description='集成7种基线方法对比实验 - 多次运行统计版')

    # 现有参数...
    parser.add_argument('--qr_model', type=str, default='BEST_qr_scaling.pth',
                        help='Q/R缩放模型路径 (default: BEST_qr_scaling.pth)')
    parser.add_argument('--vector_model', type=str, default='vector_best_model.pth', help='VECTOR模型路径')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--max_points', type=int, default=700, help='每数据集最大帧数')
    parser.add_argument('--data_files', type=str, help='数据文件列表，逗号分隔')
    parser.add_argument('--max_samples', type=int, default=100, help='每方法最大测试样本数')

    # 新增参数
    parser.add_argument('--num_runs', type=int, default=5, help='独立运行次数 (default: 5)')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子基值 (default: 42)')

    args = parser.parse_args()

    print("集成7种高质量基线方法对比实验（多次运行统计版）")
    print("=" * 60)
    print("实验设置:")
    print(f"- 独立运行次数: {args.num_runs}")
    print(f"- 随机种子基值: {args.random_seed}")
    print(f"- 每次运行将使用种子: {[args.random_seed + i for i in range(args.num_runs)]}")
    print("=" * 60)

    # 准备数据集（只需要加载一次）
    if args.data_files:
        data_files = [f.strip() for f in args.data_files.split(',') if f.strip()]
    else:
        data_files = [
            "drone_flight_data.csv",
            "drone_dt01.csv",
            "complex_fixed_wing_trajectory.csv",
            "complex_heavy_multirotor_trajectory.csv"
        ]

    print(f"\n准备数据集...")
    datasets = []
    for data_file in data_files:
        if os.path.exists(data_file):
            dataset = load_dataset_continuous_style(data_file, args.max_points)
            if dataset is not None:
                datasets.append(dataset)
        else:
            print(f"数据文件不存在: {data_file}")

    if len(datasets) == 0:
        print("没有成功加载任何数据集")
        return

    print(f"成功加载 {len(datasets)} 个数据集")

    # 多次运行实验
    all_runs_results = []

    for run_idx in range(args.num_runs):
        print(f"\n{'=' * 60}")
        print(f"开始第 {run_idx + 1}/{args.num_runs} 次运行")
        print("=" * 60)

        # 设置当前运行的随机种子
        current_seed = args.random_seed + run_idx
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(current_seed)

        print(f"当前运行随机种子: {current_seed}")

        # 创建评估器（每次运行都重新创建）
        evaluator = ImprovedBaselineMethodsEvaluator(
            qr_model_path=args.qr_model,
            vector_model_path=args.vector_model,
            device=args.device
        )

        # 当前运行的结果
        current_run_results = []

        # 对每个数据集进行评估
        for dataset_idx, dataset in enumerate(datasets):
            print(f"\n--- Run {run_idx + 1}: 数据集 {dataset_idx + 1}/{len(datasets)} ---")

            # 根据数据集载具类型初始化方法
            vehicle_type = dataset['vehicle_type']
            if not evaluator.initialize_methods(dt=dataset['dt'], vehicle_type=vehicle_type):
                print(f"Run {run_idx + 1}: 方法初始化失败")
                continue

            # 评估当前数据集
            dataset_results = evaluator.evaluate_continuous_trajectory_with_metrics(
                dataset, args.max_samples
            )

            if dataset_results:
                dataset_results['dataset_info'] = {
                    'filename': dataset['filename'],
                    'vehicle_type': dataset['vehicle_type'],
                    'dt': dataset['dt'],
                    'run_index': run_idx
                }
                current_run_results.append(dataset_results)

        # 保存当前运行的结果
        all_runs_results.append(current_run_results)
        print(f"第 {run_idx + 1} 次运行完成")

    # 统计分析和结果输出
    if len(all_runs_results) > 0:
        print(f"\n{'=' * 60}")
        print(f"所有 {args.num_runs} 次运行完成，开始统计分析...")
        print("=" * 60)

        final_stats = print_final_results_with_stats(all_runs_results, args.num_runs)

        # 输出实验总结
        print(f"\n{'=' * 60}")
        print("实验总结:")
        print("=" * 60)
        print(f"✅ 成功完成 {args.num_runs} 次独立运行")
        print(f"✅ 评估了 {len(evaluator.methods) if 'evaluator' in locals() else 'N/A'} 种方法")
        print(f"✅ 测试了 {len(datasets)} 个数据集")
        print(f"✅ 采用均值±标准差报告结果，符合学术标准")
        print(f"✅ 随机种子: {args.random_seed} ~ {args.random_seed + args.num_runs - 1}")

        # 最佳方法统计
        if final_stats:
            best_method = final_stats[0]
            print(f"\n🏆 最佳方法: {best_method[0]}")
            print(f"   ADE: {best_method[1]:.6f}±{best_method[2]:.6f}m")
            print(f"   FDE: {best_method[3]:.6f}±{best_method[4]:.6f}m")
    else:
        print("没有获得有效的实验结果")


if __name__ == "__main__":
    main()