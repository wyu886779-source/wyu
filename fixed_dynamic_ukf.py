# === fixed_dynamic_ukf.py (4×6 vehicle-type × flight-mode library, improved classifier) ===
# 运行时UKF：依据【载具类型 + 在线判别飞行模式】切换 Q/R。
# - 若同目录存在 optimized_ukf_params_4x6.json/.npz，会优先加载其中的 Q/R；
# - 否则按载具类型使用稳健默认值（可解释、可复现）。

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# -----------------------------
# 配置：载具/模式枚举
# -----------------------------
VEHICLE_TYPES = ["micro_quad", "medium_large_quad", "fixed_wing", "heavy_multirotor"]
MOTION_MODES = ["hovering", "cruising", "maneuvering", "climbing", "landing", "aggressive"]

# 按载具类型的速度阈值（m/s），结合你给的统计值定制
SPEED_RULES = {
    # 小四旋翼：最高速度相对较低
    "micro_quad": {
        "v_hover": 1.0, "v_cruise_max": 6.0, "v_maneuver_max": 12.0, "v_aggr_min": 12.0,
        "vz_climb": 2.5, "vz_desc": -2.5
    },
    # 中大型四旋翼（你的均速≈17.7，Max≈30）
    "medium_large_quad": {
        "v_hover": 1.0, "v_cruise_max": 10.0, "v_maneuver_max": 20.0, "v_aggr_min": 20.0,
        "vz_climb": 3.0, "vz_desc": -3.0
    },
    # 固定翼：速度范围高，垂直速度阈值也放宽
    "fixed_wing": {
        "v_hover": 1.0, "v_cruise_max": 25.0, "v_maneuver_max": 40.0, "v_aggr_min": 30.0,
        "vz_climb": 5.0, "vz_desc": -5.0
    },
    # 重型多旋翼：最大速度约 ≤ 20
    "heavy_multirotor": {
        "v_hover": 1.0, "v_cruise_max": 8.0, "v_maneuver_max": 18.0, "v_aggr_min": 18.0,
        "vz_climb": 3.0, "vz_desc": -3.0
    },
}

DEFAULT_PARAM_LIB_PATHS = [
    "optimized_ukf_params_4x6.json",
    "optimized_ukf_params_4x6.npz",
]


# -----------------------------
# 实用函数：从文件名推断载具类型（可手动指定覆盖）
# -----------------------------
def infer_vehicle_type_from_path(path: str) -> str:
    p = (path or "").lower()
    if "enhanced_drone_data" in p:
        return "medium_large_quad"
    if "drone_flight_data" in p:
        return "micro_quad"
    if "complex_fixed_wing_trajectory" in p or "fixed_wing" in p:
        return "fixed_wing"
    if "complex_heavy_multirotor_trajectory" in p or "heavy_multirotor" in p:
        return "heavy_multirotor"
    return "medium_large_quad"


def detect_dt_from_csv(csv_file):
    """从CSV文件检测时间间隔"""
    try:
        df = pd.read_csv(csv_file)
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        if time_cols:
            time_diffs = np.diff(df[time_cols[0]].values)
            median_dt = np.median(time_diffs)
            if 0.08 <= median_dt <= 0.12:
                return 0.1
            elif 0.9 <= median_dt <= 1.1:
                return 1.0
            else:
                return round(median_dt, 3)
        return 0.1
    except:
        return 0.1


# -----------------------------
# 参数库 I/O
# -----------------------------
def load_param_library(paths=None) -> Optional[Dict[str, Dict[str, Dict]]]:
    paths = paths or DEFAULT_PARAM_LIB_PATHS
    for p in paths:
        if os.path.exists(p) and p.endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        if os.path.exists(p) and p.endswith(".npz"):
            npz = np.load(p, allow_pickle=True)
            return npz["params"].item()
    return None


def default_param_for(vehicle_type: str, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """稳健默认 Q/R（6×6、3×3），按载具类型有缩放，按模式有结构差异。"""
    vt_scale = {
        "micro_quad": 1.0,
        "medium_large_quad": 1.2,
        "fixed_wing": 0.8,
        "heavy_multirotor": 1.5,
    }.get(vehicle_type, 1.0)

    if mode in ["climbing", "landing"]:
        q_pos, q_vel, r_obs = 0.15, 0.6, 0.4
        Q = np.diag([q_pos, q_pos, q_pos * 1.5, q_vel, q_vel, q_vel * 2.0]) * vt_scale
        R = np.diag([r_obs, r_obs, r_obs * 1.2]) * vt_scale
    elif mode == "hovering":
        q_pos, q_vel, r_obs = 0.05, 0.2, 0.2
        Q = np.diag([q_pos * 0.5, q_pos * 0.5, q_pos * 0.5, q_vel * 0.3, q_vel * 0.3, q_vel * 0.3]) * vt_scale
        R = np.diag([r_obs * 0.7, r_obs * 0.7, r_obs * 0.7]) * vt_scale
    elif mode == "aggressive":
        q_pos, q_vel, r_obs = 0.4, 2.5, 1.0
        Q = np.diag([q_pos * 1.5, q_pos * 1.5, q_pos * 1.5, q_vel * 2.0, q_vel * 2.0, q_vel * 2.0]) * vt_scale
        R = np.diag([r_obs * 1.5, r_obs * 1.5, r_obs * 1.5]) * vt_scale
    elif mode == "cruising":
        q_pos, q_vel, r_obs = 0.1, 0.5, 0.3
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * vt_scale
        R = np.diag([r_obs, r_obs, r_obs]) * vt_scale
    else:  # maneuvering
        q_pos, q_vel, r_obs = 0.2, 1.2, 0.6
        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * vt_scale
        R = np.diag([r_obs, r_obs, r_obs]) * vt_scale
    return Q, R


def select_QR(vehicle_type: str, mode: str, param_lib: Optional[Dict[str, Dict[str, Dict]]]) -> Tuple[
    np.ndarray, np.ndarray]:
    vehicle_type = vehicle_type if vehicle_type in VEHICLE_TYPES else "medium_large_quad"
    mode = mode if mode in MOTION_MODES else "cruising"
    if param_lib is None:
        return default_param_for(vehicle_type, mode)
    try:
        entry = param_lib[vehicle_type][mode]
        Q = np.array(entry["Q"], dtype=float)
        R = np.array(entry["R"], dtype=float)
        if Q.shape == (6, 6) and R.shape == (3, 3):
            return Q, R
    except Exception:
        pass
    return default_param_for(vehicle_type, mode)


# -----------------------------
# 12 维特征 + 模式判别（按载具阈值）
# -----------------------------
def _extract_features(window_xyz: np.ndarray, dt: float) -> Dict[str, float]:
    if window_xyz is None or len(window_xyz) < 3:
        return dict(avg_speed=0.0, max_speed=0.0, speed_std=0.0, vz_mean=0.0, avg_acc=0.0, max_acc=0.0, curvature=0.0)
    vel = np.diff(window_xyz, axis=0) / max(dt, 1e-6)
    acc = np.diff(vel, axis=0) / max(dt, 1e-6) if len(vel) > 1 else np.zeros((1, 3))
    vel_norm = np.linalg.norm(vel, axis=1)
    avg_speed = float(np.mean(vel_norm))
    max_speed = float(np.max(vel_norm))
    speed_std = float(np.std(vel_norm))
    vz = vel[:, 2]
    vz_mean = float(np.mean(vz))
    acc_norm = np.linalg.norm(acc, axis=1) if len(acc) > 0 else np.array([0.0])
    avg_acc = float(np.mean(acc_norm))
    max_acc = float(np.max(acc_norm))

    # 曲率近似（相邻速度向量夹角均值）
    curvature_vals: List[float] = []
    if len(vel) > 1:
        dirs = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-8)
        cos = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1, 1)
        ang = np.arccos(cos)
        curvature_vals = ang.tolist()
    curvature = float(np.mean(curvature_vals)) if curvature_vals else 0.0

    return dict(avg_speed=avg_speed, max_speed=max_speed, speed_std=speed_std,
                vz_mean=vz_mean, avg_acc=avg_acc, max_acc=max_acc, curvature=curvature)


def classify_motion_mode(window_xyz: np.ndarray, dt: float, vehicle_type: str) -> Tuple[str, float]:
    rules = SPEED_RULES.get(vehicle_type, SPEED_RULES["medium_large_quad"])
    f = _extract_features(window_xyz, dt)
    v, vmax, vz, acc_max, curv = f["avg_speed"], f["max_speed"], f["vz_mean"], f["max_acc"], f["curvature"]

    # 1) 垂直优先（明显爬升/下降）
    if vz > rules["vz_climb"]:
        return "climbing", 0.9
    if vz < rules["vz_desc"]:
        return "landing", 0.9

    # 2) 水平速度区间
    if v < rules["v_hover"]:
        return "hovering", 0.95
    if v < rules["v_cruise_max"]:
        return "cruising", 0.9
    if v <= rules["v_maneuver_max"]:
        # 若转向/加速度很大，提升到 aggressive
        if vmax > rules["v_aggr_min"] + 2.0 or acc_max > 8.0 or curv > 1.0:
            return "aggressive", 0.75
        return "maneuvering", 0.85

    # 3) 极高速或强机动
    if v >= rules["v_aggr_min"] or vmax >= rules["v_aggr_min"] or acc_max > 10.0 or curv > 1.2:
        return "aggressive", 0.8

    return "maneuvering", 0.6


# -----------------------------
# 基线UKF（固定参数对比）
# -----------------------------
class BaselineUKF:
    """固定参数基线UKF，用于性能对比"""

    def __init__(self, dt: float = 0.1):
        self.dt = dt
        points = MerweScaledSigmaPoints(n=6, alpha=1e-4, beta=2.0, kappa=0.0)
        self.ukf = UKF(dim_x=6, dim_z=3, dt=self.dt, hx=lambda x: x[:3], fx=self._fx, points=points)

        # 固定的中等参数设置
        self.ukf.Q = np.diag([0.2, 0.2, 0.2, 1.0, 1.0, 1.0])
        self.ukf.R = np.diag([0.5, 0.5, 0.5])
        self.ukf.P = np.eye(6) * 10.0
        self.ukf.x = np.zeros(6, dtype=float)

    def _fx(self, x: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        dt = self.dt if dt is None else dt
        nx = x.copy()
        nx[0] += nx[3] * dt
        nx[1] += nx[4] * dt
        nx[2] += nx[5] * dt
        return nx

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化基线UKF"""
        try:
            if len(initial_state) >= 6:
                self.ukf.x = initial_state[:6].copy().astype(float)
            elif len(initial_state) >= 3:
                self.ukf.x = np.array([
                    initial_state[0], initial_state[1], initial_state[2],
                    0.0, 0.0, 0.0
                ], dtype=float)
            else:
                return False
            return True
        except Exception as e:
            print(f"❌ BaselineUKF初始化失败: {e}")
            return False

    def step(self, z_xyz: np.ndarray) -> np.ndarray:
        z = np.asarray(z_xyz, dtype=float).reshape(3)
        self.ukf.predict()
        self.ukf.update(z)
        return self.ukf.x.copy()

    def predict_and_update(self, measurement: np.ndarray) -> np.ndarray:
        """兼容接口方法"""
        return self.step(measurement[:3])


# -----------------------------
# UKF 封装
# -----------------------------
@dataclass
class UKFConfig:
    dt: float = 0.1
    points_alpha: float = 1e-4
    points_beta: float = 2.0
    points_kappa: float = 0.0


class IndependentDynamicUKF:
    def __init__(self, dt: float = 0.1, vehicle_type: str = "medium_large_quad",
                 param_library: Optional[Dict] = None, source_path: Optional[str] = None):
        self.dt = float(dt)
        if vehicle_type is None and source_path is not None:
            vehicle_type = infer_vehicle_type_from_path(source_path)
        self.vehicle_type = vehicle_type if vehicle_type in VEHICLE_TYPES else "medium_large_quad"
        self.param_library = param_library if param_library is not None else load_param_library()

        points = MerweScaledSigmaPoints(n=6, alpha=1e-4, beta=2.0, kappa=0.0)
        self.ukf = UKF(dim_x=6, dim_z=3, dt=self.dt, hx=lambda x: x[:3], fx=self._fx, points=points)
        self.ukf.P = np.eye(6) * 10.0
        self.ukf.x = np.zeros(6, dtype=float)

        self._window: List[np.ndarray] = []
        self._last_mode = "cruising"
        self.mode_history = []
        self.mode_switches = []
        Q, R = select_QR(self.vehicle_type, self._last_mode, self.param_library)
        self.ukf.Q, self.ukf.R = Q, R



    def _fx(self, x: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
        dt = self.dt if dt is None else dt
        nx = x.copy()
        nx[0] += nx[3] * dt
        nx[1] += nx[4] * dt
        nx[2] += nx[5] * dt
        return nx

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化动态UKF"""
        try:
            # 强制把 initial_state 转成 numpy array，兼容 list / tuple / ndarray
            init = np.asarray(initial_state, dtype=float)
            if init.size >= 6:
                self.ukf.x = init[:6].copy().astype(float)
            elif init.size >= 3:
                self.ukf.x = np.array([
                    float(init[0]), float(init[1]), float(init[2]),
                    0.0, 0.0, 0.0
                ], dtype=float)
            else:
                return False

            # 重置历史记录
            self._window.clear()
            self.mode_history.clear()
            self.mode_switches.clear()

            return True
        except Exception as e:
            print(f"❌ IndependentDynamicUKF初始化失败: {e}")
            return False

    def _update_mode_and_params(self, z_xyz: np.ndarray):
        self._window.append(z_xyz.copy())
        if len(self._window) > 100:
            self._window.pop(0)

        # 每 30 步更新一次模式
        if len(self._window) >= 30 and (len(self._window) % 30 == 0):
            window_arr = np.array(self._window[-100:], dtype=float)
            mode, conf = classify_motion_mode(window_arr, self.dt, self.vehicle_type)

            # 记录模式切换
            if mode != self._last_mode:
                self.mode_switches.append({
                    'step': len(self.mode_history),
                    'from_mode': self._last_mode,
                    'to_mode': mode,
                    'confidence': conf
                })


            self._last_mode = mode
            self.mode_history.append(mode)
            Q, R = select_QR(self.vehicle_type, mode, self.param_library)
            self.ukf.Q, self.ukf.R = Q, R

    def step(self, z_xyz: np.ndarray) -> np.ndarray:
        """输入观测 z=[x,y,z]，执行 predict+update，内部自动判别模式并切换 Q/R。"""
        z = np.asarray(z_xyz, dtype=float).reshape(3)
        self._update_mode_and_params(z)
        self.ukf.predict()
        self.ukf.update(z)
        return self.ukf.x.copy()

    def predict_and_update(self, measurement: np.ndarray) -> np.ndarray:
        """兼容接口方法"""
        return self.step(measurement[:3])


# -----------------------------
# 测试和比较函数
# -----------------------------
def test_ukf_comparison(csv_file: str, vehicle_type: Optional[str] = None):
    """测试动态UKF vs 基线UKF性能比较"""

    print(f"🚀 UKF性能比较测试")
    print(f"📁 数据文件: {csv_file}")
    print("=" * 60)

    # 1. 加载数据
    try:
        df = pd.read_csv(csv_file)
        positions = df[['x', 'y', 'z']].values
        print(f"✅ 加载 {len(positions)} 个数据点")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None

    # 2. 检测时间间隔
    dt = detect_dt_from_csv(csv_file)
    print(f"⏰ 检测到时间间隔: {dt}s")

    # 3. 推断载具类型
    if vehicle_type is None:
        vehicle_type = infer_vehicle_type_from_path(csv_file)
    print(f"🚁 载具类型: {vehicle_type}")

    # 4. 添加观测噪声
    np.random.seed(42)
    noise_std = 0.25
    observations = positions + np.random.normal(0, noise_std, positions.shape)
    print(f"📊 观测噪声: σ={noise_std}m")

    # 5. 初始化滤波器
    print(f"🔧 初始化滤波器...")

    # 基线UKF
    baseline_ukf = BaselineUKF(dt=dt)
    baseline_ukf.ukf.x = np.array([positions[0, 0], positions[0, 1], positions[0, 2], 0, 0, 0])

    # 动态UKF
    dynamic_ukf = IndependentDynamicUKF(dt=dt, vehicle_type=vehicle_type, source_path=csv_file)
    dynamic_ukf.ukf.x = np.array([positions[0, 0], positions[0, 1], positions[0, 2], 0, 0, 0])

    # 6. 运行预测
    print(f"🚀 运行预测...")
    baseline_predictions = []
    dynamic_predictions = []

    for i in range(1, len(observations)):
        # 基线预测
        baseline_pred = baseline_ukf.step(observations[i])
        baseline_predictions.append(baseline_pred[:3])

        # 动态预测
        dynamic_pred = dynamic_ukf.step(observations[i])
        dynamic_predictions.append(dynamic_pred[:3])

        if (i + 1) % 200 == 0:
            current_mode = dynamic_ukf._last_mode
            print(f"  进度: {i + 1}/{len(observations) - 1}, 当前模式: {current_mode}")

    # 7. 计算误差
    baseline_predictions = np.array(baseline_predictions)
    dynamic_predictions = np.array(dynamic_predictions)
    true_positions = positions[1:]

    baseline_errors = np.linalg.norm(baseline_predictions - true_positions, axis=1)
    dynamic_errors = np.linalg.norm(dynamic_predictions - true_positions, axis=1)

    baseline_rmse = np.sqrt(np.mean(baseline_errors ** 2))
    dynamic_rmse = np.sqrt(np.mean(dynamic_errors ** 2))
    improvement = (baseline_rmse - dynamic_rmse) / baseline_rmse * 100

    # 8. 显示结果
    print(f"\n📊 UKF性能比较结果:")
    print(f"   数据集: {os.path.basename(csv_file)}")
    print(f"   载具类型: {vehicle_type}")
    print(f"   总数据点: {len(true_positions)}")
    print(f"   参数库状态: {'已加载' if dynamic_ukf.param_library else '默认参数'}")
    print("-" * 50)
    print(f"   固定参数UKF RMSE: {baseline_rmse:.3f}m")
    print(f"   飞行感知UKF RMSE: {dynamic_rmse:.3f}m")
    print(f"   🎯 性能改进: {improvement:.1f}%")

    # 9. 模式切换统计
    mode_switches = dynamic_ukf.mode_switches
    print(f"   模式切换次数: {len(mode_switches)}")

    if mode_switches:
        print(f"   切换序列:")
        for switch in mode_switches[-5:]:  # 显示最后5次切换
            step = switch['step']
            from_mode = switch['from_mode']
            to_mode = switch['to_mode']
            confidence = switch['confidence']
            print(f"     步骤{step}: {from_mode} → {to_mode} (置信度: {confidence:.3f})")

    # 10. 模式分布统计
    mode_history = dynamic_ukf.mode_history
    if mode_history:
        unique_modes, counts = np.unique(mode_history, return_counts=True)
        print(f"\n📈 飞行模式分布:")
        for mode, count in zip(unique_modes, counts):
            percentage = count / len(mode_history) * 100
            print(f"     {mode}: {count} 次 ({percentage:.1f}%)")

    # 11. 性能诊断
    print(f"\n🔍 性能分析:")
    if improvement > 10:
        diagnosis = "🟢 优秀! 飞行感知显著改进性能"
    elif improvement > 5:
        diagnosis = "🟡 良好! 飞行感知有所帮助"
    elif improvement > -2:
        diagnosis = "🟠 平稳! 性能基本持平"
    else:
        diagnosis = "🔴 需优化! 可能需要调整参数"

    print(f"   {diagnosis}")

    # 12. 创建可视化
    create_comparison_visualization(
        true_positions, baseline_predictions, dynamic_predictions,
        baseline_errors, dynamic_errors, dynamic_ukf, csv_file
    )

    return {
        'baseline_rmse': baseline_rmse,
        'dynamic_rmse': dynamic_rmse,
        'improvement': improvement,
        'mode_switches': mode_switches,
        'mode_history': mode_history,
        'vehicle_type': vehicle_type
    }


def create_comparison_visualization(true_pos, baseline_pred, dynamic_pred,
                                    baseline_err, dynamic_err, dynamic_ukf, csv_file):
    """创建性能比较可视化"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'飞行感知UKF vs 基线UKF: {os.path.basename(csv_file)}', fontsize=16, fontweight='bold')

    # 1. 轨迹对比
    axes[0, 0].plot(true_pos[:, 0], true_pos[:, 1], 'g-', label='真实轨迹', linewidth=3, alpha=0.8)
    axes[0, 0].plot(baseline_pred[:, 0], baseline_pred[:, 1], 'b--', label='固定参数UKF', linewidth=2)
    axes[0, 0].plot(dynamic_pred[:, 0], dynamic_pred[:, 1], 'r-', label='飞行感知UKF', linewidth=2)
    axes[0, 0].set_title('XY轨迹对比', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('X [m]')
    axes[0, 0].set_ylabel('Y [m]')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    # 2. 误差对比
    baseline_rmse = np.sqrt(np.mean(baseline_err ** 2))
    dynamic_rmse = np.sqrt(np.mean(dynamic_err ** 2))
    improvement = (baseline_rmse - dynamic_rmse) / baseline_rmse * 100

    axes[0, 1].plot(baseline_err, 'b-', label=f'固定参数UKF (RMSE: {baseline_rmse:.3f}m)', linewidth=2)
    axes[0, 1].plot(dynamic_err, 'r-', label=f'飞行感知UKF (RMSE: {dynamic_rmse:.3f}m)', linewidth=2)
    axes[0, 1].set_title(f'误差对比 (改进: {improvement:.1f}%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('误差 [m]')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 标记模式切换点
    for switch in dynamic_ukf.mode_switches:
        axes[0, 1].axvline(x=switch['step'] * 30, color='orange', linestyle='--', alpha=0.7)

    # 3. 模式演化
    mode_history = dynamic_ukf.mode_history
    if mode_history:
        mode_map = {mode: i for i, mode in enumerate(MOTION_MODES)}
        mode_numeric = [mode_map.get(m, 0) for m in mode_history]

        x_steps = np.arange(len(mode_numeric)) * 30  # 每30步记录一次
        axes[1, 0].plot(x_steps, mode_numeric, 'purple', marker='o', markersize=4, linewidth=2)
        axes[1, 0].set_title('飞行模式演化', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('飞行模式')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_yticks(list(mode_map.values()))
        axes[1, 0].set_yticklabels(list(mode_map.keys()))
        axes[1, 0].grid(True, alpha=0.3)

        # 标记切换点
        for switch in dynamic_ukf.mode_switches:
            axes[1, 0].axvline(x=switch['step'] * 30, color='red', linestyle='--', alpha=0.7)
    else:
        axes[1, 0].text(0.5, 0.5, '无模式切换数据', ha='center', va='center',
                        transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('飞行模式演化')

    # 4. 模式分布饼图
    if mode_history:
        unique_modes, counts = np.unique(mode_history, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_modes)))

        axes[1, 1].pie(counts, labels=unique_modes, autopct='%1.1f%%', colors=colors)
        axes[1, 1].set_title('飞行模式分布')
    else:
        axes[1, 1].text(0.5, 0.5, '无模式数据', ha='center', va='center',
                        transform=axes[1, 1].transAxes, fontsize=14)
        axes[1, 1].set_title('飞行模式分布')

    plt.tight_layout()

    # 保存图像
    output_file = f'ukf_comparison_{os.path.basename(csv_file).replace(".csv", "")}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 可视化结果已保存: {output_file}")
    plt.show()


# 简易工厂
def make_ukf(dt: float = 0.1, vehicle_type: Optional[str] = None,
             source_path: Optional[str] = None) -> IndependentDynamicUKF:
    vt = vehicle_type or infer_vehicle_type_from_path(source_path or "")
    return IndependentDynamicUKF(dt=dt, vehicle_type=vt, param_library=load_param_library(), source_path=source_path)


# 在你的 fixed_dynamic_ukf.py 中添加这个增强版本

# 在 fixed_dynamic_ukf.py 中添加这些修复方法

import numpy as np
from scipy.linalg import cholesky, LinAlgError

# 完整的 EnhancedIndependentDynamicUKF 实现
# 将此代码添加到你的 fixed_dynamic_ukf.py 文件中

import numpy as np
from scipy.linalg import cholesky, LinAlgError
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from typing import Optional


class NumericalStabilityMixin:
    """UKF数值稳定性修复混入类"""

    def ensure_positive_definite(self, P, regularization=1e-6):
        """确保协方差矩阵正定"""
        try:
            # 尝试Cholesky分解检查正定性
            cholesky(P)
            return P
        except LinAlgError:
            # 如果不是正定，进行修复
            eigenvals, eigenvecs = np.linalg.eigh(P)

            # 将负特征值设为小正数
            eigenvals = np.maximum(eigenvals, regularization)

            # 重构正定矩阵
            P_fixed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            # 确保对称性
            P_fixed = (P_fixed + P_fixed.T) / 2

            return P_fixed

    def safe_parameter_validation(self, alpha, beta, kappa, p_pos, p_vel):
        """安全的参数验证和修正"""
        # 限制参数到安全范围
        alpha = np.clip(alpha, 1e-4, 1.0)
        beta = np.clip(beta, 0.1, 4.0)
        kappa = np.clip(kappa, -1.0, 3.0)
        p_pos = np.clip(p_pos, 0.1, 100.0)
        p_vel = np.clip(p_vel, 0.01, 50.0)

        # 检查alpha-kappa组合的数值稳定性
        n = 6  # 状态维度
        lambda_val = alpha ** 2 * (n + kappa) - n

        # 如果lambda值过小，可能导致数值问题
        if lambda_val < 1e-6:
            # 调整kappa使lambda值合理
            kappa = max(kappa, (1e-6 + n) / alpha ** 2 - n)

        return alpha, beta, kappa, p_pos, p_vel


class EnhancedIndependentDynamicUKF(NumericalStabilityMixin):
    """增强数值稳定性的动态UKF - 完整版本"""

    def __init__(self, dt=0.1):
        self.dt = dt
        self.ukf = None
        self.initialized = False

        # 安全的默认参数
        self.default_params = {
            'alpha': 0.1,
            'beta': 2.0,
            'kappa': 0.0,
            'p_pos': 5.0,
            'p_vel': 2.0
        }

        # 统计信息
        self.recovery_count = 0
        self.total_steps = 0
        self.nn_success_count = 0
        self.nn_failure_count = 0

        # 历史状态用于恢复
        self.last_velocity = None
        self.position_history = []

    def _fx(self, x: np.ndarray, dt: Optional[float] = None) -> np.ndarray:
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

    def initialize(self, initial_state: np.ndarray) -> bool:
        """初始化UKF"""
        try:
            # 创建sigma点生成器
            points = MerweScaledSigmaPoints(
                n=6,
                alpha=self.default_params['alpha'],
                beta=self.default_params['beta'],
                kappa=self.default_params['kappa']
            )

            # 创建UKF实例
            self.ukf = UKF(
                dim_x=6,
                dim_z=3,
                dt=self.dt,
                hx=self._hx,
                fx=self._fx,
                points=points
            )

            # 设置噪声矩阵
            self.ukf.Q = np.diag([0.1, 0.1, 0.1, 0.5, 0.5, 0.5])  # 过程噪声
            self.ukf.R = np.diag([0.5, 0.5, 0.5])  # 观测噪声

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
            self.ukf.P[0:3, 0:3] *= self.default_params['p_pos']  # 位置不确定性
            self.ukf.P[3:6, 3:6] *= self.default_params['p_vel']  # 速度不确定性

            # 确保协方差矩阵正定
            self.ukf.P = self.ensure_positive_definite(self.ukf.P)

            # 重置统计
            self.recovery_count = 0
            self.total_steps = 0
            self.nn_success_count = 0
            self.nn_failure_count = 0
            self.position_history = [initial_state[:3].copy()]

            self.initialized = True

            return True

        except Exception as e:
            print(f"EnhancedIndependentDynamicUKF初始化失败: {e}")
            return False

    def update_parameters(self, alpha, beta, kappa, p_pos, p_vel):
        """更新UKF参数（带安全检查）"""
        if not self.initialized or self.ukf is None:
            return

        try:
            # 验证和修正参数
            alpha, beta, kappa, p_pos, p_vel = self.safe_parameter_validation(
                alpha, beta, kappa, p_pos, p_vel
            )

            # 创建新的sigma点生成器
            points = MerweScaledSigmaPoints(
                n=6, alpha=alpha, beta=beta, kappa=kappa
            )

            # 更新UKF的sigma点生成器
            self.ukf.points_fn = points

            # 调整协方差矩阵的尺度
            P_new = self.ukf.P.copy()
            current_pos_scale = np.mean(np.diag(P_new[0:3, 0:3]))
            current_vel_scale = np.mean(np.diag(P_new[3:6, 3:6]))

            if current_pos_scale > 0:
                P_new[0:3, 0:3] *= (p_pos / current_pos_scale)
            else:
                P_new[0:3, 0:3] = np.eye(3) * p_pos

            if current_vel_scale > 0:
                P_new[3:6, 3:6] *= (p_vel / current_vel_scale)
            else:
                P_new[3:6, 3:6] = np.eye(3) * p_vel

            # 确保正定性
            self.ukf.P = self.ensure_positive_definite(P_new)

            self.nn_success_count += 1

        except Exception as e:
            self.nn_failure_count += 1
            print(f"NN参数更新失败，使用默认参数: {e}")
            self._use_default_parameters()

    def _use_default_parameters(self):
        """使用安全的默认参数"""
        if self.ukf is not None:
            try:
                points = MerweScaledSigmaPoints(
                    n=6,
                    alpha=self.default_params['alpha'],
                    beta=self.default_params['beta'],
                    kappa=self.default_params['kappa']
                )
                self.ukf.points_fn = points
            except Exception as e:
                print(f"设置默认参数失败: {e}")

    def step(self, measurement):
        """安全的UKF步骤"""
        self.total_steps += 1
        measurement = np.array(measurement[:3], dtype=float)

        if not self.initialized or self.ukf is None:
            return measurement

        try:
            # 保存当前状态用于恢复
            prev_x = self.ukf.x.copy()
            prev_P = self.ukf.P.copy()

            # 预测步骤
            self.ukf.predict()

            # 检查预测后的协方差矩阵
            if np.any(np.isnan(self.ukf.P)) or np.any(np.isinf(self.ukf.P)):
                raise ValueError("协方差矩阵包含NaN或Inf")

            # 确保协方差矩阵正定
            self.ukf.P = self.ensure_positive_definite(self.ukf.P)

            # 更新步骤
            self.ukf.update(measurement)

            # 更新后再次检查
            self.ukf.P = self.ensure_positive_definite(self.ukf.P)

            # 更新历史信息
            self.position_history.append(measurement.copy())
            if len(self.position_history) > 10:  # 保持最近10个位置
                self.position_history.pop(0)

            # 更新速度估计
            if len(self.position_history) >= 2:
                self.last_velocity = (self.position_history[-1] - self.position_history[-2]) / self.dt

            return self.ukf.x[:3].copy()

        except Exception as e:
            self.recovery_count += 1
            error_msg = str(e).lower()

            print(f"UKF步骤失败，使用恢复策略: {e}")

            # 恢复策略1: 协方差矩阵问题
            if "positive definite" in error_msg or "cholesky" in error_msg:
                try:
                    # 重置协方差矩阵
                    self.ukf.P = np.eye(6) * 10.0
                    self.ukf.P[0:3, 0:3] *= 5.0  # 位置不确定性
                    self.ukf.P[3:6, 3:6] *= 2.0  # 速度不确定性

                    # 恢复到安全状态
                    if len(self.position_history) > 0:
                        self.ukf.x[:3] = self.position_history[-1]
                    else:
                        self.ukf.x[:3] = measurement

                    # 使用默认参数
                    self._use_default_parameters()

                    print("协方差矩阵已重置")

                except Exception as e2:
                    print(f"协方差重置失败: {e2}")

            # 恢复策略2: 使用物理模型预测
            try:
                if self.last_velocity is not None and len(self.position_history) >= 2:
                    # 基于历史速度的预测
                    predicted = self.position_history[-1] + self.last_velocity * self.dt

                    # 限制预测到合理范围
                    max_velocity = 25.0  # m/s
                    vel_norm = np.linalg.norm(self.last_velocity)
                    if vel_norm > max_velocity:
                        self.last_velocity = self.last_velocity * (max_velocity / vel_norm)
                        predicted = self.position_history[-1] + self.last_velocity * self.dt

                    # 更新UKF状态
                    if self.ukf is not None:
                        self.ukf.x[:3] = predicted
                        self.ukf.x[3:6] = self.last_velocity

                    return predicted
                else:
                    # 没有速度历史，返回当前观测
                    return measurement

            except Exception as e3:
                print(f"物理预测也失败: {e3}")
                return measurement

    def get_nn_performance_stats(self):
        """获取NN性能统计"""
        total_nn_attempts = self.nn_success_count + self.nn_failure_count
        return {
            'total_steps': self.total_steps,
            'recovery_count': self.recovery_count,
            'recovery_rate': self.recovery_count / max(self.total_steps, 1),
            'nn_success_count': self.nn_success_count,
            'nn_failure_count': self.nn_failure_count,
            'nn_success_rate': self.nn_success_count / max(total_nn_attempts, 1) if total_nn_attempts > 0 else 0,
            'total_nn_attempts': total_nn_attempts
        }

    def get_debug_info(self):
        """获取详细调试信息"""
        stats = self.get_nn_performance_stats()
        base_info = {
            'model_type': 'enhanced_independent_dynamic_ukf',
            'initialized': self.initialized,
            'dt': self.dt,
            'position_history_length': len(self.position_history),
            'has_velocity_estimate': self.last_velocity is not None
        }
        base_info.update(stats)
        return base_info


# 为了保持兼容性，创建一个别名
def IndependentDynamicUKF_WithNN(*args, **kwargs):
    """兼容性工厂函数"""
    return EnhancedIndependentDynamicUKF(*args, **kwargs)
# -----------------------------
# 主函数
# -----------------------------
def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("🚁 飞行感知UKF测试器")
        print("=" * 50)
        print("用法: python fixed_dynamic_ukf.py <csv_file> [vehicle_type]")
        print("")
        print("📋 支持的载具类型:")
        print("  • micro_quad - 微小型四旋翼")
        print("  • medium_large_quad - 中大型四旋翼")
        print("  • fixed_wing - 固定翼无人机")
        print("  • heavy_multirotor - 重型多旋翼")
        print("  (不指定将自动推断)")
        print("")
        print("🎯 功能特性:")
        print("  ✅ 4×6载具-模式参数库")
        print("  ✅ 实时飞行模式识别")
        print("  ✅ 动态参数切换")
        print("  ✅ 基线UKF性能对比")
        print("  ✅ 可视化分析报告")
        print("")
        print("📊 输出信息:")
        print("  • RMSE性能对比")
        print("  • 飞行模式切换统计")
        print("  • 模式分布分析")
        print("  • 轨迹预测可视化")
        print("")
        print("示例:")
        print("  python fixed_dynamic_ukf.py drone_flight_data.csv")
        print("  python fixed_dynamic_ukf.py enhanced_drone_data.csv medium_large_quad")
        return

    csv_file = sys.argv[1]
    vehicle_type = sys.argv[2] if len(sys.argv) > 2 else None

    # 检查文件存在
    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return

    # 检查参数库
    param_lib = load_param_library()
    if param_lib:
        print(f"📊 参数库状态: 已加载优化参数")
    else:
        print(f"📊 参数库状态: 使用默认参数")
        print(f"💡 提示: 运行 python ukf_parameter_optimizer.py 生成优化参数")

    print()

    # 运行测试
    result = test_ukf_comparison(csv_file, vehicle_type)

    if result:
        print(f"\n✅ 测试完成!")
        print(f"🏆 最终结果:")
        print(f"   固定参数UKF: {result['baseline_rmse']:.3f}m")
        print(f"   飞行感知UKF: {result['dynamic_rmse']:.3f}m")
        print(f"   🎯 性能改进: {result['improvement']:.1f}%")
        print(f"   载具类型: {result['vehicle_type']}")
        print(f"   模式切换: {len(result['mode_switches'])} 次")

        if result['improvement'] > 0:
            print(f"\n🎯 结论: 飞行感知UKF策略有效!")
        else:
            print(f"\n🔧 结论: 系统运行正常，在更复杂数据上可能表现更佳")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 简单冒烟测试（不会读文件，只跑几步）
        ukf = make_ukf(0.1, source_path="enhanced_drone_data.csv")  # 推断为 medium_large_quad
        for i in range(60):
            # 模拟观测：直线前进 + 轻微噪声
            z = np.array([i * 1.2, 0.0, 0.0]) + np.random.normal(0, 0.05, 3)
            x = ukf.step(z)
        print("Vehicle:", ukf.vehicle_type, "Mode:", ukf._last_mode)
    else:
        main()