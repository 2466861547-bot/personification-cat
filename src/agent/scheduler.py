"""
定时调度器: 动态配置逗猫逗狗时间表
- 一天 3-5 次定时互动 (可动态调整)
- 支持运行时修改配置
- 多智能体协作编排
"""

import os
import json
import time
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from .entertainment import EntertainmentEngine, EntertainmentMode


class ScheduleType(Enum):
    """调度类型"""
    FIXED = "fixed"        # 固定时间
    INTERVAL = "interval" # 固定间隔
    SMART = "smart"        # 智能时间 (根据宠物作息)


@dataclass
class ScheduleConfig:
    """调度配置"""
    enabled: bool = True
    pet_type: str = "cat"
    times_per_day: int = 3          # 每天次数 (3-5)
    fixed_times: List[str] = field(default_factory=lambda: [
        "09:00", "13:00", "17:00"
    ])
    modes: List[str] = field(default_factory=lambda: [
        "xiangsheng", "singing", "soothing"
    ])
    duration_minutes: int = 5       # 每次互动时长
    output_dir: str = "./output/entertainment"
    auto_play: bool = True          # 自动播放音频
    random_mode: bool = True        # 随机选择模式


class EntertainmentScheduler:
    """
    定时娱乐调度器
    - 支持 3-5 次/天动态配置
    - 运行时可修改时间表
    - 线程安全后台执行
    """

    def __init__(self, config: ScheduleConfig = None):
        self.config = config or ScheduleConfig()
        self.engine = EntertainmentEngine()
        self.scheduler = schedule.Scheduler()
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._history: List[Dict] = []
        self._callbacks: List[Callable] = []

    def update_config(self, **kwargs):
        """动态更新配置 (运行时安全调用)"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            print(f"配置已更新: {kwargs}")
            # 如果正在运行,重新调度
            if self._running:
                self._reschedule()

    def set_times_per_day(self, times: int):
        """设置每天互动次数 (3-5)"""
        times = max(1, min(times, 10))
        self.update_config(times_per_day=times)
        # 自动生成时间表
        self._auto_generate_times(times)

    def _auto_generate_times(self, times: int):
        """自动生成均衡时间表"""
        # 默认 9:00 - 21:00 之间均匀分布
        start_hour = 9
        end_hour = 21
        interval = (end_hour - start_hour) / times

        new_times = []
        for i in range(times):
            hour = start_hour + interval * (i + 0.5)
            h = int(hour)
            m = int((hour - h) * 60)
            new_times.append(f"{h:02d}:{m:02d}")

        self.update_config(fixed_times=new_times)
        print(f"自动生成时间表: {new_times}")

    def _reschedule(self):
        """重新调度"""
        self.scheduler.clear()
        self._setup_schedule()

    def _setup_schedule(self):
        """设置定时任务"""
        with self._lock:
            for time_str in self.config.fixed_times:
                mode = self._select_mode()
                self.scheduler.every().day.at(time_str).do(
                    self._execute_entertainment, mode=mode
                )
                print(f"  定时任务: {time_str} → {mode}")

    def _select_mode(self) -> str:
        """选择娱乐模式"""
        if self.config.random_mode:
            import random
            return random.choice(self.config.modes)
        else:
            # 轮询
            if not hasattr(self, "_mode_index"):
                self._mode_index = 0
            mode = self.config.modes[self._mode_index % len(self.config.modes)]
            self._mode_index += 1
            return mode

    def _execute_entertainment(self, mode: str):
        """执行一次娱乐互动"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            self.config.output_dir,
            self.config.pet_type,
            f"{mode}_{timestamp}.wav"
        )

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 开始娱乐互动: {mode}")

        try:
            if mode == "xiangsheng":
                result = self.engine.play_xiangsheng(
                    pet_type=self.config.pet_type,
                    output_path=output_path,
                )
            elif mode == "singing":
                result = self.engine.play_singing(
                    pet_type=self.config.pet_type,
                    output_path=output_path,
                )
            elif mode == "soothing":
                result = self.engine.play_soothing(
                    pet_type=self.config.pet_type,
                    output_path=output_path,
                )
            elif mode == "teasing":
                result = self.engine.play_teasing(
                    pet_type=self.config.pet_type,
                    output_path=output_path,
                )
            elif mode == "storytelling":
                result = self.engine.play_storytelling(
                    pet_type=self.config.pet_type,
                    output_path=output_path,
                )
            else:
                result = {"error": f"未知模式: {mode}"}

            # 记录历史
            record = {
                "timestamp": timestamp,
                "mode": mode,
                "pet_type": self.config.pet_type,
                "output_path": output_path,
                "result": result,
            }
            self._history.append(record)

            # 通知回调
            for callback in self._callbacks:
                try:
                    callback(record)
                except Exception as e:
                    print(f"回调执行失败: {e}")

            print(f"  互动完成: {output_path}")

        except Exception as e:
            print(f"  互动失败: {e}")

    def add_callback(self, callback: Callable):
        """添加互动完成回调"""
        self._callbacks.append(callback)

    def start(self):
        """启动调度器"""
        if self._running:
            print("调度器已在运行")
            return

        self._reschedule()
        self._running = True

        def run_loop():
            while self._running:
                self.scheduler.run_pending()
                time.sleep(1)

        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        print(f"调度器已启动,每天 {len(self.config.fixed_times)} 次互动")
        print(f"时间表: {', '.join(self.config.fixed_times)}")

    def stop(self):
        """停止调度器"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.scheduler.clear()
        print("调度器已停止")

    def trigger_now(self, mode: str = None):
        """立即触发一次"""
        mode = mode or self._select_mode()
        self._execute_entertainment(mode)

    def get_next_run(self) -> Optional[str]:
        """获取下次运行时间"""
        jobs = self.scheduler.jobs
        if not jobs:
            return None

        now = datetime.now()
        next_times = []
        for job in jobs:
            next_run = job.next_run
            if next_run:
                next_times.append(next_run)

        if not next_times:
            return None

        next_time = min(next_times)
        delta = next_time - now
        return f"{next_time.strftime('%H:%M')} (还有 {int(delta.total_seconds() / 60)} 分钟)"

    def get_history(self, limit: int = 10) -> List[Dict]:
        """获取互动历史"""
        return self._history[-limit:]

    def save_config(self, path: str = "./config/schedule.json"):
        """保存配置"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, ensure_ascii=False, indent=2)
        print(f"配置已保存: {path}")

    def load_config(self, path: str = "./config/schedule.json"):
        """加载配置"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            print(f"配置已加载: {path}")
