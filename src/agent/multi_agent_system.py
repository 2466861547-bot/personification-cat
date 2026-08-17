"""
多智能体娱乐编排系统
- Composer Agent: 根据宠物状态和时间段编排娱乐内容
- Performer Agent: 执行具体的娱乐内容生成
- Scheduler Agent: 管理定时调度和动态调整
- Observer Agent: 模拟宠物反馈,调整策略

基于 AutoGen 多智能体对话模式 + CrewAI 角色协作
"""

import os
import json
import random
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from .entertainment import (
    EntertainmentEngine,
    EntertainmentMode,
    XIANGSHENG_SCRIPTS,
    SINGING_SONGS,
    SOOTHING_SOUNDS,
)
from .scheduler import EntertainmentScheduler, ScheduleConfig


class AgentRole(Enum):
    """智能体角色"""
    COMPOSER = "composer"     # 内容编排者
    PERFORMER = "performer"   # 内容执行者
    SCHEDULER = "scheduler"  # 调度管理者
    OBSERVER = "observer"     # 观察反馈者


@dataclass
class PetState:
    """宠物状态 (模拟)"""
    mood: str = "neutral"       # neutral / happy / sleepy / bored / excited
    energy_level: float = 0.5   # 0-1
    last_interaction: str = ""  # 上次互动时间
    interaction_count: int = 0  # 今日互动次数
    preferred_modes: List[str] = None  # 偏好模式

    def __post_init__(self):
        if self.preferred_modes is None:
            self.preferred_modes = ["singing", "soothing"]


class ComposerAgent:
    """
    编排者 Agent: 根据宠物状态 + 时间段决定娱乐内容
    - 早上: 活泼内容(逗乐/相声)唤醒宠物
    - 中午: 轻松内容(唱歌/讲故事)
    - 晚上: 安抚内容(安抚音/摇篮曲)助眠
    """

    def __init__(self, llm=None):
        self.llm = llm

    def compose(
        self,
        pet_state: PetState,
        pet_type: str = "cat",
        current_time: datetime = None,
    ) -> Dict:
        """编排娱乐内容"""
        current_time = current_time or datetime.now()
        hour = current_time.hour

        # 根据时间段决定基调
        if 6 <= hour < 11:
            # 早上: 活泼
            preferred_modes = ["teasing", "xiangsheng", "singing"]
            mood_target = "excited"
        elif 11 <= hour < 14:
            # 中午: 轻松
            preferred_modes = ["singing", "storytelling", "xiangsheng"]
            mood_target = "happy"
        elif 14 <= hour < 18:
            # 下午: 互动
            preferred_modes = ["xiangsheng", "teasing", "singing"]
            mood_target = "happy"
        elif 18 <= hour < 21:
            # 晚上: 温馨
            preferred_modes = ["storytelling", "singing", "soothing"]
            mood_target = "content"
        else:
            # 夜晚: 安抚
            preferred_modes = ["soothing", "storytelling"]
            mood_target = "sleepy"

        # 根据宠物状态调整
        if pet_state.energy_level < 0.3:
            # 能量低,选择安抚
            preferred_modes = ["soothing", "storytelling"]
        elif pet_state.energy_level > 0.7 and pet_state.mood == "bored":
            # 能量高但无聊,选择逗乐
            preferred_modes = ["teasing", "xiangsheng"]

        # 如果有偏好,优先
        if pet_state.preferred_modes:
            for pref in pet_state.preferred_modes:
                if pref in preferred_modes:
                    preferred_modes.insert(0, pref)
                    break

        selected_mode = preferred_modes[0]

        # 如果有 LLM,做更智能的编排
        if self.llm:
            prompt = (
                f"当前时间: {current_time.strftime('%H:%M')}\n"
                f"宠物类型: {pet_type}\n"
                f"宠物状态: 情绪={pet_state.mood}, 能量={pet_state.energy_level:.1f}\n"
                f"今日已互动: {pet_state.interaction_count} 次\n"
                f"可选模式: xiangsheng, singing, soothing, teasing, storytelling\n"
                f"请选择最合适的娱乐模式,并说明理由。"
            )
            llm_suggestion = self.llm.chat(
                prompt,
                system_prompt="你是宠物娱乐编排专家。根据宠物状态选择最佳娱乐模式。",
                max_tokens=100,
            )
        else:
            llm_suggestion = ""

        plan = {
            "mode": selected_mode,
            "pet_type": pet_type,
            "time_slot": self._get_time_slot(hour),
            "mood_target": mood_target,
            "reason": f"当前时段({self._get_time_slot(hour)})推荐{selected_mode}模式"
                      + (f"\nAI建议: {llm_suggestion}" if llm_suggestion else ""),
            "alternatives": preferred_modes[1:3],
        }

        print(f"[Composer] 编排方案: {plan['mode']} (目标情绪: {mood_target})")
        return plan

    def _get_time_slot(self, hour: int) -> str:
        if 6 <= hour < 11:
            return "早上"
        elif 11 <= hour < 14:
            return "中午"
        elif 14 <= hour < 18:
            return "下午"
        elif 18 <= hour < 21:
            return "晚上"
        else:
            return "夜晚"


class PerformerAgent:
    """
    执行者 Agent: 执行 Composer 编排的娱乐内容
    """

    def __init__(self, engine: EntertainmentEngine = None):
        self.engine = engine or EntertainmentEngine()

    def perform(self, plan: Dict, output_dir: str = "./output/entertainment") -> Dict:
        """执行娱乐内容"""
        mode = plan["mode"]
        pet_type = plan["pet_type"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, pet_type, f"{mode}_{timestamp}.wav")

        print(f"[Performer] 执行模式: {mode}, 宠物: {pet_type}")

        if mode == "xiangsheng":
            result = self.engine.play_xiangsheng(
                pet_type=pet_type,
                script_index=random.randint(0, 2),
                output_path=output_path,
            )
        elif mode == "singing":
            result = self.engine.play_singing(
                pet_type=pet_type,
                song_index=random.randint(0, 2),
                output_path=output_path,
            )
        elif mode == "soothing":
            sound_types = list(SOOTHING_SOUNDS.get(pet_type, {}).keys())
            result = self.engine.play_soothing(
                pet_type=pet_type,
                sound_type=random.choice(sound_types),
                duration=300,
                output_path=output_path,
            )
        elif mode == "teasing":
            result = self.engine.play_teasing(
                pet_type=pet_type,
                output_path=output_path,
            )
        elif mode == "storytelling":
            result = self.engine.play_storytelling(
                pet_type=pet_type,
                output_path=output_path,
            )
        else:
            result = {"error": f"未知模式: {mode}"}

        result["plan"] = plan
        return result


class ObserverAgent:
    """
    观察者 Agent: 模拟宠物反馈,调整策略
    - 根据互动模式模拟宠物反应
    - 记录偏好,优化下次推荐
    """

    # 宠物反应模拟规则
    REACTION_RULES = {
        "cat": {
            "xiangsheng": {"reaction": "tilts_head", "engagement": 0.6, "mood_change": +0.1},
            "singing": {"reaction": "purrs", "engagement": 0.8, "mood_change": +0.2},
            "soothing": {"reaction": "falls_asleep", "engagement": 0.9, "mood_change": +0.3},
            "teasing": {"reaction": "chases", "engagement": 0.9, "mood_change": +0.3},
            "storytelling": {"reaction": "listens", "engagement": 0.5, "mood_change": +0.1},
        },
        "dog": {
            "xiangsheng": {"reaction": "tilts_head", "engagement": 0.7, "mood_change": +0.1},
            "singing": {"reaction": "howls_along", "engagement": 0.8, "mood_change": +0.2},
            "soothing": {"reaction": "calms_down", "engagement": 0.85, "mood_change": +0.25},
            "teasing": {"reaction": "excited", "engagement": 0.95, "mood_change": +0.3},
            "storytelling": {"reaction": "listens", "engagement": 0.6, "mood_change": +0.1},
        },
    }

    def observe(self, performance: Dict, pet_state: PetState) -> Dict:
        """观察宠物反应"""
        mode = performance.get("plan", {}).get("mode", "")
        pet_type = performance.get("plan", {}).get("pet_type", "cat")

        rules = self.REACTION_RULES.get(pet_type, {}).get(mode, {})
        reaction = rules.get("reaction", "neutral")
        engagement = rules.get("engagement", 0.5)
        mood_change = rules.get("mood_change", 0.0)

        # 添加随机性
        engagement = min(1.0, engagement + random.uniform(-0.1, 0.1))

        # 更新宠物状态
        pet_state.mood = self._mood_from_engagement(engagement)
        pet_state.energy_level = max(0, min(1, pet_state.energy_level - 0.1))
        pet_state.interaction_count += 1
        pet_state.last_interaction = datetime.now().isoformat()

        # 记录偏好
        if engagement > 0.7 and mode not in pet_state.preferred_modes:
            pet_state.preferred_modes.insert(0, mode)
            pet_state.preferred_modes = pet_state.preferred_modes[:3]

        observation = {
            "reaction": reaction,
            "engagement": round(engagement, 2),
            "mood_after": pet_state.mood,
            "energy_after": round(pet_state.energy_level, 2),
            "mode_played": mode,
            "preferred_modes": pet_state.preferred_modes,
        }

        print(f"[Observer] 宠物反应: {reaction}, 参与度: {engagement:.0%}")
        return observation

    def _mood_from_engagement(self, engagement: float) -> str:
        if engagement > 0.8:
            return "excited"
        elif engagement > 0.6:
            return "happy"
        elif engagement > 0.4:
            return "content"
        elif engagement > 0.2:
            return "bored"
        else:
            return "sleepy"


class MultiAgentEntertainmentSystem:
    """
    多智能体娱乐编排系统
    整合 Composer + Performer + Observer + Scheduler
    """

    def __init__(
        self,
        pet_type: str = "cat",
        times_per_day: int = 3,
        llm=None,
        output_dir: str = "./output/entertainment",
    ):
        self.pet_type = pet_type
        self.llm = llm
        self.output_dir = output_dir

        # 初始化各 Agent
        self.composer = ComposerAgent(llm=llm)
        self.performer = PerformerAgent()
        self.observer = ObserverAgent()

        # 宠物状态
        self.pet_state = PetState(
            mood="neutral",
            energy_level=0.5,
        )

        # 调度器
        self.scheduler = EntertainmentScheduler(
            config=ScheduleConfig(
                pet_type=pet_type,
                times_per_day=times_per_day,
                output_dir=output_dir,
            )
        )

        # 互动记录
        self.interaction_log: List[Dict] = []

    def run_once(self, forced_mode: str = None) -> Dict:
        """执行一次完整的多智能体协作"""
        print("=" * 60)
        print(f"[MultiAgent] 开始互动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # 1. Composer: 编排内容
        if forced_mode:
            plan = {
                "mode": forced_mode,
                "pet_type": self.pet_type,
                "time_slot": "自定义",
                "mood_target": "custom",
                "reason": f"用户指定模式: {forced_mode}",
                "alternatives": [],
            }
        else:
            plan = self.composer.compose(
                pet_state=self.pet_state,
                pet_type=self.pet_type,
            )

        # 2. Performer: 执行
        performance = self.performer.perform(plan, self.output_dir)

        # 3. Observer: 观察反馈
        observation = self.observer.observe(performance, self.pet_state)

        # 记录
        record = {
            "timestamp": datetime.now().isoformat(),
            "plan": plan,
            "performance": {
                "output_path": performance.get("output_path", ""),
                "duration": performance.get("duration", 0),
            },
            "observation": observation,
            "pet_state": {
                "mood": self.pet_state.mood,
                "energy": self.pet_state.energy_level,
                "interaction_count": self.pet_state.interaction_count,
            },
        }
        self.interaction_log.append(record)

        print(f"[MultiAgent] 互动完成, 宠物当前情绪: {self.pet_state.mood}")
        print(f"[MultiAgent] 宠物参与度: {observation['engagement']:.0%}")

        return record

    def start_scheduled(self):
        """启动定时调度"""
        # 覆盖调度器的执行函数
        self.scheduler._execute_entertainment = lambda mode: self.run_once(mode)
        self.scheduler.start()

    def stop_scheduled(self):
        """停止调度"""
        self.scheduler.stop()

    def update_schedule(self, times_per_day: int = None, modes: List[str] = None):
        """更新调度配置"""
        if times_per_day:
            self.scheduler.set_times_per_day(times_per_day)
        if modes:
            self.scheduler.update_config(modes=modes)

    def get_status(self) -> Dict:
        """获取系统状态"""
        return {
            "pet_type": self.pet_type,
            "pet_state": {
                "mood": self.pet_state.mood,
                "energy": self.pet_state.energy_level,
                "interactions_today": self.pet_state.interaction_count,
                "preferred_modes": self.pet_state.preferred_modes,
            },
            "schedule": {
                "times_per_day": self.scheduler.config.times_per_day,
                "fixed_times": self.scheduler.config.fixed_times,
                "modes": self.scheduler.config.modes,
            },
            "next_run": self.scheduler.get_next_run(),
            "total_interactions": len(self.interaction_log),
        }

    def get_interaction_history(self, limit: int = 10) -> List[Dict]:
        """获取互动历史"""
        return self.interaction_log[-limit:]

    def save_log(self, path: str = "./output/interaction_log.json"):
        """保存互动日志"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.interaction_log, f, ensure_ascii=False, indent=2, default=str)
        print(f"互动日志已保存: {path}")
