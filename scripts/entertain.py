"""
逗猫逗狗娱乐脚本
用法:
  # 立即执行一次
  python scripts/entertain.py --mode singing --pet cat

  # 启动定时调度 (一天3次)
  python scripts/entertain.py --schedule --pet cat --times 3

  # 查看可用模式
  python scripts/entertain.py --list --pet cat

  # 多智能体模式
  python scripts/entertain.py --multi-agent --pet dog --times 5
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.entertainment import EntertainmentEngine
from src.agent.scheduler import EntertainmentScheduler, ScheduleConfig
from src.agent.multi_agent_system import MultiAgentEntertainmentSystem


def list_modes(pet_type: str):
    """列出所有模式"""
    engine = EntertainmentEngine()
    modes = engine.list_modes(pet_type)
    print(f"\n{'='*50}")
    print(f"  {pet_type.upper()} 可用娱乐模式")
    print(f"{'='*50}")
    for mode_id, info in modes.items():
        print(f"\n  [{mode_id}] {info['name']}")
        print(f"  描述: {info['description']}")
        if 'scripts' in info:
            print(f"  相声段子: {', '.join(info['scripts'])}")
        if 'songs' in info:
            print(f"  歌曲: {', '.join(info['songs'])}")
        if 'sounds' in info:
            print(f"  安抚音: {', '.join(info['sounds'])}")


def play_once(mode: str, pet_type: str, output_dir: str):
    """立即执行一次"""
    engine = EntertainmentEngine()
    timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, pet_type, f"{mode}_{timestamp}.wav")

    if mode == "xiangsheng":
        result = engine.play_xiangsheng(pet_type=pet_type, output_path=output_path)
    elif mode == "singing":
        result = engine.play_singing(pet_type=pet_type, output_path=output_path)
    elif mode == "soothing":
        result = engine.play_soothing(pet_type=pet_type, duration=60, output_path=output_path)
    elif mode == "teasing":
        result = engine.play_teasing(pet_type=pet_type, output_path=output_path)
    elif mode == "storytelling":
        result = engine.play_storytelling(pet_type=pet_type, output_path=output_path)
    else:
        print(f"未知模式: {mode}")
        return

    print(f"\n{'='*50}")
    print(f"  执行完成: {mode}")
    print(f"{'='*50}")
    print(f"  宠物类型: {result.get('pet_type', pet_type)}")
    if result.get('title'):
        print(f"  标题: {result['title']}")
    if result.get('description'):
        print(f"  描述: {result['description']}")
    if result.get('lyrics'):
        print(f"\n  歌词:\n{result['lyrics']}")
    if result.get('script'):
        print(f"\n  相声内容:")
        for i, line in enumerate(result['script']):
            speaker = "逗哏" if i % 2 == 0 else "捧哏"
            print(f"    {speaker}: {line}")
    if result.get('story'):
        print(f"\n  故事:\n{result['story']}")
    print(f"\n  输出文件: {result.get('output_path', '')}")
    print(f"  时长: {result.get('duration', 0):.1f} 秒")


def start_scheduler(pet_type: str, times: int, modes: list, output_dir: str):
    """启动定时调度"""
    config = ScheduleConfig(
        pet_type=pet_type,
        times_per_day=times,
        modes=modes or ["xiangsheng", "singing", "soothing"],
        output_dir=output_dir,
    )

    scheduler = EntertainmentScheduler(config)

    # 自动生成时间表
    scheduler.set_times_per_day(times)

    # 添加回调 (互动完成时打印)
    def on_complete(record):
        print(f"\n[回调] 互动完成: {record['mode']}")
        print(f"  输出: {record['output_path']}")

    scheduler.add_callback(on_complete)

    # 启动
    scheduler.start()

    print(f"\n{'='*50}")
    print(f"  定时娱乐调度器已启动")
    print(f"{'='*50}")
    print(f"  宠物类型: {pet_type}")
    print(f"  每日次数: {times}")
    print(f"  时间表: {', '.join(config.fixed_times)}")
    print(f"  模式: {', '.join(config.modes)}")
    print(f"  下次运行: {scheduler.get_next_run()}")
    print(f"\n  按 Ctrl+C 停止\n")

    try:
        while True:
            import time
            time.sleep(60)
            print(f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] 等待中... 下次: {scheduler.get_next_run()}")
    except KeyboardInterrupt:
        scheduler.stop()
        scheduler.save_config()
        print("\n调度器已停止,配置已保存")


def start_multi_agent(pet_type: str, times: int, output_dir: str):
    """启动多智能体系统"""
    system = MultiAgentEntertainmentSystem(
        pet_type=pet_type,
        times_per_day=times,
        output_dir=output_dir,
    )

    # 先执行一次
    print("\n[首次运行]")
    system.run_once()

    # 打印状态
    status = system.get_status()
    print(f"\n{'='*50}")
    print(f"  多智能体系统状态")
    print(f"{'='*50}")
    print(f"  宠物类型: {status['pet_type']}")
    print(f"  宠物情绪: {status['pet_state']['mood']}")
    print(f"  能量值: {status['pet_state']['energy']:.1f}")
    print(f"  今日互动: {status['pet_state']['interactions_today']} 次")
    print(f"  偏好模式: {', '.join(status['pet_state']['preferred_modes'])}")

    # 启动定时
    system.start_scheduled()
    print(f"\n  定时调度已启动,每天 {times} 次")
    print(f"  按 Ctrl+C 停止\n")

    try:
        while True:
            import time
            time.sleep(60)
            status = system.get_status()
            print(f"[{__import__('datetime').datetime.now().strftime('%H:%M:%S')}] "
                  f"宠物: {status['pet_state']['mood']}, "
                  f"下次: {status['next_run']}")
    except KeyboardInterrupt:
        system.stop_scheduled()
        system.save_log()
        print("\n系统已停止,日志已保存")


def main():
    parser = argparse.ArgumentParser(description="逗猫逗狗娱乐系统")
    parser.add_argument("--mode", default="singing",
                        choices=["xiangsheng", "singing", "soothing", "teasing", "storytelling"],
                        help="娱乐模式")
    parser.add_argument("--pet", default="cat", choices=["cat", "dog"], help="宠物类型")
    parser.add_argument("--output", default="./output/entertainment", help="输出目录")
    parser.add_argument("--schedule", action="store_true", help="启动定时调度")
    parser.add_argument("--multi-agent", action="store_true", help="使用多智能体模式")
    parser.add_argument("--times", type=int, default=3, help="每日互动次数(1-10)")
    parser.add_argument("--modes", nargs="*", default=None, help="使用的模式列表")
    parser.add_argument("--list", action="store_true", help="列出所有模式")
    args = parser.parse_args()

    if args.list:
        list_modes(args.pet)
        return

    if args.multi_agent:
        start_multi_agent(args.pet, args.times, args.output)
    elif args.schedule:
        start_scheduler(args.pet, args.times, args.modes, args.output)
    else:
        play_once(args.mode, args.pet, args.output)


if __name__ == "__main__":
    main()
