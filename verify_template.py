#!/usr/bin/env python3
"""模板系统完整验证 v2 - 含品种注入"""
import random

BREED_HINTS = {
    "橘猫": ["橘胖", "小胖橘"],
    "英短": ["英短", "蓝胖子"],
    "美短": ["美短", "花纹"],
    "布偶": ["布偶", "小仙女"],
    "柯基": ["柯基", "小短腿"],
    "金毛": ["金毛", "大暖男"],
    "泰迪": ["泰迪", "小机灵"],
    "哈士奇": ["二哈", "撒手没"],
}

EMOTION_LABELS = [
    "alert", "seek_attention", "hungry", "happy", "angry",
    "fearful", "content", "pain", "curious", "lonely", "anxious",
    "excited", "frustrated", "relaxed", "territorial", "greeting",
    "confused", "jealous", "sad", "playful",
]

def _template_anthropomorphic(pet_type, breed, emotion, description):
    hints = BREED_HINTS.get(breed, [breed])

    cat_templates = {
        "hungry": [f"{hints[0]}：铲屎官，我的饭碗空啦~快喂我！", f"喵~ 肚子咕咕叫了~"],
        "happy": [f"喵呜~ 今天阳光好好~", f"{hints[0]}翻出肚皮：摸摸我！"],
        "angry": [f"哈！别碰我！", f"{hints[0]}甩尾巴：我生气了！"],
        "sad": [f"喵...好难过...", f"{hints[0]}安静地走开了..."],
        "playful": [f"来呀来呀！玩逗猫棒~", f"{hints[0]}蹦跶起来！"],
        "greeting": [f"主人回来啦！", f"{hints[0]}跑到门口迎接~"],
    }

    dog_templates = {
        "hungry": [f"汪！该开饭啦！", f"{hints[0]}叼着饭盆跑过来~"],
        "happy": [f"汪汪汪！主人回来啦！", f"{hints[0]}摇着尾巴转圈！"],
        "angry": [f"汪！别碰我玩具！", f"{hints[0]}低吼：走开！"],
        "sad": [f"呜...主人别难过...", f"{hints[0]}安静地趴在你脚边~"],
        "playful": [f"来玩拔河！", f"{hints[0]}叼着球跑过来！"],
        "greeting": [f"主人！你终于回来啦！", f"{hints[0]}扑过来舔你！"],
    }

    if pet_type == "cat":
        templates = cat_templates
    elif pet_type == "dog":
        templates = dog_templates
    else:
        templates = {}

    if not templates:
        templates = cat_templates

    emotion_pool = templates.get(emotion)
    if emotion_pool:
        chosen = random.choice(emotion_pool)
        if breed and breed != "通用" and breed not in chosen:
            hint = hints[0] if hints else breed
            if hint not in chosen:
                chosen = f"{hint}小声说: " + chosen
        return chosen

    # 智能兜底
    emotion_category = {
        "happy": "happy", "excited": "happy", "playful": "playful",
        "angry": "angry", "sad": "sad", "hungry": "hungry",
    }
    mapped = emotion_category.get(emotion, "happy")
    if mapped in templates:
        chosen = random.choice(templates[mapped])
        if breed and breed != "通用" and breed not in chosen:
            hint = hints[0] if hints else breed
            if hint not in chosen:
                chosen = f"{hint}小声说: " + chosen
        return chosen

    pet_nice = "小猫咪" if pet_type == "cat" else "小狗狗"
    fallback = f"{pet_nice}想说：主人，我在这里哦~"
    if breed and breed != "通用":
        hint = hints[0] if hints else breed
        fallback = f"{hint}：{fallback}"
    return fallback


print("=" * 70)
print("模板系统完整验证 v2 - 含品种注入")
print("=" * 70)

errors = []

# 测试品种注入
print("\n📋 测试品种名称注入 (每种组合跑 10 次)")
test_breeds = [
    ("cat", "橘猫", "hungry", "橘胖"),
    ("cat", "英短", "happy", "英短"),
    ("cat", "布偶", "sad", "布偶"),
    ("dog", "金毛", "happy", "金毛"),
    ("dog", "柯基", "hungry", "柯基"),
    ("dog", "哈士奇", "playful", "二哈"),
]

for pet, breed, emo, expected in test_breeds:
    for i in range(10):
        result = _template_anthropomorphic(pet, breed, emo, "")
        if expected not in result and breed not in result:
            errors.append(f"[{pet}] {breed}/{emo}: 缺少品种标识 → '{result}'")
            print(f"  ❌ {breed}/{emo} 第{i+1}次: {result}")
            break
    else:
        print(f"  ✅ {breed}/{emo}: 10次均包含品种标识")

# 测试通用品种
print("\n📋 测试通用品种 (无特定品种)")
for pet in ["cat", "dog"]:
    result = _template_anthropomorphic(pet, "通用", "happy", "")
    print(f"  {pet}/通用/happy: {result}")

# 测试所有情绪 (猫狗各一次)
print("\n📋 所有情绪测试")
for pet_type in ["cat", "dog"]:
    breed = "橘猫" if pet_type == "cat" else "金毛"
    for emotion in EMOTION_LABELS:
        result = _template_anthropomorphic(pet_type, breed, emotion, "")
        if not result or len(result) < 3:
            errors.append(f"[{pet_type}] {emotion}: 返回空 → '{result}'")
        print(f"  {pet_type}/{emotion}: {result[:50]}")

print("\n" + "=" * 70)
if errors:
    print(f"❌ 发现 {len(errors)} 个问题:")
    for e in errors:
        print(f"   - {e}")
else:
    print("✅ 所有测试通过！模板系统正常工作！")
print("=" * 70)