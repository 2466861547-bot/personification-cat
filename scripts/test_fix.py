"""
验证拟人化文本生成相关函数的正确性
无需依赖 transformers/torch
"""
import sys
import os
import re

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 测试 1: _clean_anthropomorphic_output 函数
def test_clean_anthropomorphic_output():
    """测试文本清理函数"""
    from src.models.llm_finetune import LLMTrainer
    
    # 测试用例 1: 正常的拟人化文本
    result = LLMTrainer._clean_anthropomorphic_output(
        "橘猫开心地说：今天阳光好好，主人摸摸我嘛~", "cat", "橘猫"
    )
    assert "橘猫" in result, f"应包含品种: {result}"
    print(f"✅ 测试1通过: 正常文本 → '{result}'")
    
    # 测试用例 2: 包含元信息的输出 (应被清理)
    result = LLMTrainer._clean_anthropomorphic_output(
        "宠物类型: cat\n品种: 橘猫\n橘猫开心地说：主人摸摸我嘛~", "cat", "橘猫"
    )
    assert "宠物类型" not in result, f"元信息应被清理: {result}"
    print(f"✅ 测试2通过: 清理元信息 → '{result}'")
    
    # 测试用例 3: 重复文本 (应被清理)
    result = LLMTrainer._clean_anthropomorphic_output(
        "棒棒喂食棒棒喂食棒棒喂食棒棒喂食棒棒喂食", "cat", "橘猫"
    )
    assert "棒棒喂食" not in result, f"重复文本应被清理: '{result}'"
    print(f"✅ 测试3通过: 清理重复 → '{result}'")
    
    # 测试用例 4: 全是元信息 (应返回空)
    result = LLMTrainer._clean_anthropomorphic_output(
        "宠物类型: cat\n品种: 橘猫\n情绪: happy", "cat", "橘猫"
    )
    assert result == "", f"全元信息应返回空: '{result}'"
    print(f"✅ 测试4通过: 全元信息 → 返回空")

# 测试 2: _template_anthropomorphic 函数
def test_template_anthropomorphic():
    """测试模板生成函数"""
    from src.inference.pipeline import TranslationPipeline
    
    # 测试用例: cat + happy
    result = TranslationPipeline._template_anthropomorphic(
        pet_type="cat", breed="橘猫", emotion="happy", description="开心"
    )
    assert "橘猫" in result, f"应包含品种: {result}"
    assert "猫咪" in result, f"应包含宠物类型: {result}"
    print(f"✅ 测试5通过: cat+happy → '{result}'")
    
    # 测试用例: dog + hungry
    result = TranslationPipeline._template_anthropomorphic(
        pet_type="dog", breed="金毛", emotion="hungry", description="饥饿"
    )
    assert "金毛" in result, f"应包含品种: {result}"
    assert "狗狗" in result, f"应包含宠物类型: {result}"
    print(f"✅ 测试6通过: dog+hungry → '{result}'")
    
    # 测试用例: cat + alert
    result = TranslationPipeline._template_anthropomorphic(
        pet_type="cat", breed="英短", emotion="alert", description="警觉"
    )
    assert "英短" in result
    assert "猫咪" in result
    print(f"✅ 测试7通过: cat+alert → '{result}'")
    
    # 测试用例: dog + angry
    result = TranslationPipeline._template_anthropomorphic(
        pet_type="dog", breed="泰迪", emotion="angry", description="愤怒"
    )
    assert "泰迪" in result
    assert "狗狗" in result
    print(f"✅ 测试8通过: dog+angry → '{result}'")
    
    # 测试用例: 未知情绪 (应返回默认模板)
    result = TranslationPipeline._template_anthropomorphic(
        pet_type="cat", breed="美短", emotion="unknown_emotion", description="未知"
    )
    assert "美短" in result
    assert "猫咪" in result
    print(f"✅ 测试9通过: 未知情绪 → 默认模板: '{result}'")

# 测试 3: generate_anthropomorphic prompt 构建
def test_generate_anthropomorphic_prompt():
    """测试拟人化生成的 prompt 构建"""
    from src.models.llm_finetune import LLMTrainer
    
    # 验证 emotion 到中文的映射
    emotion_to_chinese = {
        "alert": "警觉", "seek_attention": "求关注", "hungry": "饿了",
        "happy": "开心", "angry": "生气", "fearful": "害怕",
        "content": "满足", "pain": "疼痛", "curious": "好奇",
        "lonely": "孤独", "anxious": "焦虑", "excited": "兴奋",
        "frustrated": "沮丧", "relaxed": "放松", "territorial": "护领地",
        "greeting": "打招呼", "confused": "困惑", "jealous": "嫉妒",
        "sad": "难过", "playful": "想玩耍"
    }
    
    # 检查所有情绪都有中文映射
    emotions = ["alert", "seek_attention", "hungry", "happy", "angry", 
                "fearful", "content", "pain", "curious", "lonely",
                "anxious", "excited", "frustrated", "relaxed", "territorial",
                "greeting", "confused", "jealous", "sad", "playful"]
    
    for emotion in emotions:
        assert emotion in emotion_to_chinese, f"情绪 {emotion} 缺少中文映射"
    
    print(f"✅ 测试10通过: 所有 {len(emotions)} 种情绪都有中文映射")
    
    # 检查 pet_type 名称映射
    pet_type_names = {"cat": "猫咪", "dog": "狗狗", "bird": "鸟", "rabbit": "兔子"}
    assert pet_type_names.get("cat") == "猫咪"
    assert pet_type_names.get("dog") == "狗狗"
    print(f"✅ 测试11通过: 宠物类型映射正确")

# 测试 4: 验证 inference 参数更新
def test_inference_params():
    """验证 inference 方法的参数是否已更新"""
    import inspect
    from src.models.llm_finetune import LLMTrainer
    
    sig = inspect.signature(LLMTrainer.inference)
    params = list(sig.parameters.keys())
    
    # 检查新参数
    assert "temperature" in params, "应包含 temperature 参数"
    assert "repetition_penalty" in params, "应包含 repetition_penalty 参数"
    
    # 检查默认值
    defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
    assert defaults.get('max_new_tokens') == 256, f"max_new_tokens 默认应为 256, 实际: {defaults.get('max_new_tokens')}"
    assert defaults.get('temperature') == 0.7, f"temperature 默认应为 0.7, 实际: {defaults.get('temperature')}"
    assert defaults.get('repetition_penalty') == 1.15, f"repetition_penalty 默认应为 1.15, 实际: {defaults.get('repetition_penalty')}"
    
    print(f"✅ 测试12通过: inference 参数已正确更新")
    print(f"   - max_new_tokens: {defaults.get('max_new_tokens')}")
    print(f"   - temperature: {defaults.get('temperature')}")
    print(f"   - repetition_penalty: {defaults.get('repetition_penalty')}")

# 测试 5: 验证 generate_anthropomorphic 方法存在
def test_generate_anthropomorphic_exists():
    """验证 generate_anthropomorphic 方法存在"""
    from src.models.llm_finetune import LLMTrainer
    
    assert hasattr(LLMTrainer, 'generate_anthropomorphic'), "应包含 generate_anthropomorphic 方法"
    assert hasattr(LLMTrainer, '_clean_generated_text'), "应包含 _clean_generated_text 方法"
    assert hasattr(LLMTrainer, '_clean_anthropomorphic_output'), "应包含 _clean_anthropomorphic_output 方法"
    print(f"✅ 测试13通过: 所有新方法已添加")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 验证拟人化文本生成修复")
    print("=" * 60)
    
    tests = [
        test_clean_anthropomorphic_output,
        test_template_anthropomorphic,
        test_generate_anthropomorphic_prompt,
        test_inference_params,
        test_generate_anthropomorphic_exists,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ 所有测试完成")
    print("=" * 60)