"""馬券の買い目を組み立てる中央モジュール。

2026-06-04 名古屋10R の検証で「モデルの馬選定は優秀だが着順が弱点」と判明。
3連単(着順厳密)は的中0%だが、3連複(順不同)はROI 122〜164%だった。
→ 3連複フォーメーションを正式サポートする。

入力は「LightGBM確率で降順ソート済み」の上位馬リスト [(馬番, p_top3), ...]。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BetPlan:
    """1レース分の各馬券種の買い目。"""
    axis: Optional[int] = None                 # 軸（確率1位）馬番
    ranked: list[int] = field(default_factory=list)   # 確率順 馬番
    sanrentan_nagashi: list[str] = field(default_factory=list)  # 3連単 軸1着流し 6点
    sanrentan_main: list[str] = field(default_factory=list)     # 3連単 本線2点
    sanrenpuku_box4: list[str] = field(default_factory=list)    # 3連複 上位4頭BOX 4点
    sanrenpuku_axis: list[str] = field(default_factory=list)    # 3連複 軸1頭-相手4頭 6点
    umaren_nagashi: list[str] = field(default_factory=list)     # 馬連 軸流し 4点


def _fmt3(a: int, b: int, c: int) -> str:
    return f"{a}-{b}-{c}"


def build_bets(ranked: list[tuple[int, float]]) -> BetPlan:
    """確率降順の (馬番, p_top3) リストから各買い目を生成。"""
    nums = [int(n) for n, _ in ranked]
    plan = BetPlan(ranked=nums)
    if len(nums) < 3:
        return plan
    plan.axis = nums[0]
    a = nums[0]
    rest = nums[1:5]  # 相手 最大4頭

    # --- 3連単 ---
    # 本線2点: 軸→2位→3位 と 2-3着入替
    if len(nums) >= 3:
        b, c = nums[1], nums[2]
        plan.sanrentan_main = [_fmt3(a, b, c), _fmt3(a, c, b)]
    # 軸1着流し: 相手から2頭を選び両順 (相手3頭=6点 / 4頭なら頭3点に絞る案内は呼び出し側)
    if len(rest) >= 3:
        partners3 = rest[:3]
        plan.sanrentan_nagashi = [
            _fmt3(a, x, y) for x, y in itertools.permutations(partners3, 2)
        ]

    # --- 3連複 ---
    # 上位4頭BOX (4C3 = 4点)
    if len(nums) >= 4:
        plan.sanrenpuku_box4 = [_fmt3(*c) for c in itertools.combinations(nums[:4], 3)]
    # 軸1頭 - 相手4頭から2頭 (4C2 = 6点)
    if len(rest) >= 4:
        plan.sanrenpuku_axis = [
            _fmt3(a, x, y) for x, y in itertools.combinations(rest, 2)
        ]

    # --- 馬連 軸流し (相手4頭 = 4点) ---
    if len(rest) >= 1:
        plan.umaren_nagashi = [f"{a}-{x}" for x in rest]

    return plan


def rationale(plan: BetPlan, names: Optional[dict[int, str]] = None,
              probs: Optional[dict[int, float]] = None) -> str:
    """買い目の根拠テキストを生成。"""
    names = names or {}
    probs = probs or {}
    a = plan.axis
    if a is None:
        return "出走馬不足"
    nm = lambda n: names.get(n, "")
    pr = lambda n: f"{probs.get(n, 0):.0%}" if n in probs else "—"
    rest = plan.ranked[1:5]
    rel = "・".join(f"{n}番{nm(n)}({pr(n)})" for n in rest[:3])
    return (
        f"【軸】{a}番 {nm(a)}（3着内確率 {pr(a)}）。LightGBMで最上位。\n"
        f"【相手】{rel} を中心に。\n"
        f"【推奨】着順を厳密に当てる3連単は的中が薄いため、"
        f"順不同で取れる3連複を本線に。検証(名古屋10R)で3連複は回収率プラス、3連単は0%だった。\n"
        f"・3連複 上位4頭BOX(4点): {' / '.join(plan.sanrenpuku_box4)}\n"
        f"・3連複 軸1頭-相手4頭(6点): {' / '.join(plan.sanrenpuku_axis)}\n"
        f"・3連単 本線(2点): {' / '.join(plan.sanrentan_main)}"
    )


if __name__ == "__main__":
    demo = [(8, 0.624), (11, 0.497), (12, 0.427), (2, 0.328), (1, 0.236)]
    p = build_bets(demo)
    print("軸:", p.axis)
    print("3連単本線:", p.sanrentan_main)
    print("3連複BOX4:", p.sanrenpuku_box4)
    print("3連複軸流し6:", p.sanrenpuku_axis)
    print("馬連流し:", p.umaren_nagashi)
    print("---")
    print(rationale(p, names={8: "ドラゴンガール"}, probs=dict(demo)))
