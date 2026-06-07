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


def sanrentan_5(nums: list[int]) -> list[str]:
    """3連単5点 1着固定流し（軸=確率1位を1着固定）。"""
    if len(nums) < 4:
        return []
    a, b, c, d = nums[0], nums[1], nums[2], nums[3]
    return [f"{a}-{b}-{c}", f"{a}-{c}-{b}", f"{a}-{b}-{d}", f"{a}-{d}-{b}", f"{a}-{c}-{d}"]


def sanrenpuku_5(nums: list[int]) -> list[str]:
    """3連複5点 = 上位4頭BOX(4点) + 軸-2位-5位(1点)。各点は馬番昇順（3連複の標準表記）。"""
    if len(nums) < 4:
        return []
    box = ["-".join(map(str, sorted(c))) for c in itertools.combinations(nums[:4], 3)]
    if len(nums) >= 5:
        box.append("-".join(map(str, sorted((nums[0], nums[1], nums[4])))))
    return box


def picks_5(plan: BetPlan, bet_type: str) -> list[str]:
    """馬券種に応じた「5点」を返す。bet_type: 'sanrentan' | 'sanrenpuku'。"""
    nums = plan.ranked
    return sanrentan_5(nums) if bet_type == "sanrentan" else sanrenpuku_5(nums)


@dataclass
class Formation:
    name: str
    picks: list[str]
    n_points: int
    axis: Optional[int]
    axis_conf: float          # 軸の信頼度 = 較正確率の1位-2位の差
    note: str = ""
    skip: bool = False        # 団子レース=見送り推奨


def recommend_sanrenpuku(pairs: list[tuple[int, float]]) -> Formation:
    """較正確率(pairs=(馬番, p_cal) 降順)から、軸の信頼度で買い方を可変。

    - 軸断然 (gap≥0.15): 軸1頭固定 - 相手5頭ながし (C(5,2)=10点)
    - 標準   (0.05≤gap<0.15): 上位5頭BOX (10点)
    - 団子   (gap<0.05): 見送り推奨。参考に上位5頭BOXを提示
    相手を5頭目まで広げるのが基本（名古屋検証で的中率が大きく改善）。
    """
    nums = [int(n) for n, _ in pairs]
    ps = [float(p) for _, p in pairs]
    if len(nums) < 4:
        return Formation("点数不足", [], 0, nums[0] if nums else None, 0.0,
                         "出走/取得が少なく組成不可", skip=True)
    axis = nums[0]
    gap = ps[0] - ps[1]

    def box(horses: list[int]) -> list[str]:
        return ["-".join(map(str, sorted(c))) for c in itertools.combinations(horses, 3)]

    if gap >= 0.15:
        partners = nums[1:6]                       # 相手 最大5頭
        picks = ["-".join(map(str, sorted((axis, x, y))))
                 for x, y in itertools.combinations(partners, 2)]
        return Formation("軸1頭固定-相手5頭ながし", picks, len(picks), axis, gap,
                         f"軸{axis}が断然(確率差{gap:.0%})。軸を固定し相手を広く取る型。")
    elif gap >= 0.05:
        picks = box(nums[:5])                       # 上位5頭BOX
        return Formation("上位5頭BOX", picks, len(picks), axis, gap,
                         f"上位拮抗(確率差{gap:.0%})。5頭BOXで取りこぼしを防ぐ。")
    else:
        picks = box(nums[:5])
        return Formation("上位5頭BOX(見送り推奨)", picks, len(picks), axis, gap,
                         f"超団子(確率差{gap:.0%})。妙味薄く見送り推奨。買うなら5頭BOX。",
                         skip=True)


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
