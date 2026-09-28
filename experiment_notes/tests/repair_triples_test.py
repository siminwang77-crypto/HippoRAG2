def repair_triple(triple):
    """
    将 LLM 输出的非标准 triple 修复为严格的：
    [subject, predicate, object]

    规则：
    - 3 元组：直接保留
    - 4/5+ 元组：把第 3 项之后的内容合并到 object
    - 2 元组：暂不自动修复，返回 None
    """

    if not isinstance(triple, (list, tuple)):
        return None

    triple = [str(x).strip() for x in triple]

    # 正常三元组
    if len(triple) == 3:
        return triple

    # 两元组：暂时不自动猜语义
    if len(triple) == 2:
        return None

    # 四元组、五元组及更长：
    # [S, P, O, qualifier...] -> [S, P, O qualifier...]
    if len(triple) > 3:
        return [
            triple[0],
            triple[1],
            " ".join(x for x in triple[2:] if x)
        ]

    return None


tests = [
    ['Rhythm', 'calls', 'Dan Kavanagh',
     'hard hitting rock fiend juggling two intense gigs'],

    ["Her Majesty's Government", 'is', 'led by', 'Prime Minister'],

    ['Alaskan Sami', 'left Alaska after selling herds'],

    ['Civil War', 'had port and riverboat landing abandoned'],

    ['United States', 'had', 'prisons since', '1500s'],

    ['1998 ICC KnockOut Trophy', 'took place from',
     '24 October', 'to', '2 November 1998'],

    ['Raymond', 'is a summer recreation area'],

    ['South Africa', 'defeated', 'West Indies', 'in the final'],

    ['Ray Ricardo Wynter', 'played cricket from',
     '1975', 'until', '1982'],
]


for triple in tests:
    repaired = repair_triple(triple)

    print("=" * 70)
    print("原始:", triple)
    print("修复:", repaired)

    if repaired is None:
        print("状态: NEED_LLM_REPAIR")
    elif len(repaired) == 3:
        print("状态: VALID")
    else:
        print("状态: ERROR")
