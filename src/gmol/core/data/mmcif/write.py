from collections.abc import Sequence


def mmcif_bond_order(order: int) -> str:
    return {1: "sing", 2: "doub", 3: "trip", 4: "quad"}[order]


def mmcif_bool(pred: bool) -> str:
    return "Y" if pred else "N"


def _mmcif_escape(v: str) -> str:
    if any(c.isspace() for c in v) or v.startswith(("_", ";")):
        v = f"'{v}'"
    return v


def _assert_join(row: Sequence[str | int], num_fields: int) -> str:
    assert len(row) == num_fields
    return " ".join(_mmcif_escape(str(c)) for c in row)


def mmcif_write_block(
    block: str,
    fields: Sequence[str],
    data: Sequence[Sequence[str | int]],
) -> str:
    if not fields:
        raise ValueError("Fields must not be empty")

    if not data:
        return ""

    content = """
#
loop_
""" + "\n".join(f"_{block}.{field}" for field in fields)

    content += "\n" + "\n".join(_assert_join(row, len(fields)) for row in data)

    return content
