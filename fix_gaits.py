# fix_gaits.py  — run this from your project root
# Fixes all gait files to be robot-agnostic by:
#   1. Replacing hardcoded startswith leg-side checks with self.lateral_sign()
#   2. Replacing hardcoded startswith fore/aft checks with self.fore_aft_sign()
#   3. Replacing manual __init__ boilerplate with super().__init__()

import os
import re
from pathlib import Path

GAITS_DIR = Path("gaits")

# ── Patterns to replace ───────────────────────────────────────────────────────

# Matches any of these patterns for left-side detection:
#   leg_name.startswith('FL') or leg_name.startswith('RL')
#   leg_name.startswith('FL') or leg_name.startswith('LH')
#   etc. — any startswith combo used for lateral side logic
LATERAL_SIDE_PATTERNS = [
    # if leg_name.startswith('X') or leg_name.startswith('Y'):
    #     lateral_sign = 1.0 / -1.0
    # else:
    #     lateral_sign = -1.0 / 1.0
    (
        re.compile(
            r"if leg_name\.startswith\(['\"][A-Z_]+['\"]\).*?lateral_sign\s*=\s*[-0-9.]+.*?"
            r"(?:else:.*?lateral_sign\s*=\s*[-0-9.]+)",
            re.DOTALL
        ),
        "lateral_sign = self.lateral_sign(leg_name)"
    ),
]

# Matches fore/aft checks like:
#   if leg_name.startswith('FL') or leg_name.startswith('FR'):
#       fore_sign = 1.0
#   else:
#       fore_sign = -1.0
FORE_AFT_PATTERNS = [
    (
        re.compile(
            r"if leg_name\.startswith\(['\"][A-Z_]+['\"]\).*?fore_sign\s*=\s*[-0-9.]+.*?"
            r"(?:else:.*?fore_sign\s*=\s*[-0-9.]+)",
            re.DOTALL
        ),
        "fore_sign = self.fore_aft_sign(leg_name)"
    ),
]

# ── Simple line-by-line fixes (safer for most files) ─────────────────────────

LEFT_PREFIXES  = {"FL", "RL", "LF", "LH"}
RIGHT_PREFIXES = {"FR", "RR", "RF", "RH"}
FRONT_PREFIXES = {"FL", "FR", "LF", "RF"}
REAR_PREFIXES  = {"RL", "RR", "LH", "RH"}


def is_lateral_sign_block(lines, i):
    """Detect a lateral_sign if/else block starting at line i."""
    block = "\n".join(lines[i:i+6])
    return "lateral_sign" in block and "startswith" in block


def is_fore_aft_sign_block(lines, i):
    """Detect a fore_sign / fore_aft_sign if/else block starting at line i."""
    block = "\n".join(lines[i:i+6])
    return ("fore_sign" in block or "fore_aft_sign" in block) and "startswith" in block


def fix_file(path: Path) -> bool:
    """
    Fix a single gait file. Returns True if any changes were made.
    """
    original = path.read_text()
    lines = original.splitlines()
    new_lines = []
    i = 0
    changed = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── Detect: if leg_name.startswith(...) lateral_sign block ──────────
        if (
            stripped.startswith("if leg_name.startswith(") and
            i + 3 < len(lines)
        ):
            block = "\n".join(lines[i:i+6])

            if "lateral_sign" in block:
                # Find indentation
                indent = len(line) - len(line.lstrip())
                ind = " " * indent
                new_lines.append(f"{ind}lateral_sign = self.lateral_sign(leg_name)")
                # Skip until end of if/else block (consume up to 6 lines)
                j = i
                while j < min(i + 8, len(lines)):
                    l = lines[j].strip()
                    if l.startswith("lateral_sign") or l.startswith("if ") or l.startswith("else"):
                        j += 1
                    else:
                        break
                i = j
                changed = True
                continue

            elif "fore_sign" in block or "fore_aft_sign" in block:
                indent = len(line) - len(line.lstrip())
                ind = " " * indent
                new_lines.append(f"{ind}fore_sign = self.fore_aft_sign(leg_name)")
                j = i
                while j < min(i + 8, len(lines)):
                    l = lines[j].strip()
                    if l.startswith("fore_sign") or l.startswith("fore_aft_sign") or l.startswith("if ") or l.startswith("else"):
                        j += 1
                    else:
                        break
                i = j
                changed = True
                continue

        new_lines.append(line)
        i += 1

    result = "\n".join(new_lines)
    if changed:
        path.write_text(result)
        return True
    return False


def fix_super_init(path: Path) -> bool:
    """
    Replace manual __init__ boilerplate with super().__init__() call.
    Only applies if the class inherits from BaseMotionGenerator and
    manually sets self.t, self.root_pos etc. without calling super().
    """
    text = path.read_text()

    # Only process if it's a BaseMotionGenerator subclass
    if "BaseMotionGenerator" not in text:
        return False

    # Only if super().__init__ is NOT already called
    if "super().__init__" in text:
        return False

    # Find __init__ that manually sets self.leg_names and self.t etc.
    if "self.leg_names = leg_names" not in text:
        return False
    if "self.t = 0.0" not in text:
        return False

    lines = text.splitlines()
    new_lines = []
    in_init = False
    init_indent = ""
    boilerplate_keys = {
        "self.leg_names = leg_names",
        "self.t = 0.0",
        "self.root_pos = np.zeros(3)",
        "self.root_pos = np.array([0.0, 0.0, 0.0])",
        "self.root_quat = np.array([1.0, 0.0, 0.0, 0.0])",
        "self.vel_world = np.zeros(3)",
        "self.omega_world = np.zeros(3)",
    }
    super_inserted = False
    changed = False

    for line in lines:
        stripped = line.strip()

        # Detect def __init__
        if re.match(r"def __init__\(self", stripped):
            in_init = True
            init_indent = " " * (len(line) - len(line.lstrip()) + 4)
            new_lines.append(line)
            continue

        if in_init and not super_inserted:
            # Insert super().__init__ after the def line, before other setup
            if stripped and not stripped.startswith("#"):
                # Check if this is a real body line (not just a docstring open)
                if not stripped.startswith('"""') and not stripped.startswith("'''"):
                    new_lines.append(f"{init_indent}super().__init__(initial_foot_positions_body, freq=self.freq if hasattr(self, 'freq') else 1.0)")
                    super_inserted = True
                    changed = True

        # Skip boilerplate lines that super().__init__ now handles
        if super_inserted and any(stripped.startswith(k) for k in boilerplate_keys):
            changed = True
            continue

        new_lines.append(line)

    if changed:
        path.write_text("\n".join(new_lines))
        return True
    return False


def main():
    fixed = []
    skipped = []

    all_py = list(GAITS_DIR.rglob("*.py"))
    print(f"Found {len(all_py)} Python files in gaits/")

    for path in sorted(all_py):
        if path.name in ("__init__.py", "base.py"):
            continue
        try:
            c1 = fix_file(path)
            c2 = fix_super_init(path)
            if c1 or c2:
                fixed.append(str(path))
            else:
                skipped.append(str(path))
        except Exception as e:
            print(f"  ERROR on {path}: {e}")

    print(f"\n✅ Fixed {len(fixed)} files:")
    for f in fixed:
        print(f"   {f}")
    print(f"\n⏭  Skipped (no changes needed): {len(skipped)} files")


if __name__ == "__main__":
    main()