"""#5 reward-guided fine-tuning: reward math + MatterGen output extraction."""

import zipfile

from stealth.discovery.extract_cifs import extract
from stealth.discovery.rl_finetune import combine_reward


def test_impractical_gets_floor_reward():
    assert combine_reward(0.1, 0.9, 5.0, 0.9, practical=False) == -1.0


def test_reward_prefers_stealthy_light_durable_manufacturable():
    base = dict(practical=True)
    # lower objective (stealthier) -> higher reward
    assert combine_reward(0.2, 0.9, 5, 0.9, **base) > combine_reward(0.8, 0.9, 5, 0.9, **base)
    # lighter -> higher
    assert combine_reward(0.3, 0.9, 3, 0.9, **base) > combine_reward(0.3, 0.9, 11, 0.9, **base)
    # more durable -> higher
    assert combine_reward(0.3, 0.9, 5, 0.95, **base) > combine_reward(0.3, 0.9, 5, 0.2, **base)
    # more manufacturable -> higher
    assert combine_reward(0.3, 0.95, 5, 0.9, **base) > combine_reward(0.3, 0.4, 5, 0.9, **base)


def test_extract_cifs_from_zip(tmp_path):
    z = tmp_path / "generated_crystals_cif.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("gen_0.cif", "# cif 0")
        zf.writestr("sub/gen_1.cif", "# cif 1")
        zf.writestr("notes.txt", "ignore")
    n = extract(str(tmp_path), prefix="radar")
    assert n == 2
    names = sorted(p.name for p in (tmp_path / "cifs").glob("*.cif"))
    assert names == ["radar_gen_0.cif", "radar_gen_1.cif"]
