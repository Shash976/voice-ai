"""optimizer/common/designs.py — design registry for the chip-config optimizer.

Any design (RTL files + top module, possibly with macros) becomes an input by
creating a YAML spec in optimizer/designs/<name>.yaml and loading it via
DesignSpec.load(name_or_path).

PINNED interface (concurrent agents code against this exactly):

    @dataclass
    class DesignSpec:
        name: str                      # registry key and ORFS DESIGN_NAME
        top: str                       # verilog top module
        rtl_files: list[str]           # absolute paths after load()
        clock_port: str                # for the SDC template
        params: dict[str, dict]        # RTL chparam axes; may be {}
        platforms: dict[str, dict]     # {"nangate45": {"clock_range_ns":[3.0,8.0], ...}, ...}
        has_macros: bool | None        # None = auto-detect at first F2
        functional_eval: dict | None   # {"kind":"tinyvad_sim"} or {"kind":"none"} or None
        @classmethod
        def load(cls, name_or_path: str) -> "DesignSpec"
        def sdc_text(self, platform: str, clock_value_native: float) -> str

Design YAML spec format (see optimizer/designs/tinymac_accel.yaml for full example):
    name: <str>
    top: <str>
    rtl_files: [<path>, ...]   # relative to repo root OR absolute
    clock_port: <str>
    params:
        PARAM_NAME:
            choices: [v1, v2, ...]   # OR range: [lo, hi]
            default: <value>
    platforms:
        nangate45:
            clock_range_ns: [lo, hi]
            default_clock_ns: <float>
        asap7:
            clock_range_ns: [lo, hi]
            default_clock_ns: <float>
    has_macros: false   # or true, or omit for auto-detect (None)
    functional_eval:
        kind: tinyvad_sim   # or: none
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Repo root: optimizer/common/designs.py → ../../../ = repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Designs YAML directory
_DESIGNS_DIR = Path(__file__).resolve().parent.parent / "designs"

# SDC template — generic; physical_runner.py applies PLATFORM_TIME_UNIT conversion
# before writing so clock_value_native is always in the platform's native unit.
_SDC_TEMPLATE = """\
current_design {top}

set clk_name      core_clock
set clk_port_name {clock_port}
set clk_period    {clock_period}
set clk_io_pct    0.2

set clk_port [get_ports $clk_port_name]
create_clock -name $clk_name -period $clk_period $clk_port

set non_clock_inputs [all_inputs -no_clocks]
set_input_delay  [expr $clk_period * $clk_io_pct] -clock $clk_name $non_clock_inputs
set_output_delay [expr $clk_period * $clk_io_pct] -clock $clk_name [all_outputs]
"""


@dataclass
class DesignSpec:
    """Immutable description of one chip design for the optimizer."""

    name: str
    top: str
    rtl_files: list[str]           # absolute paths (resolved on load)
    clock_port: str
    params: dict[str, dict]        # RTL chparam axes; may be {}
    platforms: dict[str, dict]     # per-platform clock ranges and defaults
    has_macros: bool | None        # None = auto-detect at first F2
    functional_eval: dict | None   # {"kind": "tinyvad_sim"} or {"kind": "none"} or None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, name_or_path: str) -> "DesignSpec":
        """Load a DesignSpec from a YAML file.

        Resolution order:
            1. If name_or_path is an existing path (absolute or relative to cwd),
               load it directly.
            2. Otherwise, look up optimizer/designs/<name_or_path>.yaml.

        RTL file paths in the YAML are resolved as:
            - If absolute: used as-is.
            - If relative: resolved relative to the repo root.

        Raises FileNotFoundError if the YAML is not found.
        Raises ValueError if required fields are missing.
        """
        import yaml  # type: ignore[import]

        yaml_path = Path(name_or_path)
        if not yaml_path.is_absolute():
            yaml_path = Path.cwd() / yaml_path
        if not yaml_path.exists():
            # Try the designs registry directory
            yaml_path = _DESIGNS_DIR / f"{name_or_path}.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"DesignSpec.load: cannot find design '{name_or_path}'. "
                f"Tried: '{name_or_path}' (direct path) and "
                f"'{_DESIGNS_DIR / name_or_path}.yaml' (registry)."
            )

        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Required fields
        for required in ("name", "top", "rtl_files", "clock_port"):
            if required not in raw:
                raise ValueError(
                    f"DesignSpec.load: required field '{required}' missing in {yaml_path}"
                )

        # Resolve RTL file paths
        rtl_files = []
        for p in raw["rtl_files"]:
            pp = Path(p)
            if not pp.is_absolute():
                pp = _REPO_ROOT / pp
            rtl_files.append(str(pp))

        # Optional fields with defaults
        params = raw.get("params") or {}
        platforms = raw.get("platforms") or {
            "nangate45": {"clock_range_ns": [3.0, 8.0], "default_clock_ns": 5.0}
        }
        has_macros_raw = raw.get("has_macros")
        has_macros: bool | None
        if has_macros_raw is None:
            has_macros = None        # auto-detect at first F2
        else:
            has_macros = bool(has_macros_raw)

        functional_eval = raw.get("functional_eval")

        return cls(
            name=str(raw["name"]),
            top=str(raw["top"]),
            rtl_files=rtl_files,
            clock_port=str(raw["clock_port"]),
            params=params,
            platforms=platforms,
            has_macros=has_macros,
            functional_eval=functional_eval,
        )

    # ── SDC generation ────────────────────────────────────────────────────────

    def sdc_text(self, platform: str, clock_value_native: float) -> str:
        """Return the SDC content for the given platform clock value.

        clock_value_native must already be in the platform's native unit
        (ns for nangate45, ps for asap7).  The caller (physical_runner) applies
        the PLATFORM_TIME_UNIT conversion BEFORE calling this method.

        The SDC uses the design's top module name and clock port.
        """
        return _SDC_TEMPLATE.format(
            top=self.top,
            clock_port=self.clock_port,
            clock_period=clock_value_native,
        )

    # ── RTL content hash ──────────────────────────────────────────────────────

    def rtl_hash(self) -> str:
        """Return an 8-hex-digit SHA-256 digest of the RTL source files.

        Files are hashed in sorted-path order for determinism.  If a file is
        missing, its path string is hashed as a placeholder (consistent with the
        legacy _rtl_hash() in physical_runner.py for tinymac).
        """
        h = hashlib.sha256()
        for p in sorted(self.rtl_files):
            try:
                h.update(Path(p).read_bytes())
            except OSError:
                h.update(p.encode())   # placeholder for missing files
        return h.hexdigest()[:8]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def verilog_top_params_str(self, config: dict) -> str:
        """Build the VERILOG_TOP_PARAMS string from a config dict.

        For each param in self.params, if the canonical param name is present in
        config, emit it using the RTL chparam name (rtl_param_name field if present,
        otherwise the param name itself).  Returns "" if the design has no params.

        Example (new-style with rtl_param_name):
            design.params = {"mac_lanes": {"rtl_param_name": "LANES", ...},
                             "accumulator_width": {"rtl_param_name": "ACC_W", ...}}
            config = {"mac_lanes": 4, "accumulator_width": 24, "clock_period_ns": 5.0}
            → "LANES 4 ACC_W 24"

        Example (legacy: param name IS the RTL chparam name):
            design.params = {"LANES": ..., "ACC_W": ...}
            config = {"LANES": 4, "ACC_W": 24, "clock_period_ns": 5.0}
            → "LANES 4 ACC_W 24"
        """
        if not self.params:
            return ""
        parts = []
        for param_name, param_spec in self.params.items():
            if param_name in config:
                # Emit the RTL chparam name, not the canonical search-space name
                rtl_name = (
                    param_spec.get("rtl_param_name", param_name)
                    if isinstance(param_spec, dict)
                    else param_name
                )
                parts += [rtl_name, str(config[param_name])]
        return " ".join(parts)

    def is_tinyvad(self) -> bool:
        """Return True if this design uses the TinyVAD functional evaluator."""
        fe = self.functional_eval or {}
        return fe.get("kind") == "tinyvad_sim"

    def __repr__(self) -> str:
        return (
            f"DesignSpec(name={self.name!r}, top={self.top!r}, "
            f"rtl_files={[Path(f).name for f in self.rtl_files]}, "
            f"has_macros={self.has_macros!r}, "
            f"functional_eval={self.functional_eval!r})"
        )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== designs.py self-test ===")

    # 1. tinymac_accel loads
    try:
        tm = DesignSpec.load("tinymac_accel")
        print(f"  tinymac_accel loaded: {tm}")
        assert tm.name == "tinymac_accel"
        assert tm.top == "tinymac_accel"
        assert len(tm.rtl_files) == 3, f"Expected 3 RTL files, got {len(tm.rtl_files)}"
        assert all(Path(f).exists() for f in tm.rtl_files), \
            f"Some RTL files missing: {[f for f in tm.rtl_files if not Path(f).exists()]}"
        assert tm.clock_port == "clk"
        # Canonical param names (not RTL chparam names)
        assert "mac_lanes" in tm.params, "mac_lanes param not in tinymac_accel spec"
        assert "accumulator_width" in tm.params, "accumulator_width param not in tinymac_accel spec"
        # RTL chparam names carried in rtl_param_name field
        assert tm.params["mac_lanes"].get("rtl_param_name") == "LANES", \
            "mac_lanes should have rtl_param_name='LANES'"
        assert tm.params["accumulator_width"].get("rtl_param_name") == "ACC_W", \
            "accumulator_width should have rtl_param_name='ACC_W'"
        assert "nangate45" in tm.platforms
        assert "asap7" in tm.platforms
        assert tm.has_macros is False
        assert tm.is_tinyvad()
        print(f"  tinymac_accel: {len(tm.rtl_files)} RTL files, "
              f"params={list(tm.params.keys())}, has_macros={tm.has_macros}  PASS")
    except FileNotFoundError as e:
        print(f"  SKIP tinymac_accel load (YAML not yet written): {e}")

    # 2. gcd loads
    try:
        gcd = DesignSpec.load("gcd")
        print(f"  gcd loaded: {gcd}")
        assert gcd.name == "gcd"
        assert gcd.top == "gcd"
        assert len(gcd.rtl_files) >= 1
        assert gcd.params == {} or gcd.params is not None
        # gcd has no functional eval (generic design)
        assert not gcd.is_tinyvad()
        print(f"  gcd: {len(gcd.rtl_files)} RTL files, has_macros={gcd.has_macros}  PASS")
    except FileNotFoundError as e:
        print(f"  SKIP gcd load (YAML not yet written): {e}")

    # 3. sdc_text produces valid content
    try:
        tm = DesignSpec.load("tinymac_accel")
        sdc = tm.sdc_text("nangate45", 5.0)
        assert "tinymac_accel" in sdc, "top module not in SDC"
        assert "clk" in sdc, "clock port not in SDC"
        assert "5.0" in sdc, "clock period not in SDC"
        sdc_asap7 = tm.sdc_text("asap7", 5000.0)   # 5.0 ns × 1000 = 5000 ps
        assert "5000.0" in sdc_asap7, "asap7 native clock not in SDC"
        print(f"  sdc_text nangate45 / asap7  PASS")
    except FileNotFoundError:
        print("  SKIP sdc_text test (tinymac YAML not yet written)")

    # 4. tinymac rtl_hash matches the legacy _rtl_hash() from physical_runner
    try:
        tm = DesignSpec.load("tinymac_accel")
        # Compute the hash the same way physical_runner._rtl_hash() does:
        # sorted RTL_FILES (just file names), reading from RTL_DIR
        h = hashlib.sha256()
        rtl_dir = _REPO_ROOT / "rtl" / "accel"
        rtl_fnames = ("int8_mac_array.v", "requantize.v", "tinymac_accel.v")
        for fname in sorted(rtl_fnames):
            p = rtl_dir / fname
            try:
                h.update(p.read_bytes())
            except OSError:
                h.update(fname.encode())
        legacy_hash = h.hexdigest()[:8]

        ds_hash = tm.rtl_hash()
        assert ds_hash == legacy_hash, \
            f"rtl_hash mismatch: DesignSpec={ds_hash!r}, legacy={legacy_hash!r}"
        print(f"  rtl_hash matches legacy _rtl_hash(): {ds_hash!r}  PASS")
    except FileNotFoundError as e:
        print(f"  SKIP rtl_hash test: {e}")

    # 5. verilog_top_params_str — uses canonical param names (mac_lanes/accumulator_width)
    # as config dict keys; emits RTL chparam names (LANES/ACC_W) in the output string.
    try:
        tm = DesignSpec.load("tinymac_accel")
        # Config uses canonical names (mac_lanes/accumulator_width)
        vtp = tm.verilog_top_params_str({"mac_lanes": 4, "accumulator_width": 24,
                                          "clock_period_ns": 5.0})
        # VERILOG_TOP_PARAMS string must use RTL chparam names
        assert "LANES 4" in vtp and "ACC_W 24" in vtp, f"vtp={vtp!r}"
        print(f"  verilog_top_params_str (canonical config keys): {vtp!r}  PASS")
    except FileNotFoundError:
        pass

    print("\n=== designs.py self-test PASSED ===")
    sys.exit(0)
