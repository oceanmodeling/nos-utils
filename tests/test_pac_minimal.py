"""STOFS-3D-PAC minimal run: new config keys keep other systems unchanged. MJ (11/16/26)"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("netCDF4")
from netCDF4 import Dataset  # noqa: E402

from nos_utils.config import ForcingConfig  # noqa: E402
from nos_utils.forcing import river_clim as rc  # noqa: E402
from nos_utils.forcing.rtofs import RTOFSProcessor  # noqa: E402
from nos_utils.forcing.tidal import TidalProcessor  # noqa: E402

OTHERS = ("secofs", "secofs_ufs", "stofs_3d_atl", "stofs_3d_atl_ufs", "stofs_2d_glo")


def _cfg(name, **kw):
    return getattr(ForcingConfig, "for_" + name)(pdy="20261001", cyc=12, **kw)


class TestDefaultsKeepOtherSystems:
    @pytest.mark.parametrize("name", OTHERS)
    def test_new_keys_are_inert(self, name):
        c = _cfg(name)
        assert c.obc_slot_discovery is False and c.obc_slot_records is False
        assert c.river_ob_clim_enabled is False
        assert c.river_ops_static_names is None and c.tide_fac_executables is None
        assert c.critical_sources is None

    def test_atl_river_fix_names_are_the_original_three(self):
        from nos_utils.forcing.nwm import NWMProcessor
        p = NWMProcessor.__new__(NWMProcessor)
        p.config = _cfg("stofs_3d_atl")
        assert p._ops_river_fix == {
            "source_sink.in": "stofs_3d_atl_river_source_sink.in",
            "msource.th": "stofs_3d_atl_river_msource.th",
            "vsink.th": "stofs_3d_atl_river_vsink.th"}
        p.config = _cfg("stofs_3d_atl", river_ops_static_names={"msource.th": "x.th"})
        assert p._ops_river_fix == {"msource.th": "x.th"}

    def test_atl_uses_the_ops_timeline_for_both_parts(self, tmp_path):
        p = RTOFSProcessor(_cfg("stofs_3d_atl"), tmp_path, tmp_path)
        assert p._slot_discovery and p._slot_records

    def test_secofs_style_stofs_mode_uses_neither(self, tmp_path):
        c = _cfg("stofs_3d_atl", obc_ops_timeline=False)
        p = RTOFSProcessor(c, tmp_path, tmp_path)
        assert not p._slot_discovery and not p._slot_records


def _yaml_cfg(tmp_path, forcing, system="x"):
    doc = {"system": {"name": system},
           "grid": {"domain": {"lon_min": 93.0, "lon_max": 289.0, "lat_min": -33.0, "lat_max": 66.0},
                    "n_levels": 84},
           "model": {"physics": {"nws": 2}, "run": {"nowcast_hours": 24, "forecast_hours": 48}},
           "forcing": forcing}
    f = tmp_path / "x.yaml"
    f.write_text(yaml.dump(doc))
    return ForcingConfig.from_yaml(f, pdy="20261001", cyc=12)


class TestPacificValues:
    def test_preset_and_ufs_preset(self):
        for name in ("stofs_3d_pac", "stofs_3d_pac_ufs"):
            c = _cfg(name)
            assert c.obc_slot_discovery and c.obc_slot_records and c.obc_ops_timeline is False
            assert c.river_ob_clim_enabled
            assert c.tide_fac_executables == ["stofs_3d_pac_tide_fac"]
            assert "NWM" not in c.critical_sources and "RIVER_OB_CLIM" in c.critical_sources

    def test_from_yaml_parses_the_new_keys(self, tmp_path):
        c = _yaml_cfg(tmp_path, {
            "ocean": {"obc": {"slot_discovery": True, "slot_records": True}},
            "river": {"ops_static_names": {"source_sink.in": "a"},
                      "ob_clim": {"enabled": True, "discharge_file": "q.txt"}},
            "tidal": {"tide_fac_executables": ["stofs_3d_pac_tide_fac"]}})
        assert c.obc_slot_discovery and c.obc_slot_records
        assert c.river_ops_static_names == {"source_sink.in": "a"}
        assert c.river_ob_clim_enabled and c.river_ob_clim_discharge_file == "q.txt"
        assert c.river_ob_clim_temperature_file == "river_clim_temperature.txt"
        assert c.tide_fac_executables == ["stofs_3d_pac_tide_fac"]

    def test_from_yaml_defaults_match_the_dataclass(self, tmp_path):
        c = _yaml_cfg(tmp_path, {"ocean": {"nudging": {"enabled": True}}})
        d = ForcingConfig(0, 1, 0, 1, '20261001', 12)
        for k in ("obc_slot_discovery", "obc_slot_records", "river_ob_clim_enabled",
                  "river_ob_clim_discharge_file", "river_ob_clim_temperature_file",
                  "river_ops_static_names", "tide_fac_executables"):
            assert getattr(c, k) == getattr(d, k)

    def test_from_yaml_rejects_bad_types(self, tmp_path):
        with pytest.raises(ValueError):
            _yaml_cfg(tmp_path, {"tidal": {"tide_fac_executables": "stofs_3d_pac_tide_fac"}})
        with pytest.raises(ValueError):
            _yaml_cfg(tmp_path, {"river": {"ops_static_names": ["a"]}})


class TestTideFacNames:
    def _run(self, tmp_path, monkeypatch, exe_name, cfg):
        exe_dir = tmp_path / "exec"
        exe_dir.mkdir(parents=True)
        exe = exe_dir / exe_name
        exe.write_text("#!/bin/sh\ncat > /dev/null\ncp bctides.in_template bctides.in\n")
        exe.chmod(0o755)
        for v in ("EXECnos", "EXECofs", "EXECstofs3d"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("EXECstofs3d", str(exe_dir))
        fix = tmp_path / "fix"
        fix.mkdir(parents=True)
        (fix / "sys.bctides.in_template").write_text("T" * 1200 + "\n")
        return TidalProcessor(cfg, fix, tmp_path / "w").process()

    def test_default_finds_the_atl_name_and_not_the_pacific_one(self, tmp_path, monkeypatch):
        cfg = _cfg("stofs_3d_atl", restart_min_bytes=100)
        assert self._run(tmp_path / "a", monkeypatch, "stofs_3d_atl_tide_fac", cfg).success
        res = self._run(tmp_path / "b", monkeypatch, "stofs_3d_pac_tide_fac", cfg)
        assert not res.success

    def test_configured_name_is_found(self, tmp_path, monkeypatch):
        cfg = _cfg("stofs_3d_atl", tide_fac_executables=["stofs_3d_pac_tide_fac"])
        assert self._run(tmp_path, monkeypatch, "stofs_3d_pac_tide_fac", cfg).success


def _tsuv_files(tmp_path):
    rng = np.random.default_rng(0)
    files = []
    for tag in ("n000", "f006"):
        p = tmp_path / f"rtofs_glo_3dz_{tag}_6hrly_hvr_US_east.nc"
        with Dataset(p, "w") as d:
            d.createDimension("MT", 1); d.createDimension("Depth", 3)
            d.createDimension("Y", 5); d.createDimension("X", 6)
            d.createVariable("Depth", "f4", ("Depth",))[:] = [0, 10, 20]
            d.createVariable("Longitude", "f4", ("Y", "X"))[:] = np.tile(np.arange(6.), (5, 1)) - 70
            d.createVariable("Latitude", "f4", ("Y", "X"))[:] = np.tile(np.arange(5.)[:, None], (1, 6)) + 30
            for v in ("temperature", "salinity", "u", "v"):
                d.createVariable(v, "f4", ("MT", "Depth", "Y", "X"))[:] = rng.random((1, 3, 5, 6))
        files.append(p)
    return files


class TestTsuvTimeAxis:
    @pytest.mark.parametrize("dt,name", [(None, "stofs_3d_atl"), (86400.0, "stofs_3d_atl")])
    def test_axis_follows_obc_dt_and_atl_stays_6_hourly(self, tmp_path, dt, name):
        cfg = _cfg(name)
        if dt:
            cfg.obc_dt = dt
        cfg.obc_roi_3d = {"x1": 0, "x2": 4, "y1": 0, "y2": 3}
        out = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")._stofs_prepare_tsuv(
            _tsuv_files(tmp_path), tmp_path)
        with Dataset(out) as d:
            np.testing.assert_array_equal(d["time"][:], np.arange(2) * (dt or 21600.0))
        assert ForcingConfig.for_stofs_3d_atl("20261001", 12).obc_dt == 21600.0


def _daily_tree(root, pdy, cyc_hours=("n024", "f024", "f048", "f072")):
    """Ops Pacific inputs: rtofs.PDY-1 n024, rtofs.PDY n024/f024/..., 2D n012/f012/f036/f060/f084."""
    def touch(d, name):
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"x")
    prev = (datetime.strptime(pdy, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    for v in "tsuv":
        touch(root / f"rtofs.{prev}", f"rtofs_glo_3dz_n024_daily_3z{v}io.nc")
        for tag in cyc_hours:
            touch(root / f"rtofs.{pdy}", f"rtofs_glo_3dz_{tag}_daily_3z{v}io.nc")
    for tag in ("n012", "f012", "f036", "f060", "f084"):
        touch(root / f"rtofs.{pdy}", f"rtofs_glo_2ds_{tag}_diag.nc")


class TestDailyDiscovery:
    def _proc(self, tmp_path, monkeypatch):
        cfg = _cfg("stofs_3d_pac")
        monkeypatch.setattr(RTOFSProcessor, "validate_file_size", lambda self, p, m=0: True)
        return RTOFSProcessor(cfg, tmp_path, tmp_path / "out")

    def test_pacific_picks_daily_groups_with_ops_off(self, tmp_path, monkeypatch):
        _daily_tree(tmp_path, "20261001")
        p = self._proc(tmp_path, monkeypatch)
        assert p.config.obc_ops_timeline is False
        f2, f3 = p.find_input_files_by_type()
        assert [x.name.split("_")[3] for x in f2][:4] == ["n012", "f012", "f036", "f060"]
        assert len(f3) == 20
        groups = rc_group(f3)
        assert [g[0].parent.name + "/" + g[0].name.split("_")[3] for g in groups] == [
            "rtofs.20260930/n024", "rtofs.20261001/n024",
            "rtofs.20261001/f024", "rtofs.20261001/f048", "rtofs.20261001/f072"]
        t0 = datetime(2026, 9, 30, 12)
        for k, g in enumerate(groups):
            assert {p.valid_time(f) for f in g} == {t0 + timedelta(days=k)}

    def test_atl_style_6hourly_selection_is_unchanged(self, tmp_path):
        d = tmp_path / "rtofs.20261001"
        d.mkdir()
        for tag in ("n012", "n018", "f006", "f012"):
            (d / f"rtofs_glo_2ds_{tag}_diag.nc").write_bytes(b"x")
        cfg = _cfg("stofs_3d_atl")
        p = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")
        p.validate_file_size = lambda path, m=0: True
        f2, _ = p.find_input_files_by_type()
        assert [x.name.split("_")[3] for x in f2] == ["n012", "n018", "f006", "f012"]


def rc_group(files):
    from nos_utils.forcing.rtofs import group_daily_3z
    return group_daily_3z(files)


FILE = ["# Seq a b c"] + ["%d %s" % (i, " ".join(str(-(i * 10 + j)) for j in range(1, 8)))
                          for i in range(1, 377)]


class TestOBClimRows:
    def test_normal_day_uses_ops_awk_window(self):
        rows = rc.ob_clim_rows(FILE, 100)
        assert [r[0] for r in rows] == ["99", "100", "101", "102"]

    def test_text_replaces_column_one_and_keeps_values_and_sign(self):
        txt = rc.ob_clim_text(FILE, 100).splitlines()
        assert txt[0].startswith("0.000 -991 ") and len(txt[0].split()) == 8
        assert [ln.split()[0] for ln in txt] == ["0.000", "86400.000", "172800.000", "259200.000"]
        assert txt[1].split()[1:] == FILE[100].split()[1:]
        assert all(float(v) < 0 for ln in txt for v in ln.split()[1:])

    def test_day_one_avoids_the_header(self, caplog):
        rows = rc.ob_clim_rows(FILE, 1)
        assert [r[0] for r in rows] == ["1", "2", "3", "4"]
        assert "header" in caplog.text

    def test_day_two_is_the_first_awk_window_without_header(self):
        assert [r[0] for r in rc.ob_clim_rows(FILE, 2)] == ["1", "2", "3", "4"]

    def test_leap_day_366(self):
        assert [r[0] for r in rc.ob_clim_rows(FILE, 366)] == ["365", "366", "367", "368"]

    def test_short_file_raises(self):
        with pytest.raises(ValueError):
            rc.ob_clim_rows(FILE[:5], 100)

    def test_processor_writes_both_files_from_the_nowcast_begin_day(self, tmp_path):
        fix = tmp_path / "fix"
        fix.mkdir(parents=True)
        (fix / "river_clim_discharge.txt").write_text("\n".join(FILE) + "\n")
        (fix / "river_clim_temperature.txt").write_text(
            "\n".join(["# Seq"] + ["%d %s" % (i, " ".join(["5.0"] * 7)) for i in range(1, 377)]) + "\n")
        cfg = _cfg("stofs_3d_pac")
        res = rc.RiverOBClimProcessor(cfg, fix, tmp_path / "out").process()
        assert res.success
        day0 = datetime(2026, 9, 30).timetuple().tm_yday
        flux = (tmp_path / "out" / "flux.th").read_text().splitlines()
        assert flux[0].split()[1] == str(-((day0 - 1) * 10 + 1))
        assert len((tmp_path / "out" / "TEM_1.th").read_text().splitlines()) == 4

    def test_missing_file_fails(self, tmp_path):
        res = rc.RiverOBClimProcessor(_cfg("stofs_3d_pac"), tmp_path, tmp_path / "out").process()
        assert not res.success

    def test_orchestrator_wiring(self, tmp_path):
        from nos_utils.orchestrator import PrepOrchestrator
        fix = tmp_path / "fix"
        fix.mkdir(parents=True)
        for n in ("river_clim_discharge.txt", "river_clim_temperature.txt"):
            (fix / n).write_text("\n".join(FILE) + "\n")
        orch = PrepOrchestrator(_cfg("stofs_3d_pac"), {"fix": str(fix)})
        out = tmp_path / "w"
        res = orch._run_river_ob_clim(out)
        assert res.success and res.source == "RIVER_OB_CLIM"
