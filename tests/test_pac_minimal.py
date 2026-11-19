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
            "river": {"ops_static_names": {"source_sink.in": "a", "msource.th": "b", "vsink.th": "c"},
                      "ob_clim": {"enabled": True, "discharge_file": "q.txt"}},
            "tidal": {"tide_fac_executables": ["stofs_3d_pac_tide_fac"]}})
        assert c.obc_slot_discovery and c.obc_slot_records
        assert c.river_ops_static_names == {"source_sink.in": "a", "msource.th": "b", "vsink.th": "c"}
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
        for tag in ("n018", "f006", "f012"):
            (d / f"rtofs_glo_3dz_{tag}_6hrly_hvr_US_east.nc").write_bytes(b"x")
        _, f3 = p.find_input_files_by_type()
        assert [x.name.split("_")[3] for x in f3] == ["n018", "f006", "f012"]
        assert p._per_time(f3) == f3


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

    def test_orchestrator_run_calls_the_stage(self, tmp_path):
        from unittest.mock import MagicMock
        from nos_utils.forcing.base import ForcingResult
        from nos_utils.orchestrator import PrepOrchestrator
        orch = PrepOrchestrator(_cfg("stofs_3d_pac"), {"fix": str(tmp_path), "output": str(tmp_path / "w")},
                                run_name="stofs_3d_pac", skip_legacy=True)
        orch._run_river_ob_clim = MagicMock(
            return_value=ForcingResult(success=True, source="RIVER_OB_CLIM"))
        res = orch.run("nowcast")
        orch._run_river_ob_clim.assert_called_once()
        assert "RIVER_OB_CLIM" in {r.source for r in res.results}

    def test_stage_writes_files_and_logs_inputs(self, tmp_path):
        from nos_utils.forcing._log import start_input_capture, drain_input_capture
        fix = tmp_path / "fix"
        fix.mkdir()
        for n in ("river_clim_discharge.txt", "river_clim_temperature.txt"):
            (fix / n).write_text("\n".join(FILE) + "\n")
        start_input_capture()
        res = rc.RiverOBClimProcessor(_cfg("stofs_3d_pac"), fix, tmp_path / "w").process()
        groups = drain_input_capture()
        assert res.success and "RIVER_OB_CLIM" in repr(groups)

    def test_both_flux_writers_enabled_is_an_error(self):
        with pytest.raises(ValueError, match="both write flux.th"):
            _cfg("stofs_3d_atl", river_ob_clim_enabled=True)


class TestSshLongitudeConvention:
    def _ssh1(self, tmp_path, cfg, lon1):
        nx, ny = len(lon1), 5
        f = tmp_path / "rtofs_glo_2ds_f012_diag.nc"
        with Dataset(str(f), "w") as d:
            d.createDimension("MT", 1); d.createDimension("Y", ny); d.createDimension("X", nx)
            d.createVariable("Longitude", "f4", ("Y", "X"))[:] = np.tile(lon1, (ny, 1))
            d.createVariable("Latitude", "f4", ("Y", "X"))[:] = np.tile(np.linspace(0, 40, ny)[:, None], (1, nx))
            d.createVariable("ssh", "f4", ("MT", "Y", "X"))[:] = 0.5
        cfg.obc_roi_2d = {"x1": 0, "x2": nx - 1, "y1": 0, "y2": ny - 1}
        p = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")
        return p, p._stofs_prepare_ssh([f], tmp_path)

    def test_pacific_keeps_0360_and_adt_bounds_overlap(self, tmp_path):
        from nos_utils.forcing.adt import ADTBlender
        lon1 = np.linspace(92.0, 284.0, 41)
        p, ssh1 = self._ssh1(tmp_path, _cfg("stofs_3d_pac"), lon1)
        with Dataset(str(ssh1)) as d:
            xl = d["xlon"][:]
        assert xl.min() > 90 and xl.max() < 290
        p._bnd_lons = np.array([100.0, 179.0, 181.0, 250.0])
        p._bnd_lats = np.full(4, 20.0)
        *_, inside = p._ops_weights(xl, np.tile(np.linspace(0, 40, 5)[:, None], (1, xl.shape[1])))
        assert all(inside)
        lo, hi, _, _ = ADTBlender(_cfg("stofs_3d_pac"), tmp_path)._dest_bounds(ssh1)
        assert lo < 100.0 and hi > 280.0

    def test_atl_ssh1_is_still_lon_minus_360(self, tmp_path):
        lon1 = np.linspace(285.0, 310.0, 11)
        _, ssh1 = self._ssh1(tmp_path, _cfg("stofs_3d_atl"), lon1)
        with Dataset(str(ssh1)) as d:
            np.testing.assert_array_equal(d["xlon"][0, :], (lon1 - 360).astype(np.float32))


class TestQcAndRecordCounts:
    def test_pacific_three_record_nowcast_is_not_short(self, tmp_path):
        from nos_utils.orchestrator import PrepOrchestrator
        cfg = _cfg("stofs_3d_pac")
        for name in ("elev2D.th.nc", "TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc"):
            with Dataset(tmp_path / name, "w") as d:
                d.createDimension("time", None)
                d.createVariable("time", "f8", ("time",))[:] = [0, 86400, 172800]
        orch = PrepOrchestrator(cfg, {"fix": str(tmp_path)}, run_name="stofs_3d_pac")
        assert orch._qc_obc_dimensions(tmp_path, {"2d": 5, "3d": 5}, phase="nowcast") is None
        res = orch._qc_obc_dimensions(tmp_path, {"2d": 5, "3d": 1}, phase="nowcast")
        assert res is not None and any("TEM_3D" in w for w in res.warnings)

    def test_source_records_count_daily_groups(self, tmp_path, monkeypatch):
        _daily_tree(tmp_path, "20261001")
        monkeypatch.setattr(RTOFSProcessor, "validate_file_size", lambda self, p, m=0: True)
        p = RTOFSProcessor(_cfg("stofs_3d_pac"), tmp_path, tmp_path / "out")
        _, f3 = p.find_input_files_by_type()
        assert len(f3) == 20 and len(p._per_time(f3)) == 5


class TestIncompleteDailyGroup:
    def test_missing_u_leaves_the_slot_unfilled_and_does_not_crash(self, tmp_path, monkeypatch, caplog):
        _daily_tree(tmp_path, "20261001")
        (tmp_path / "rtofs.20261001" / "rtofs_glo_3dz_f024_daily_3zuio.nc").unlink()
        monkeypatch.setattr(RTOFSProcessor, "validate_file_size", lambda self, p, m=0: True)
        p = RTOFSProcessor(_cfg("stofs_3d_pac"), tmp_path, tmp_path / "out")
        _, f3 = p.find_input_files_by_type()
        assert len(f3) == 16 and all("_f024_" not in f.name for f in f3)
        assert "lacks ['u']" in caplog.text

    def test_tsuv_skips_a_group_without_u(self, tmp_path):
        files = []
        for tag, vars_ in (("n024", "tsuv"), ("f024", "tsv")):
            for v in vars_:
                name = {"t": "temperature", "s": "salinity", "u": "u", "v": "v"}[v]
                f = tmp_path / f"rtofs_glo_3dz_{tag}_daily_3z{v}io.nc"
                with Dataset(str(f), "w") as d:
                    d.createDimension("MT", 1); d.createDimension("Depth", 2)
                    d.createDimension("Y", 3); d.createDimension("X", 4)
                    d.createVariable("Depth", "f4", ("Depth",))[:] = [0, 10]
                    d.createVariable("Longitude", "f4", ("Y", "X"))[:] = 150.0
                    d.createVariable("Latitude", "f4", ("Y", "X"))[:] = 10.0
                    d.createVariable(name, "f4", ("MT", "Depth", "Y", "X"))[:] = 1.0
                files.append(f)
        cfg = _cfg("stofs_3d_pac")
        cfg.obc_roi_3d = {"x1": 0, "x2": 3, "y1": 0, "y2": 2}
        out = RTOFSProcessor(cfg, tmp_path, tmp_path / "o")._stofs_prepare_tsuv(sorted(files), tmp_path)
        with Dataset(out) as d:
            assert d.dimensions["time"].size == 1


class TestSlotWritersSmoothing:
    def _proc(self, tmp_path, phase, smooth):
        cfg = _cfg("stofs_3d_pac", obc_time_smoothing=smooth)
        out = tmp_path / "out"
        out.mkdir(parents=True, exist_ok=True)
        p = RTOFSProcessor(cfg, tmp_path, out, phase=phase)
        p._bnd_lons = np.array([150.0, 160.0]); p._bnd_lats = np.array([0.0, 1.0])
        p._rtofs_cycle_date = datetime(2026, 10, 1)
        return cfg, p

    @pytest.mark.parametrize("phase,n", [("nowcast", 3), ("forecast", 4)])
    def test_elev2d_keeps_the_phase_count_and_averages_neighbours(self, tmp_path, phase, n):
        cfg, p = self._proc(tmp_path, phase, True)
        t0, _, dur = p._get_output_window(phase)
        assert int(np.ceil(dur / cfg.obc_dt)) + 1 == n
        files = [tmp_path / f"f{k}.nc" for k in range(6)]
        for k, f in enumerate(files):
            p._valid_times[f] = t0 + timedelta(days=k)
        ssh = np.array([[k, 10.0 * k] for k in range(6)], np.float32)
        out = p._write_elev2d_ops(files, ssh, hold=False)
        with Dataset(out) as d:
            ts = d["time_series"][:, :, 0, 0]
            assert ts.shape[0] == n
            np.testing.assert_allclose(ts[:, 0], [k + 0.5 for k in range(n)], atol=1e-5)
            np.testing.assert_array_equal(d["time"][:], np.arange(n) * cfg.obc_dt)
        cfg2, p2 = self._proc(tmp_path / "b", phase, False)
        for k, f in enumerate(files):
            p2._valid_times[f] = t0 + timedelta(days=k)
        with Dataset(p2._write_elev2d_ops(files, ssh, hold=False)) as d:
            np.testing.assert_allclose(d["time_series"][:, 0, 0, 0], np.arange(n), atol=1e-5)

    def test_3d_slot_writer_averages_t_s_uv(self, tmp_path):
        cfg, p = self._proc(tmp_path, "nowcast", True)
        t0, _, _ = p._get_output_window("nowcast")
        files = [tmp_path / f"g{k}.nc" for k in range(6)]
        for k, f in enumerate(files):
            p._valid_times[f] = t0 + timedelta(days=k)
        arrs = [[np.full((2, 3), float(k), np.float32) for k in range(6)] for _ in range(4)]
        out = p._write_3d_ops(files, *arrs)
        assert len(out) == 3
        with Dataset(out[0]) as d:
            ts = d["time_series"][:]
            assert ts.shape[0] == 3
            np.testing.assert_allclose(ts[:, 0, 0, 0], [0.5, 1.5, 2.5], atol=1e-5)


class TestDailyFallbackFailsLoudly:
    def test_no_tsuv_path_returns_no_files(self, tmp_path, caplog):
        p = RTOFSProcessor(_cfg("stofs_3d_pac"), tmp_path, tmp_path / "out")
        p._bnd_lons = np.array([150.0]); p._bnd_lats = np.array([0.0])
        files = [tmp_path / "rtofs_glo_3dz_n024_daily_3ztio.nc"]
        assert p._process_3d(files) == []
        assert "per-file fallback" in caplog.text


class TestConfigValidation:
    def test_static_names_must_be_the_three_keys(self, tmp_path):
        with pytest.raises(ValueError, match="source_sink.in, msource.th and vsink.th"):
            _yaml_cfg(tmp_path, {"river": {"ops_static_names": {"source_sink.in": "a"}}})

    def test_null_numeric_yaml_values_use_defaults(self, tmp_path):
        c = _yaml_cfg(tmp_path, {"ocean": {"obc": {"dt": None}, "nudging": {"dt": None},
                                           "adt": {"datum_offset": None}}})
        assert c.obc_dt == 21600.0 and c.nudging_dt == 10800.0 and c.adt_datum_offset == 0.0

    def test_pacific_ufs_forecast_hours(self):
        assert _cfg("stofs_3d_pac_ufs").ufs_nhours_fcst == 72

    def test_missing_fix_error_is_system_neutral_for_custom_names(self, tmp_path):
        from nos_utils.forcing.nwm import NWMProcessor
        p = NWMProcessor.__new__(NWMProcessor)
        p.config = _cfg("stofs_3d_atl", river_ops_static_names={
            "source_sink.in": "a", "msource.th": "b", "vsink.th": "c"})
        p._fix_search_dirs = lambda: [tmp_path]
        with pytest.raises(FileNotFoundError) as e:
            p._stage_ops_river_files()
        assert "ATL" not in str(e.value) and "fetch_stofs" not in str(e.value)


class _NudgeBase:
    pass


def _nudge_counts(tmp_path, phase, smooth):
    from unittest.mock import patch
    from tests.test_nudging import TestNudgingWithSyntheticRTOFS as T
    from nos_utils.forcing.nudging import NudgingProcessor
    t = T()
    n = 5
    lons = np.linspace(-77.0, -73.0, n); lats = np.linspace(30.0, 34.0, n)
    t._make_gr3(tmp_path / "n.gr3", n, lons, lats, np.ones(n))
    t._make_grid(tmp_path / "hgrid.ll", n, lons, lats, np.full(n, 100.0))
    t._make_vgrid_simple(tmp_path / "vgrid.in", nvrt=10)
    d = tmp_path / "rtofs.20260331"
    d.mkdir()
    for fhr in range(12, 133, 6):
        t._make_rtofs_3d(d / f"rtofs_glo_3dz_f{fhr:03d}_6hrly_hvr_US_east.nc", 1, 5, (-80, -70), (28, 36), 20)
    cfg = ForcingConfig(lon_min=-80.0, lon_max=-70.0, lat_min=28.0, lat_max=36.0, pdy="20260401", cyc=12,
                        nowcast_hours=24, forecast_hours=48, nudging_enabled=True, n_levels=10,
                        grid_file=tmp_path / "hgrid.ll", nudging_dt=86400.0, nudging_time_smoothing=smooth)
    proc = NudgingProcessor(cfg, tmp_path, tmp_path / "o", nudge_weight_file=tmp_path / "n.gr3",
                            rtofs_input_path=tmp_path, phase=phase)
    from nos_utils.io.schism_vgrid import SchismVgrid
    with patch.object(proc, "_load_vgrid", return_value=SchismVgrid.read(tmp_path / "vgrid.in")), \
            patch("nos_utils.forcing.rtofs.RTOFSProcessor.validate_file_size", return_value=True):
        res = proc.process()
    assert res.success, res.errors
    with Dataset(tmp_path / "o" / "TEM_nu.nc") as ds:
        return ds.dimensions["time"].size


class TestNudgingSmoothedCounts:
    @pytest.mark.parametrize("phase,n", [("nowcast", 2), ("forecast", 3)])
    def test_daily_smoothed_records_match_what_the_run_needs(self, tmp_path, phase, n):
        assert _nudge_counts(tmp_path, phase, True) == n

    @pytest.mark.parametrize("phase,n", [("nowcast", 2), ("forecast", 3)])
    def test_unsmoothed_counts_are_unchanged(self, tmp_path, phase, n):
        assert _nudge_counts(tmp_path, phase, False) == n


class TestFollowUps:
    def test_non_daily_tsuv_does_not_silently_skip_a_group(self, tmp_path, caplog):
        files = _tsuv_files(tmp_path)
        bad = tmp_path / "rtofs_glo_3dz_f012_6hrly_hvr_US_east.nc"
        with Dataset(bad, "w") as d:
            d.createDimension("MT", 1); d.createDimension("Depth", 3)
            d.createDimension("Y", 5); d.createDimension("X", 6)
            d.createVariable("Depth", "f4", ("Depth",))[:] = [0, 10, 20]
            d.createVariable("Longitude", "f4", ("Y", "X"))[:] = np.tile(np.arange(6.), (5, 1)) - 70
            d.createVariable("Latitude", "f4", ("Y", "X"))[:] = np.tile(np.arange(5.)[:, None], (1, 6)) + 30
            for v in ("temperature", "salinity", "v"):
                d.createVariable(v, "f4", ("MT", "Depth", "Y", "X"))[:] = 1.0
        cfg = _cfg("stofs_3d_atl")
        cfg.obc_roi_3d = {"x1": 0, "x2": 4, "y1": 0, "y2": 3}
        proc = RTOFSProcessor(cfg, tmp_path, tmp_path / "out")
        with pytest.raises(IndexError):
            proc._stofs_prepare_tsuv([files[0], bad, files[1]], tmp_path)
        assert "skipped" not in caplog.text

    def test_daily_fallback_removes_stale_files_and_prep_fails(self, tmp_path):
        p = RTOFSProcessor(_cfg("stofs_3d_pac"), tmp_path, tmp_path / "out")
        p.output_path.mkdir(parents=True)
        for n in ("TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc"):
            (p.output_path / n).write_bytes(b"old")
        p._bnd_lons = np.array([150.0]); p._bnd_lats = np.array([0.0])
        assert p._process_3d([tmp_path / "rtofs_glo_3dz_n024_daily_3ztio.nc"]) == []
        assert not [n for n in ("TEM_3D.th.nc", "SAL_3D.th.nc", "uv3D.th.nc") if (p.output_path / n).exists()]

    def test_required_set_check_covers_slot_records(self):
        import inspect
        src = inspect.getsource(RTOFSProcessor._process_stofs)
        assert "(region or self._slot_records) and not (elev2d_ok and obc3d_ok)" in src

    @pytest.mark.parametrize("slot,n_expected", [(True, 5), (False, 4)])
    def test_fallback_elev2d_smoothing_skipped_for_slot_records(self, tmp_path, slot, n_expected):
        from tests.test_obc_ops_timeline import _ops_proc
        cfg, proc, files = _ops_proc(tmp_path, "nowcast", obc_time_smoothing=True,
                                     obc_ops_timeline=False, obc_slot_records=slot)
        proc._ssh1_path = None
        out = proc._process_2d(files)
        with Dataset(out) as d:
            n_on = d["time_series"].shape[0]
        cfg.obc_time_smoothing = False
        out = proc._process_2d(files)
        with Dataset(out) as d:
            n_off = d["time_series"].shape[0]
        assert n_on == (n_off if slot else n_off - 1)
